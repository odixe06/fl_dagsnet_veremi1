"""One persistent process per GPU.

Each worker holds its own resident copy of the whole training matrix and of its
disjoint test shard, and a compiled DAGSNet. Clients are handed out dynamically
from a shared queue, so a slow client never strands a GPU; the parent -- not the
completion order -- fixes the aggregation order.

Different clients are different models. Nothing here forms a process group and no
gradient is ever synchronized between GPUs.
"""
import math
import time
import traceback

import numpy as np
import torch

import flatpack
from afpha import ADAM_BETAS, ADAM_EPS, ADAM_WEIGHT_DECAY, GRAD_CLIP, client_shuffle_seed
from dagsnet import build_dagsnet

COMPILE_FALLBACK = ["reduce-overhead", "default", "eager"]


def _to_device_chunked(arr, device, chunk=4_000_000):
    """Copy a mmapped array to the GPU without materializing a host copy."""
    out = torch.empty(arr.shape, dtype=torch.float16 if arr.dtype == np.float16
                      else torch.uint8, device=device)
    for i in range(0, len(arr), chunk):
        out[i:i + chunk] = torch.from_numpy(np.ascontiguousarray(arr[i:i + chunk])).to(
            device, non_blocking=True)
    return out


class Trainer:
    """Model, compiled graph and resident tensors for one GPU."""

    def __init__(self, device, batch, compile_mode, log):
        self.device = device
        self.batch = batch
        self.log = log
        self.model = build_dagsnet().to(device)
        self.params = list(self.model.parameters())
        self.eager = self.model
        self.compiled = self.model
        self.compile_mode = "eager"
        self.crit = torch.nn.CrossEntropyLoss()
        self.template = flatpack.Template(self.eager)
        # Skipped AMP steps are counted by watching the scale tensor without a host sync
        # (get_scale() would .item() every step). _scale is created lazily on the first
        # scale() call, so probe it that way; fail loudly if a torch upgrade removes it.
        probe = torch.amp.GradScaler("cuda")
        probe.scale(torch.zeros(1, device=device))
        assert isinstance(getattr(probe, "_scale", None), torch.Tensor), \
            "GradScaler._scale is unavailable; skipped-step counting needs a new mechanism"
        del probe
        if compile_mode != "eager":
            self._try_compile(compile_mode)

    def _try_compile(self, requested):
        """Compile, then prove it with a real step and a logit comparison.

        torch.compile is lazy, so a bare try around the call catches nothing --
        the trial forward/backward below is what actually validates the backend.
        The logit check runs in eval() so Dropout cannot make two forwards differ,
        and the trial steps' BatchNorm drift is undone from a snapshot afterwards.
        """
        order = COMPILE_FALLBACK[COMPILE_FALLBACK.index(requested):]
        snapshot = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        x = torch.randn(self.batch, 66, device=self.device, dtype=torch.float16)
        y = torch.randint(0, 16, (self.batch,), device=self.device)

        def restore():
            self.model.load_state_dict(snapshot, strict=True)
            self.model.zero_grad(set_to_none=True)
            self.model.train()

        self.model.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            ref = self.eager(x).float().clone()
        self.model.train()

        for mode in order:
            if mode == "eager":
                break
            try:
                cand = torch.compile(self.model, mode=None if mode == "default" else mode)
                # Mirror the real step exactly, zero_grad included. Consecutive backwards
                # without it accumulate into CUDA-graph-owned .grad buffers and raise
                # "gradient tensor output of CUDAGraphs ... overwritten by a subsequent run".
                opt = torch.optim.Adam(self.params, lr=1e-8, fused=True)
                sc = torch.amp.GradScaler("cuda")
                for _ in range(4):
                    with torch.autocast("cuda", dtype=torch.float16):
                        sc.scale(self.crit(cand(x), y)).backward()
                    sc.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                    sc.step(opt); sc.update()
                    opt.zero_grad(set_to_none=True)
                del opt, sc
                restore()

                self.model.eval()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    got = cand(x).float().clone()
                self.model.train()
                delta = (got - ref).abs().max().item()
                # Argmax on a freshly initialized model is decided by near-ties well below
                # fp16 resolution, so compare only rows whose top-2 gap is clearly larger
                # than the observed numerical difference.
                top2 = ref.topk(2, dim=1).values
                decisive = (top2[:, 0] - top2[:, 1]) > max(10 * delta, 1e-3)
                # Count disagreements as an integer. torch.mean on CUDA multiplies by a
                # rounded reciprocal, so a perfect match returns 0.99999994 for many row
                # counts (77 of the first 600) and a `< 1.0` test rejects a correct build
                # at random -- silently costing the CUDA-graph speedup.
                n_dec = int(decisive.sum())
                n_bad = int((got.argmax(1) != ref.argmax(1))[decisive].sum())
                if not math.isfinite(delta) or delta > 1e-2 or n_bad:
                    raise RuntimeError(
                        f"compiled logits diverge: max|d|={delta}, "
                        f"{n_bad} argmax disagreements over {n_dec}/{len(ref)} decisive rows")
                restore()
                self.compiled, self.compile_mode = cand, mode
                self.log(f"    compile mode={mode} ok (max|dlogit|={delta:.2e}, "
                         f"0 argmax disagreements on {n_dec}/{len(ref)} decisive rows)")
                return
            except Exception as exc:                       # noqa: BLE001 - fall back on anything
                self.log(f"    compile mode={mode} rejected: {type(exc).__name__}: {exc}")
                restore()
        self.compiled, self.compile_mode = self.eager, "eager"
        self.log("    running eager")

    def train_client(self, X, Y, lo, hi, rnd, cid, lr, mu, w_global):
        """One full local epoch over rows [lo, hi). Returns (state_dict, stats)."""
        t0 = time.perf_counter()
        model = self.model
        model.train()
        opt = torch.optim.Adam(self.params, lr=lr, betas=ADAM_BETAS, eps=ADAM_EPS,
                               weight_decay=ADAM_WEIGHT_DECAY, fused=True)
        scaler = torch.amp.GradScaler("cuda")

        seed = client_shuffle_seed(rnd, cid)
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        perm = lo + torch.randperm(hi - lo, generator=g, device=self.device)
        # Dropout draws from the global generators, not from `g`. Without seeding them
        # per (round, client) the mask sequence depends on which GPU picked the client
        # up and on how many clients ran before it, so a rescheduled run would differ
        # for a reason that has nothing to do with the algorithm.
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        n = hi - lo
        full = (n // self.batch) * self.batch          # graph-captured steps
        steps = 0
        # Accumulate on the device and read once at the end. Two float() calls per step
        # cost a measured 13% (8.12 vs 7.20 ms/step) by forcing extra host syncs.
        # Accumulate only over APPLIED steps. A step the GradScaler skipped overflowed,
        # so its gradient norm is inf by construction and its loss may be nan; folding
        # those into the mean turns a normal AMP warm-up into a fake divergence signal.
        ce_acc = torch.zeros((), device=self.device)
        gn_acc = torch.zeros((), device=self.device)
        skip_acc = torch.zeros((), device=self.device)
        zero = torch.zeros((), device=self.device)

        for i in range(0, n, self.batch):
            idx = perm[i:i + self.batch]
            # The tail batch is a real optimizer step and must not be dropped, but its
            # shape would force a recompile -- run that one step on the eager module.
            net = self.compiled if i < full else self.eager
            with torch.autocast("cuda", dtype=torch.float16):
                loss = self.crit(net(X[idx]), Y[idx])
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # d/dw of (mu/2)||w - w_global||^2 == mu (w - w_global); added after unscale so
            # it is in true gradient units, and before clipping so the clip sees both terms.
            # rebuilt every step: zero_grad(set_to_none=True) replaces the .grad objects
            torch._foreach_add_([p.grad for p in self.params],
                                torch._foreach_sub(self.params, w_global), alpha=mu)
            gn = torch.nn.utils.clip_grad_norm_(self.params, GRAD_CLIP)
            prev = scaler._scale.clone()
            scaler.step(opt)
            scaler.update()
            applied = scaler._scale >= prev         # a halved scale means the step was skipped
            skip_acc += ~applied
            opt.zero_grad(set_to_none=True)
            # torch.where, not multiplication: inf * 0 is nan.
            ce_acc += torch.where(applied, loss.detach(), zero)
            gn_acc += torch.where(applied, gn, zero)
            steps += 1

        ce_sum, gn_sum, skipped = float(ce_acc), float(gn_acc), int(skip_acc)
        applied_steps = steps - skipped
        ce_mean = ce_sum / max(applied_steps, 1)
        gn_mean = gn_sum / max(applied_steps, 1)

        vec_f, vec_i = self.template.flatten(model.state_dict())
        finite = bool(torch.isfinite(vec_f).all())
        stats = {"client_id": cid, "rows": int(n), "steps": steps, "skipped": skipped,
                 "applied_steps": applied_steps,
                 "skip_pct": 100.0 * skipped / max(steps, 1),
                 # means over applied steps only; see the accumulator comment above
                 "ce_mean": ce_mean, "grad_norm_mean": gn_mean,
                 "lr": lr, "mu": mu, "seed": seed,
                 # Three separate health facts. Finite weights alone do not mean the
                 # client trained: every step can be skipped and still return the
                 # global weights unchanged and finite.
                 "finite_weights": finite,
                 "finite_loss": math.isfinite(ce_mean),
                 "finite_grad": math.isfinite(gn_mean),
                 "seconds": time.perf_counter() - t0, "device": str(self.device)}
        return vec_f, vec_i, stats

    @torch.inference_mode()
    def evaluate(self, X, Y, num_classes, eval_batch):
        """Confusion counts and predictions for this worker's contiguous test shard."""
        t0 = time.perf_counter()
        self.model.eval()
        conf = torch.zeros(num_classes * num_classes, dtype=torch.int64, device=self.device)
        preds = torch.empty(len(X), dtype=torch.uint8, device=self.device)
        full = (len(X) // eval_batch) * eval_batch
        for i in range(0, len(X), eval_batch):
            net = self.compiled if i < full else self.eager
            with torch.autocast("cuda", dtype=torch.float16):
                logits = net(X[i:i + eval_batch])
            p = logits.float().argmax(1)               # graph output: consumed immediately
            preds[i:i + eval_batch] = p.to(torch.uint8)
            conf += torch.bincount(Y[i:i + eval_batch].long() * num_classes + p,
                                   minlength=num_classes * num_classes)
        self.model.train()
        return (conf.view(num_classes, num_classes).cpu().numpy(),
                preds.cpu().numpy(), time.perf_counter() - t0)


def worker_main(rank, dev_index, cfg, paths, shard, ctrl_q, task_q, res_q):
    """Process entry point. Never raises past the reporting boundary."""
    def log(msg):
        print(f"[gpu{rank}] {msg}", flush=True)

    try:
        torch.set_num_threads(1)
        torch.backends.cudnn.benchmark = True
        device = torch.device(f"cuda:{dev_index}")
        torch.cuda.set_device(device)
        log(f"{torch.cuda.get_device_name(device)} cap={torch.cuda.get_device_capability(device)}")

        Xm = np.load(paths["train_X"], mmap_mode="r")
        Ym = np.load(paths["train_y"], mmap_mode="r")
        Xt = np.load(paths["test_X"], mmap_mode="r")
        Yt = np.load(paths["test_y"], mmap_mode="r")
        t0 = time.perf_counter()
        X = _to_device_chunked(Xm, device)
        Y = _to_device_chunked(Ym, device).long()
        lo, hi = shard
        Xs = _to_device_chunked(Xt[lo:hi], device)
        Ys = _to_device_chunked(Yt[lo:hi], device).long()
        log(f"resident: train {X.shape} + test shard {Xs.shape} in {time.perf_counter()-t0:.0f}s, "
            f"{torch.cuda.memory_allocated(device)/2**30:.2f} GiB allocated")

        tr = Trainer(device, cfg["batch"], cfg["compile_mode"], log)
        free, total = torch.cuda.mem_get_info(device)
        res_q.put(("ready", rank, None,
                   {"compile_mode": tr.compile_mode,
                    "device_name": torch.cuda.get_device_name(device),
                    "capability": list(torch.cuda.get_device_capability(device)),
                    "free_GiB": free / 2**30, "total_GiB": total / 2**30,
                    "resident_GiB": torch.cuda.memory_allocated(device) / 2**30}))

        while True:
            msg = ctrl_q.get()
            kind = msg[0]
            if kind == "stop":
                break
            if kind == "round":
                _, rnd, lr, mus, gf, gi = msg
                global_sd = {k: v.to(device) for k, v in tr.template.unflatten(gf, gi).items()}
                tr.model.load_state_dict(global_sd, strict=True)
                w_global = [p.detach().clone() for p in tr.params]
                for cid in iter(task_q.get, None):
                    tr.model.load_state_dict(global_sd, strict=True)
                    vec_f, vec_i, stats = tr.train_client(
                        X, Y, *cfg["spans"][cid], rnd, cid, lr, mus[cid], w_global)
                    res_q.put(("client", rank, (vec_f, vec_i), stats))
                res_q.put(("round_done", rank, None, None))
            elif kind == "eval":
                _, rnd, gf, gi = msg
                tr.model.load_state_dict(
                    {k: v.to(device) for k, v in tr.template.unflatten(gf, gi).items()},
                    strict=True)
                conf, preds, secs = tr.evaluate(Xs, Ys, cfg["num_classes"], cfg["eval_batch"])
                res_q.put(("eval", rank, conf, {"round": rnd, "seconds": secs,
                                                "shard": [lo, hi]}))
                res_q.put(("preds", rank, preds, None))
            elif kind == "peak":
                res_q.put(("peak", rank, None,
                           {"peak_GiB": torch.cuda.max_memory_allocated(device) / 2**30}))
    except Exception:                                   # noqa: BLE001 - report, never hang
        res_q.put(("error", rank, None, {"traceback": traceback.format_exc()}))
        raise
