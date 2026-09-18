"""One persistent process per GPU, owning a fixed set of clients for the whole run.

Why static ownership rather than a dynamic client queue: client sizes are known from the
partition sidecar before the run, so a longest-processing-time split is already balanced to
within 2% (measured: 1.96% / 0.03% / 0.11% for 20 / 50 / 100 clients). Static ownership then
buys three things a dynamic scheduler cannot:

* each worker holds only ITS clients' rows on the GPU (~2.6 GiB fp16 instead of 5.29 GiB);
* per-client AdamW state and GradScaler state never cross a process boundary;
* the eval split across GPUs is exactly the training split, so both phases stay balanced.

One nn.Module per worker, not one per client. `torch.compile(mode="reduce-overhead")` captures
a CUDA graph over the forward/backward, and that graph is only valid while parameter storage
addresses are stable -- so client weights are swapped in with `copy_`, never by rebuilding the
module. Each client gets its own AdamW object over the SAME parameter tensors, which keeps
per-client moment estimates independent without touching those addresses.

Different clients are different models. They never form a process group and never all-reduce
with each other.
"""
from __future__ import annotations

import math
import os
import time
import traceback

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from . import ckpt as C
from .data import decode_clients, decode_test
from .evaluate import VmapEvaluator, eval_sequential, fold_bn, make_meta_base
from .model import FlatPacker, build_model
from .protos import ProtoAccumulator, ProtoRegularizer


def client_seed(seed: int, rnd: int, cid: int) -> int:
    """A pure function of (seed, round, client) so scheduling order can never leak into the RNG."""
    return (seed * 1_000_003 + rnd * 10_007 + cid * 97) % (2 ** 31 - 1)


def probe_verdict(ref, got, *, min_decisive_frac, max_delta=None, decisive_margin=0.01,
                  label="compile"):
    """Decide whether a probe proves the compiled path agrees with the eager one.

    Pure and CPU-testable on purpose: the gate it implements is the only thing standing between
    a silently wrong kernel and a 34-hour run, so it must be exercisable by a test that feeds it
    deliberately wrong tensors -- not only by a GPU probe that has to pass to be observed.

    Order matters. The decisive-margin threshold scales with the measured delta, so a large
    delta widens the margin until NO row qualifies and the argmax test then passes vacuously on
    an empty set. The ceiling and the coverage floor are checked first to close that hole.
    """
    delta = float((ref - got).abs().max())
    if not torch.isfinite(got).all():
        raise RuntimeError(f"{label}: logits are non-finite")
    if max_delta is not None and delta > max_delta:
        raise RuntimeError(f"{label}: logit delta {delta:.4g} exceeds the stated ceiling "
                           f"{max_delta}; precision granularity does not explain this")
    # The margin is FIXED, not a multiple of the measured delta. With `margin > 10*delta` the
    # disagreement test below could never fire: flipping an argmax needs a perturbation of at
    # least margin/2, so delta >= margin/2, and `margin > 10*delta >= 5*margin` is impossible.
    # The old rule therefore certified argmax stability on a set that excluded every flip.
    top2 = ref.topk(2, dim=1).values
    decisive = (top2[:, 0] - top2[:, 1]) > decisive_margin
    frac = float(decisive.float().mean())
    if frac < min_decisive_frac:
        raise RuntimeError(f"{label}: probe inconclusive, only {frac:.1%} of rows decisive "
                           f"(need {min_decisive_frac:.0%}); zero disagreements would prove nothing")
    bad = int(((got.argmax(1) != ref.argmax(1)) & decisive).sum())
    if bad:
        raise RuntimeError(f"{label}: disagrees on {bad} decisive rows")
    return {"max_abs_dlogit": delta, "decisive_rows": int(decisive.sum()),
            "disagreements": bad, "decisive_frac": round(frac, 4)}


class Worker:
    def __init__(self, rank: int, cfg: dict, cids: list[int], log=print):
        self.rank = rank
        self.cfg = cfg
        self.cids = list(cids)
        self.log = log
        # devices[rank] so a 1-GPU laptop can exercise the real 2-worker topology by mapping
        # both workers onto cuda:0. Kaggle passes ["cuda:0", "cuda:1"].
        self.dev = torch.device(cfg.get("devices", [f"cuda:{r}" for r in range(8)])[rank])
        if self.dev.type == "cuda":
            torch.cuda.set_device(self.dev)
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(1)

        self.C = cfg["num_classes"]
        self.d = cfg["feature_dim"]
        self.s = cfg["cps_s"]
        self.amp = bool(cfg["amp"]) and self.dev.type == "cuda"
        self.batch = int(cfg["batch"])
        self.eval_batch = int(cfg["eval_batch"])

    # -- setup --------------------------------------------------------------
    def load_data(self, client_root, test_dir, features, scaler, train_index=None,
                  eval_npy=None):
        t0 = time.time()
        X, y, spans, audit_tr = decode_clients(client_root, self.cids, features,
                                               progress=lambda m: self.log(f"[r{self.rank}]{m}"))
        self.Xg = torch.from_numpy(X).to(self.dev)
        self.Yg = torch.from_numpy(y).to(self.dev).long()
        del X, y
        self.spans = spans
        # optional per-client row subset (the mu sweep trains on a 98% slice; production is None)
        self.train_idx = None
        if train_index is not None:
            from .data import worker_train_indices
            indices = worker_train_indices(train_index, spans)
            self.train_idx = {cid: torch.from_numpy(indices[cid]).to(self.dev)
                              for cid in self.cids}

        if eval_npy is not None:
            # The mu sweep scores on a validation set carved out of TRAIN, prepared once by the
            # notebook and shared by both workers. Every client is still scored on the SAME
            # rows, so per-client metrics stay comparable exactly as with the global test set.
            Xt = np.load(eval_npy["X"], mmap_mode="c")
            yt = np.load(eval_npy["y"])
            audit_te = {"rows": int(len(yt)), "source": str(eval_npy["X"]),
                        "abs_max": float(np.abs(np.asarray(Xt[:100_000], np.float32)).max()),
                        "class_counts": np.bincount(yt.astype(np.int64), minlength=self.C).tolist()}
            Xt = np.asarray(Xt)
        else:
            Xt, yt, audit_te = decode_test(test_dir, features, scaler,
                                           progress=lambda m: self.log(f"[r{self.rank}]{m}"))
        self.Xt = torch.from_numpy(Xt).to(self.dev)
        self.Yt = torch.from_numpy(yt).to(self.dev).long()
        del Xt, yt
        # exact n_ij per owned client -- what APS scales with and what mu is resolved from
        per_client = {}
        for cid in self.cids:
            lo, hi = spans[cid]
            pool_idx = self.train_idx[cid] if self.train_idx is not None else None
            yy = self.Yg[pool_idx] if pool_idx is not None else self.Yg[lo:hi]
            per_client[int(cid)] = torch.bincount(yy, minlength=self.C).cpu().numpy().tolist()
        audit_tr["per_client_class_counts"] = per_client
        self.log(f"[r{self.rank}] data resident: train {tuple(self.Xg.shape)} test {tuple(self.Xt.shape)} "
                 f"({(self.Xg.nelement() + self.Xt.nelement()) * 2 / 2**30:.2f} GiB) "
                 f"in {time.time() - t0:.0f}s")
        return audit_tr, audit_te

    def build(self, masks: np.ndarray):
        cfgm = self.cfg["model_cfg"]
        torch.manual_seed(self.cfg["seed"])
        self.model = build_model(self.cfg["n_features"], cfgm).to(self.dev)
        self.packer = FlatPacker(self.model)
        self.params = [p for p in self.model.parameters()]
        self.crit = nn.CrossEntropyLoss()

        # Every client starts from the SAME initialization: one seeded build, broadcast.
        p0, b0, i0 = self.packer.pack(self.model)
        self.cw = {cid: (p0.clone().to(self.dev), b0.clone().to(self.dev), i0.clone().to(self.dev))
                   for cid in self.cids}
        self.opt = {cid: torch.optim.AdamW(self.params, lr=self.cfg["lr"],
                                           weight_decay=self.cfg["weight_decay"],
                                           betas=tuple(self.cfg["betas"]), eps=self.cfg["eps"],
                                           fused=self.dev.type == "cuda")
                    for cid in self.cids}
        self.scaler = {cid: GradScaler("cuda", enabled=self.amp) for cid in self.cids}
        self.proto_local = {cid: torch.zeros(self.C, self.d, device=self.dev) for cid in self.cids}
        self.proto_count = {cid: torch.zeros(self.C, device=self.dev) for cid in self.cids}

        self.mask = torch.from_numpy(masks.astype(np.float32)).to(self.dev)          # (C, d)
        self.mask_idx = torch.from_numpy(
            np.stack([np.flatnonzero(masks[j]) for j in range(self.C)])).to(self.dev)  # (C, s)
        self.meta_base = make_meta_base(lambda: build_model(self.cfg["n_features"], cfgm))
        self._compiled = None
        self._compile_report = {"enabled": False, "reason": "not attempted"}

    # -- torch.compile ------------------------------------------------------
    # Stated BEFORE the probe, on purpose. A threshold derived from the error it is meant to
    # police cannot fail: a huge delta simply widens the decisive margin until no row qualifies,
    # and an empty decisive set reports zero disagreements. Measured fp16 delta on T4 is
    # ~5e-4..7e-4 (multiples of 2**-11), and 95-96% of probe rows are decisive.
    AMP_MAX_DELTA = 0.05          # above this, fp16 granularity is not the explanation
    MIN_DECISIVE_FRAC = 0.50      # below this the probe proves nothing about argmax stability
    DECISIVE_MARGIN = 0.01        # fixed, NOT a multiple of the measured delta -- see probe_verdict

    def try_compile(self):
        """Compile forward+backward and prove it before trusting it.

        `torch.compile` is lazy, so wrapping the call in `try` catches nothing -- the trial has
        to run a real step. The whole step runs inside the probe (scale, unscale, clip, step,
        update, zero_grad): three bare `backward()` calls accumulate into CUDA-graph-owned
        `.grad` buffers and raise a misleading overwrite error that looks like a compile failure.
        """
        if not self.cfg.get("compile", True):
            self._compile_report = {"enabled": False, "reason": "disabled by config"}
            return self._compile_report
        saved_weights = tuple(t.clone() for t in self.packer.pack(self.model))
        saved_rng = C.rng_state()
        was_training = self.model.training
        try:
            fn = torch.compile(self.model.forward_both, mode="reduce-overhead")
            x = self.Xg[:self.batch].float()
            y = self.Yg[:self.batch]
            opt = torch.optim.AdamW(self.params, lr=0.0, fused=True)
            sc = GradScaler("cuda", enabled=self.amp)
            for _ in range(3):
                with autocast("cuda", dtype=torch.float16, enabled=self.amp):
                    _, logits = fn(x)
                loss = self.crit(logits.float(), y)
                sc.scale(loss).backward()
                sc.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.params, self.cfg["clip"])
                sc.step(opt)
                sc.update()
                opt.zero_grad(set_to_none=True)
            self.model.eval()
            with torch.inference_mode():
                _, ref = self.model.forward_both(x)
                _, got = fn(x)
                ref, got = ref.float().clone(), got.float().clone()
                # The comparison above runs OUTSIDE autocast, i.e. in a precision this run never
                # actually trains or evaluates in. Repeat it in the mode really used, or the
                # reported delta certifies a path nobody takes.
                amp_ref = amp_got = None
                if self.amp:
                    with autocast("cuda", dtype=torch.float16, enabled=True):
                        _, amp_ref = self.model.forward_both(x)
                        _, amp_got = fn(x)
                    amp_ref, amp_got = amp_ref.float().clone(), amp_got.float().clone()
            self.model.train()
            if not torch.allclose(ref, got, rtol=1e-3, atol=1e-4):
                raise RuntimeError("compiled logits exceed rtol=1e-3, atol=1e-4: "
                                   f"{float((ref - got).abs().max())}")
            report = {"enabled": True,
                      **probe_verdict(ref, got, min_decisive_frac=self.MIN_DECISIVE_FRAC,
                                      decisive_margin=self.DECISIVE_MARGIN, label="compile")}
            if amp_ref is not None:
                # fp16 cannot be bit-equal and no tolerance on the raw value would be meaningful
                # here, so the criterion stated up front is behavioural: the ARGMAX must not move
                # on rows where the margin is wide enough for the difference to matter.
                # fp16 cannot be bit-equal, so the criterion is behavioural: the argmax must
                # not move on rows whose margin is wide enough to matter -- under a delta
                # ceiling and a coverage floor fixed before the probe ran.
                v = probe_verdict(amp_ref, amp_got, min_decisive_frac=self.MIN_DECISIVE_FRAC,
                                  max_delta=self.AMP_MAX_DELTA,
                                  decisive_margin=self.DECISIVE_MARGIN, label="autocast")
                report.update(amp_max_abs_dlogit=v["max_abs_dlogit"],
                              amp_decisive_rows=v["decisive_rows"],
                              amp_disagreements=v["disagreements"],
                              amp_decisive_frac=v["decisive_frac"])
            self._compile_report = report
            self._compiled = fn
        except Exception as e:                       # noqa: BLE001 - fall back, never abort
            self._compiled = None
            self._compile_report = {"enabled": False,
                                    "reason": f"{type(e).__name__}: {str(e)[:200]}"}
        finally:
            self.packer.load_into(self.model, *saved_weights)
            C.set_rng_state(saved_rng)
            self.model.train(was_training)
            self.model.zero_grad(set_to_none=True)
        return self._compile_report

    def _fwd(self, x):
        if self._compiled is not None and x.shape[0] == self.batch:
            return self._compiled(x)
        return self.model.forward_both(x)            # tail batch: eager, same weights, same math

    # -- one client, one local epoch ---------------------------------------
    def train_client(self, cid: int, rnd: int, reg: ProtoRegularizer | None, lam: float) -> dict:
        pk, bk, ik = self.cw[cid]
        self.packer.load_into(self.model, pk, bk, ik)
        self.model.train()
        opt, sc = self.opt[cid], self.scaler[cid]
        opt.zero_grad(set_to_none=True)

        lo, hi = self.spans[cid]
        g = torch.Generator(device=self.dev)
        g.manual_seed(client_seed(self.cfg["seed"], rnd, cid))
        if self.train_idx is not None:
            pool = self.train_idx[cid]
            perm = pool[torch.randperm(pool.numel(), generator=g, device=self.dev)]
        else:
            perm = lo + torch.randperm(hi - lo, generator=g, device=self.dev)

        ce_acc = torch.zeros((), device=self.dev)
        rg_acc = torch.zeros((), device=self.dev)
        gn_acc = torch.zeros((), device=self.dev)
        applied = torch.zeros((), device=self.dev)
        skipped = torch.zeros((), device=self.dev)
        zero = torch.zeros((), device=self.dev)
        n_steps = 0
        t0 = time.time()
        last_heartbeat = t0

        for i in range(0, perm.numel(), self.batch):
            idx = perm[i:i + self.batch]
            xb = self.Xg[idx].float()
            yb = self.Yg[idx]
            with autocast("cuda", dtype=torch.float16, enabled=self.amp):
                h, logits = self._fwd(xb)
            ce = self.crit(logits.float(), yb)                 # loss in fp32, outside autocast
            if reg is not None:
                r = reg(h.float(), yb)
                loss = ce + lam * r
            else:
                r = zero
                loss = ce
            sc.scale(loss).backward()
            sc.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(self.params, self.cfg["clip"])
            prev = sc._scale.detach().clone() if self.amp else None
            sc.step(opt)
            sc.update()
            opt.zero_grad(set_to_none=True)
            n_steps += 1
            # A skipped step's grad norm is inf BY CONSTRUCTION and its loss may be nan.
            # torch.where, not multiplication: inf * 0 is nan.
            ok = (sc._scale >= prev) if self.amp else torch.ones((), device=self.dev, dtype=torch.bool)
            ce_acc += torch.where(ok, ce.detach(), zero)
            rg_acc += torch.where(ok, r.detach(), zero)
            gn_acc += torch.where(ok, gn.detach(), zero)
            applied += ok.float()
            skipped += (~ok).float()
            if time.time() - last_heartbeat >= 30:
                self.log(f"[r{self.rank}] round {rnd} client {cid} step {n_steps}/"
                         f"{math.ceil(perm.numel()/self.batch)}")
                last_heartbeat = time.time()

        n_applied = float(applied)
        stats = {
            "client_id": cid, "rows": int(perm.numel()), "steps": n_steps,
            "applied_steps": int(n_applied), "skipped_steps": int(float(skipped)),
            "ce": float(ce_acc) / max(n_applied, 1.0),
            "reg": float(rg_acc) / max(n_applied, 1.0),
            "grad_norm": float(gn_acc) / max(n_applied, 1.0),
            "train_seconds": round(time.time() - t0, 2),
        }
        p, b, i2 = self.packer.pack(self.model)
        skip_budget = max(8, math.ceil(0.01 * n_steps))
        if (stats["applied_steps"] == 0 or stats["skipped_steps"] > skip_budget
                or not all(math.isfinite(stats[k]) for k in ("ce", "reg", "grad_norm"))
                or not torch.isfinite(p).all() or not torch.isfinite(b).all()):
            raise RuntimeError(f"invalid update from client {cid}: {stats}")
        self.cw[cid] = (p.to(self.dev), b.to(self.dev), i2.to(self.dev))
        return stats

    # -- Eq. (3): prototypes from the POST-epoch weights ---------------------
    @torch.inference_mode()
    def client_prototypes(self, cid: int) -> None:
        """One extra inference pass over the client's own rows with the final θ_i.

        The reference FedProto implementation instead averages features collected DURING the
        local epoch. That is fine at ~59 steps per epoch (CIFAR-10, 20 clients, batch 32); here
        a local epoch is up to 11,506 steps, over which the weights move enough that an
        epoch-average would describe a model that no longer exists. Eq. (3) is written for a
        single θ_i, so this build takes it literally. Cost is one inference pass over the
        training set per round.

        Run in eval mode: BatchNorm uses running statistics and dropout is off, which is exactly
        the feature extractor Eq. (12) uses at test time.
        """
        pk, bk, ik = self.cw[cid]
        self.packer.load_into(self.model, pk, bk, ik)
        self.model.eval()
        acc = ProtoAccumulator(self.C, self.d, self.dev)
        lo, hi = self.spans[cid]
        rows = self.train_idx[cid] if self.train_idx is not None else None
        n = rows.numel() if rows is not None else hi - lo
        for i in range(0, n, self.eval_batch):
            if rows is not None:
                idx = rows[i:i + self.eval_batch]
                xb, yb = self.Xg[idx].float(), self.Yg[idx]
            else:
                xb = self.Xg[lo + i:lo + min(i + self.eval_batch, n)].float()
                yb = self.Yg[lo + i:lo + min(i + self.eval_batch, n)]
            with autocast("cuda", dtype=torch.float16, enabled=self.amp):
                h, _ = self.model.forward_both(xb)
            acc.update(h.float(), yb)
        c, cnt = acc.finish()
        if not torch.isfinite(c).all() or int(cnt.sum()) != n:
            raise RuntimeError(f"client {cid}: invalid prototype or row coverage")
        self.proto_local[cid] = c
        self.proto_count[cid] = cnt
        self.model.train()

    def upload(self, cid: int) -> tuple[np.ndarray, np.ndarray]:
        """Eq. (10) client side: n_ij · ĉ_L[i,j], compressed to s dims. n_ij never goes on the
        wire by itself -- that is the privacy property APS buys."""
        c = self.proto_local[cid]
        n = self.proto_count[cid]
        comp = torch.gather(c, 1, self.mask_idx)                 # (C, s)
        scaled = comp * n[:, None]
        return scaled.double().cpu().numpy(), (n > 0).cpu().numpy()

    # -- evaluation ---------------------------------------------------------
    def evaluate(self, want_preds: bool = False):
        folded, protos, present = [], [], []
        for cid in self.cids:
            pk, bk, ik = self.cw[cid]
            self.packer.load_into(self.model, pk, bk, ik)
            folded.append(fold_bn(self.model))
            protos.append(self.proto_local[cid])
            present.append(self.proto_count[cid] > 0)
        P = torch.stack(protos)
        R = torch.stack(present)
        g = int(self.cfg.get("eval_group", 1))
        t0 = time.time()
        if g > 1:
            ev = VmapEvaluator(folded, g, self.meta_base, half=self.amp)
            cm_p, cm_c, preds = ev.run(P, R, self.Xt, self.Yt, batch=self.eval_batch,
                                       C=self.C, amp=self.amp, want_preds=want_preds)
            del ev
        else:
            cm_p, cm_c, preds = eval_sequential(folded, P, R, self.Xt, self.Yt,
                                                batch=self.eval_batch, C=self.C,
                                                amp=self.amp, want_preds=want_preds)
        del folded
        torch.cuda.empty_cache()
        return cm_p.cpu().numpy(), cm_c.cpu().numpy(), preds, round(time.time() - t0, 2)

    # -- state --------------------------------------------------------------
    def weights_blob(self) -> dict:
        p = torch.stack([self.cw[c][0] for c in self.cids]).cpu()
        b = torch.stack([self.cw[c][1] for c in self.cids]).cpu()
        i = torch.stack([self.cw[c][2] for c in self.cids]).cpu()
        return {"client_ids": list(self.cids), "params": p, "buffers": b, "int_buffers": i}

    def optim_blob(self) -> dict:
        """Per-client AdamW moments and GradScaler scale.

        Persistent across rounds: each client's model is continuous through the whole run
        (personalized FL never overwrites it with a global average), so its moment estimates
        remain valid. Resetting them every round would re-warm bias correction 50 times.
        """
        out = {}
        for cid in self.cids:
            st = self.opt[cid].state_dict()
            out[cid] = {"opt": st, "scaler": self.scaler[cid].state_dict()}
        return {"client_ids": list(self.cids), "state": out, "rng": C.rng_state()}

    def load_state(self, weights: dict, optim: dict, protos: dict) -> None:
        if optim["client_ids"] != self.cids:
            raise RuntimeError("optimizer shard ownership changed")
        C.set_rng_state(optim["rng"])
        ids = list(weights["client_ids"])
        pos = {c: k for k, c in enumerate(ids)}
        for cid in self.cids:
            k = pos[cid]
            self.cw[cid] = (weights["params"][k].to(self.dev),
                            weights["buffers"][k].to(self.dev),
                            weights["int_buffers"][k].to(self.dev))
            self.opt[cid].load_state_dict(optim["state"][cid]["opt"])
            self.scaler[cid].load_state_dict(optim["state"][cid]["scaler"])
        pp = protos["local"]
        cc = protos["counts"]
        # The parent stores prototypes indexed by GLOBAL client id (row cid), while a worker
        # blob would carry an explicit id list. Accept both.
        pid = {c: k for k, c in enumerate(protos.get("client_ids", range(len(pp))))}
        for cid in self.cids:
            self.proto_local[cid] = pp[pid[cid]].to(self.dev)
            self.proto_count[cid] = cc[pid[cid]].to(self.dev)

    def proto_blob(self) -> dict:
        return {"client_ids": list(self.cids),
                "local": torch.stack([self.proto_local[c] for c in self.cids]).cpu(),
                "counts": torch.stack([self.proto_count[c] for c in self.cids]).cpu()}


# ---------------------------------------------------------------------------
# process entry point
# ---------------------------------------------------------------------------

def worker_main(rank: int, cfg: dict, cids: list[int], cmd_q, res_q, paths: dict):
    """Long-lived: set up once, then serve commands until told to stop."""
    def log(msg):
        print(msg, flush=True)

    try:
        w = Worker(rank, cfg, cids, log=log)
        from pathlib import Path
        import json
        features = cfg["feature_cols"]
        scaler = json.loads(Path(paths["scaler"]).read_text())
        train_index = None
        if paths.get("train_index"):
            blob = np.load(paths["train_index"], allow_pickle=False)
            train_index = {int(k): blob[k] for k in blob.files if int(k) in cids}
        eval_npy = None
        if paths.get("eval_X"):
            eval_npy = {"X": paths["eval_X"], "y": paths["eval_y"]}
        audit_tr, audit_te = w.load_data(Path(paths["client_root"]), Path(paths["test_dir"]),
                                         features, scaler, train_index, eval_npy)
        w.build(np.load(paths["masks"]))
        rep = w.try_compile()
        log(f"[r{rank}] compile: {rep}")
        res_q.put({"cmd": "ready", "rank": rank, "audit_train": audit_tr,
                   "audit_test": audit_te, "compile": rep,
                   # y_true is identical for every client and every round: report it once from
                   # rank 0 and store one copy, not 100 x 50 copies of the same 10.3 MB vector.
                   "y_true": w.Yt.to(torch.uint8).cpu().numpy() if rank == 0 else None,
                   "peak_gb": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if w.dev.type == "cuda" else 0.0)})

        while True:
            msg = cmd_q.get()
            kind = msg["cmd"]
            if kind == "stop":
                break

            if kind == "round":
                rnd = msg["round"]
                lam = float(cfg["lam"])
                reg = None
                if msg["global_proto"] is not None:
                    sparse = msg["global_proto"].to(w.dev)             # (C, d) already sparse
                    hasg = msg["has_global"].to(w.dev)
                    mu = msg["mu"].to(w.dev)
                    reg = ProtoRegularizer(sparse, hasg, w.mask, mu, w.s)
                stats, uploads = [], []
                order = sorted(w.cids, key=lambda c: -(w.spans[c][1] - w.spans[c][0]))
                for cid in order:                                      # longest first
                    stats.append(w.train_client(cid, rnd, reg, lam))
                    w.client_prototypes(cid)
                for cid in sorted(w.cids):                             # deterministic order
                    up, pres = w.upload(cid)
                    uploads.append((cid, up, pres))
                res_q.put({"cmd": "trained", "rank": rank, "round": rnd,
                           "stats": stats, "uploads": uploads,
                           "peak_gb": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if w.dev.type == "cuda" else 0.0)})

            elif kind == "evaluate":
                cm_p, cm_c, preds, secs = w.evaluate(want_preds=msg.get("want_preds", False))
                res_q.put({"cmd": "evaluated", "rank": rank, "client_ids": list(w.cids),
                           "cm_proto": cm_p, "cm_clf": cm_c, "preds": preds,
                           "eval_seconds": secs,
                           "peak_gb": (round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if w.dev.type == "cuda" else 0.0)})

            elif kind == "commit":
                rnd = msg["round"]
                out = msg["run_dir"]
                C.atomic_save(w.optim_blob(),
                              os.path.join(out, "resume", f"round_{rnd:03d}.w{rank}.pt"))
                res_q.put({"cmd": "committed", "rank": rank, "round": rnd,
                           "weights": w.weights_blob(), "protos": w.proto_blob()})

            elif kind == "restore":
                w.load_state(msg["weights"], msg["optim"], msg["protos"])
                res_q.put({"cmd": "restored", "rank": rank})

            else:
                raise RuntimeError(f"unknown command {kind!r}")
    except Exception:                                # noqa: BLE001 - surface, do not hang
        res_q.put({"cmd": "error", "rank": rank, "traceback": traceback.format_exc()})
        raise
