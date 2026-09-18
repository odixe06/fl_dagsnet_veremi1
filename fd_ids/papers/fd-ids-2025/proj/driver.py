"""One persistent worker per GPU, spawned once for the whole run.

Both GPUs hold the entire partition, so any client can go to whichever GPU is free.
Clients are handed out longest-first; aggregation is re-sorted by client id so float
addition order never depends on which worker finished first, and every client re-seeds
the default generator from (seed, round, client) so its update does not depend on the
schedule either.
"""
import json, math, shutil, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp

from proj.model import build_model
from proj.fdids import (layout, flatten, unflatten_into, teacher_logits,
                        client_update, aggregate, amp)
from proj.metrics import metrics_from_confusion, per_class_from_confusion
from proj import ckpt as C


def _resident(path, dev, chunk=1 << 22):
    """mmap -> GPU in chunks. A whole-array np.ascontiguousarray would materialise 5.7 GB
    of train features in host RAM per worker before the copy, and hands torch a read-only
    array. Chunking bounds the host side to `chunk` rows."""
    a = np.load(path, mmap_mode="r")
    t = torch.empty(tuple(a.shape), dtype=torch.from_numpy(np.array(a[:1])).dtype,
                    device=dev)
    for i in range(0, len(a), chunk):
        t[i:i + chunk] = torch.from_numpy(np.array(a[i:i + chunk]))
    return t


def _compile(model, cfg, dev, sample, rank=0):
    """reduce-overhead captures forward+backward into a CUDA graph. torch.compile is lazy,
    so a try around the call catches nothing -- run real steps and compare against eager.

    Both sides are measured from the SAME state. Warm-up runs three train-mode steps, which
    move every BatchNorm running stat and consume the Dropout generator; a gate that takes
    its reference before warm-up and its candidate after is comparing two different models
    and rejects a compiler that is in fact correct. Measured on an identity compiler:
    max|dlogit| 0.214743 and 62 changed BN buffers, i.e. a false reject every time."""
    if not cfg["compile"]:
        return model
    params = list(model.parameters())
    drops = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    keep = [m.p for m in drops]
    snap = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rng = torch.get_rng_state()
    crng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def restore():
        with torch.no_grad():
            sd = model.state_dict()
            for k, v in snap.items():
                sd[k].copy_(v)                      # copy_ keeps addresses -> graph stays valid
        torch.set_rng_state(rng)
        if crng is not None: torch.cuda.set_rng_state_all(crng)
        model.zero_grad(set_to_none=True)

    def probe(mod):
        """One production-shaped step: the SAME loss the run trains on.

        A CE-only probe leaves the KD and proximal terms out of the compared graph, and
        those are two thirds of Eq. (6). log_softmax at T and the flat parameter gather are
        exactly the kind of fusion a backend gets wrong on its own."""
        restore(); model.train()
        with amp(cfg):
            out = mod(sample)
        out = out.float()
        T, lam, beta, mu = cfg["temperature"], cfg["lam"], cfg["beta"], cfg["mu"]
        l_hard = F.cross_entropy(out, labels)
        l_soft = (T * T) * F.kl_div(F.log_softmax(out / T, 1),
                                    F.log_softmax(zt / T, 1),
                                    reduction="batchmean", log_target=True)
        (lam * l_hard + (1.0 - lam) * l_soft).backward()
        flat = torch.cat([p.detach().reshape(-1) for p in params])
        o2 = 0
        for p in params:                                  # Eq. (3) exactly as production
            n = p.numel()
            p.grad.add_((flat[o2:o2 + n] - anchor[o2:o2 + n]).view_as(p), alpha=beta * mu)
            o2 += n
        torch.nn.utils.clip_grad_norm_(params, cfg["clip"])
        g = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p))
                       .reshape(-1).float().clone() for p in params])
        o = out.clone()
        model.zero_grad(set_to_none=True)
        return o, g

    try:
        labels = torch.randint(0, cfg["num_classes"], (sample.shape[0],), device=dev)
        zt = torch.randn(sample.shape[0], cfg["num_classes"], device=dev) * 3
        anchor = torch.cat([p.detach().reshape(-1) for p in params]) \
            + torch.randn(sum(p.numel() for p in params), device=dev) * 0.02
        comp = torch.compile(model, mode="reduce-overhead")     # CUDA graphs: the 2.86x on T4
        model.train()
        for _ in range(3):                                      # warm up + capture
            with amp(cfg):
                out = comp(sample)
            F.cross_entropy(out.float(), labels).backward()
            model.zero_grad(set_to_none=True)                   # graph-owned .grad released
        # Dropout OFF for the comparison only, and for BOTH sides.
        # restore() puts torch's RNG back, so eager repeats its mask exactly -- but Inductor
        # functionalises RNG and draws its own Philox offsets, so the compiled side cannot be
        # made to draw the same one. The gate then measures the difference between two dropout
        # masks and calls it compiler error: measured max|dlogit| 6.07e-01 at p=0.1 against
        # 7.32e-04 at p=0, on sm_86 where Triton is not in question. That is what disabled
        # compile on all three T4 runs. p is restored below, and the warm-up above already
        # captured the graph at the production p, so production reuses that entry.
        for m in drops: m.p = 0.0
        try:
            ref_z, ref_g = probe(model)
            got_z, got_g = probe(comp)
        finally:
            for m, p_ in zip(drops, keep): m.p = p_
        restore(); model.train()

        dz = (got_z - ref_z).abs().max().item()
        gn = ref_g.norm().item()
        dg = (got_g - ref_g).norm().item() / (gn + 1e-12)
        # count as integers: torch.mean on CUDA returns 0.99999994 for a perfect match
        flip_all = int((got_z.argmax(1) != ref_z.argmax(1)).sum())
        top2 = ref_z.topk(2, dim=1).values
        decisive = (top2[:, 0] - top2[:, 1]) > max(10 * dz, 1e-3)
        n_dec = int(decisive.sum())
        flip_dec = int((got_z.argmax(1) != ref_z.argmax(1))[decisive].sum())
        # `flip_dec == 0` over an EMPTY decisive set says nothing at all. Demand that the
        # check actually covered rows before letting it certify anything.
        if n_dec < sample.shape[0] // 10:
            raise RuntimeError(f"cannot certify: only {n_dec} of {sample.shape[0]} rows have "
                               f"a margin above {max(10 * dz, 1e-3):.2e}")
        if flip_dec or not math.isfinite(dz) or not math.isfinite(dg) \
                or dz > 5e-2 or dg > 5e-2:
            raise RuntimeError(f"compile mismatch: dlogit={dz} dgrad_rel={dg} "
                               f"flips {flip_dec}/{n_dec} decisive, {flip_all} of all")
        print(f"[rank{rank}] compile OK: max|dlogit|={dz:.2e} rel|dgrad|={dg:.2e} | "
              f"top-1 flips {flip_all} of {sample.shape[0]} rows ({flip_dec} of {n_dec} "
              f"decisive) | |g|={gn:.3e}", flush=True)
        return comp
    except Exception as e:                        # sm_75 Triton is the documented risk
        for m, p_ in zip(drops, keep): m.p = p_   # never leave the model with dropout off
        restore(); model.train()
        print(f"[rank{rank}] compile DISABLED -> eager: {e}", flush=True)
        return model


def worker(rank, cfg, task_q, res_q):
    """Wrapper: a worker that dies silently leaves the parent with only an exit code, and
    the real error is always in the CHILD traceback, not the spawn wrapper."""
    try:
        _worker(rank, cfg, task_q, res_q)
    except Exception:
        import traceback
        res_q.put(("error", rank, traceback.format_exc()))
        raise


def _worker(rank, cfg, task_q, res_q):
    cuda = cfg.get("device", "cuda") == "cuda"
    dev = torch.device(f"cuda:{rank}" if cuda else "cpu")
    if cuda: torch.cuda.set_device(dev)
    torch.manual_seed(cfg["seed"] + rank)
    cache = Path(cfg["cache"])
    X = _resident(cache / "train_X.f16.npy", dev)
    Y = _resident(cache / "train_y.u8.npy", dev)
    TX = _resident(cache / "test_X.f16.npy", dev)
    TY = _resident(cache / "test_y.u8.npy", dev)

    student = build_model(cfg).to(dev).train()
    teacher = build_model(cfg).to(dev).eval()
    for p in teacher.parameters(): p.requires_grad_(False)
    fkeys, ikeys, n_params = layout(student)
    # Probe on real rows: random N(0,1) has none of the heavy tails of the z-scored
    # features, and a kernel that is wrong only at large magnitude would pass on noise.
    comp = _compile(student, cfg, dev, X[:cfg["batch"]].float(), rank)
    opt = torch.optim.Adam(student.parameters(), lr=cfg["lr"], fused=cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    if cuda:
        scaler.scale(torch.zeros(1, device=dev))  # force _scale to exist
        assert scaler._scale is not None, "GradScaler._scale gone; skips would read as 0"
    res_q.put(("ready", rank, "eager" if comp is student else "compiled"))

    w_global = None
    while True:
        task = task_q.get()
        kind = task[0]
        if kind == "stop":
            return
        if kind == "backend":
            # Both ranks gate independently, so one can compile and the other fall back.
            # A round whose clients were trained on two different backends is not a round
            # anyone can reproduce; the driver forces the lower common denominator.
            if task[1] == "eager": comp = student
            res_q.put(("backend_ok", rank, "eager" if comp is student else "compiled"))
            continue
        if kind == "weights":
            w_global = task[1].to(dev)
            iv = task[2].to(dev)
            unflatten_into(student, w_global, iv, fkeys, ikeys)
            unflatten_into(teacher, w_global, iv, fkeys, ikeys)
            teacher.eval()
            # Adam moments are local state; FD-IDS Algorithm 1 restarts the client from
            # w_G^t each round, so the optimizer restarts with it.
            opt.state.clear()
            res_q.put(("weights_ok", rank))
            continue
        if kind == "eval":
            lo, hi = task[1], task[2]
            if cuda: torch.cuda.reset_peak_memory_stats(dev)
            cm = torch.zeros(cfg["num_classes"] ** 2, dtype=torch.long, device=dev)
            preds = torch.empty(hi - lo, dtype=torch.uint8, device=dev)
            nonfin = torch.zeros((), dtype=torch.long, device=dev)
            m = teacher                                  # teacher holds the global weights
            with torch.inference_mode():
                for i in range(lo, hi, cfg["eval_batch"]):
                    j = min(i + cfg["eval_batch"], hi)
                    with amp(cfg):
                        z = m(TX[i:j].float())
                    # argmax of a row of NaN returns 0, a perfectly ordinary class index.
                    # Count on device: a per-batch .item() would sync ~650 times a shard.
                    nonfin += (~torch.isfinite(z)).sum()
                    p = z.argmax(1)
                    preds[i - lo:j - lo] = p.to(torch.uint8)
                    cm += torch.bincount(TY[i:j].long() * cfg["num_classes"] + p,
                                         minlength=cfg["num_classes"] ** 2)
            res_q.put(("eval", rank, cm.cpu(), lo, hi, preds.cpu(), int(nonfin.item()),
                       (torch.cuda.max_memory_allocated(dev) / 2**30) if cuda else 0.0))
            continue
        if kind == "train":
            cid, lo, hi, rnd = task[1], task[2], task[3], task[4]
            t0 = time.monotonic()
            if cuda: torch.cuda.reset_peak_memory_stats(dev)
            unflatten_into(student, w_global, task[5].to(dev), fkeys, ikeys)
            opt.state.clear()
            scaler = torch.amp.GradScaler("cuda", enabled=cuda)
            if cuda: scaler.scale(torch.zeros(1, device=dev))
            t_kd = time.monotonic()
            ZT = teacher_logits(teacher, X, lo, hi, cfg, cfg["eval_batch"])
            if cuda: torch.cuda.synchronize(dev)
            t_kd = time.monotonic() - t_kd
            # Every stochastic input to this client derives from (seed, round, client):
            # the default generator drives Dropout, `g` drives the shuffle. Seeding the
            # worker once at startup instead would make each client's Dropout masks depend
            # on how many clients that rank happened to take first, so the same round on
            # the same data would give different weights whenever the LPT schedule shifted.
            s = cfg["seed"] * 1_000_003 + rnd * 10_007 + cid
            torch.manual_seed(s)
            g = torch.Generator(device=dev); g.manual_seed(s)
            student.train()
            ce, kd, gn, sk, nf, nsteps = client_update(
                comp, student, opt, scaler, X, Y, lo, hi, w_global, n_params, ZT, cfg, g)
            del ZT
            fv, iv = flatten(student, fkeys, ikeys)
            skipped = int(sk.item())
            applied = nsteps - skipped            # NOT max(1, ...): 0 must stay 0 so the
            div = max(1, applied)                 # driver can reject the client
            res_q.put(("train", cid, hi - lo, fv.cpu(), iv.cpu(),
                       {"round": rnd, "cid": cid, "n_k": hi - lo, "rank": rank,
                        "steps": nsteps, "applied": applied, "skipped": skipped,
                        "nonfinite": int(nf.item()), "seed": s,
                        "ce": float(ce) / div, "kd": float(kd) / div,
                        "gnorm": float(gn) / div, "sec": time.monotonic() - t0,
                        "teacher_sec": t_kd,
                        # reset at the top of this task, so it is THIS client's peak and
                        # not a high-water mark left behind by an earlier one
                        "vram_gb": (torch.cuda.max_memory_allocated(dev) / 2**30
                                    if cuda else 0.0)}))


def check_updates(rnd, updates, stats, n_clients, max_skips=None):
    """Every reason a round must not be aggregated, in one pure function so it can be
    tested without two GPUs and a spawned worker.

    Skipped steps are NOT a failure. Each client starts a fresh GradScaler at 2**16, so the
    first steps of nearly every client overflow while it calibrates -- that is the scaler
    doing its job, and the step it discarded never touched the weights. What must be
    rejected is a client that applied no step at all, one that skipped far more than
    calibration explains, and one whose APPLIED steps carried a non-finite gradient: the
    proximal term is added after unscale_, so that last case gets past the scaler's own
    check and does reach the weights."""
    if len(stats) != n_clients:
        raise RuntimeError(f"round {rnd}: {len(stats)} of {n_clients} clients reported")
    bad = [cid for cid, _, fv, _ in updates if not torch.isfinite(fv).all()]
    if bad:
        raise RuntimeError(f"round {rnd}: non-finite weights from clients {bad}")
    for c in sorted(stats):
        st = stats[c]
        if st["applied"] + st["skipped"] != st["steps"]:
            raise RuntimeError(f"round {rnd}: client {c} reports applied+skipped="
                               f"{st['applied'] + st['skipped']} of {st['steps']} steps")
    # A client whose every AMP step overflowed returns w_G^t unchanged. Averaging that in is
    # not a small error: it silently drops the client's data from the round and lowers the
    # effective step size, and every downstream number still looks entirely normal.
    dead = [c for c in sorted(stats) if stats[c]["applied"] == 0]
    if dead:
        raise RuntimeError(f"round {rnd}: clients {dead} applied zero optimizer steps; "
                           "they would contribute the global weights unchanged")
    diverged = [c for c in sorted(stats) if stats[c].get("nonfinite")]
    if diverged:
        raise RuntimeError(f"round {rnd}: clients {diverged} APPLIED a step whose gradient "
                           "was not finite; the scaler's own overflow check runs before the "
                           "proximal term, so this reached the weights")
    if max_skips is not None:
        over = [c for c in sorted(stats) if stats[c]["skipped"] > max_skips]
        if over:
            raise RuntimeError(
                f"round {rnd}: clients {over} skipped more than the warm-up budget "
                f"({max_skips}): " + ", ".join(f"{c}={stats[c]['skipped']}" for c in over)
                + ". Calibration costs a handful of steps; this is a training problem.")


def _collect(res_q, procs, n, timeout=1800):
    """A worker killed by the OS puts nothing on the queue. Poll in short slices and check
    liveness between them, or an OOM kill becomes a multi-hour hang."""
    out, deadline = [], time.time() + timeout
    while len(out) < n:
        try:
            msg = res_q.get(timeout=2.0)
            if msg[0] == "error":
                raise RuntimeError(f"worker {msg[1]} raised:\n{msg[2]}")
            out.append(msg)
        except RuntimeError:
            raise
        except Exception:
            for p in procs:
                if not p.is_alive() and p.exitcode not in (0, None):
                    raise RuntimeError(f"worker {p.pid} died, exitcode {p.exitcode} "
                                       f"(negative = signal; -9 is the OOM killer)")
            if time.time() > deadline:
                raise RuntimeError(f"timed out waiting for {n - len(out)} results")
    return out


def _shutdown(procs, task_qs):
    for q in task_qs:
        try: q.put(("stop",))
        except Exception: pass
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate(); p.join(timeout=10)


def run(cfg, spans, class_names, wandb_run=None, t_origin=None):
    """t_origin is a time.monotonic() reading from when the SESSION started, not from when
    this call did. Worker spawn, the resident copy and compilation are minutes the 12 h cap
    charges for, and a deadline that started here would happily begin a round the session
    cannot finish. monotonic, not time(): a wall-clock step would move the deadline."""
    t_start = t_origin if t_origin is not None else time.monotonic()
    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    task_qs = [ctx.Queue() for _ in range(cfg["world_size"])]
    res_q = ctx.Queue()
    procs = [ctx.Process(target=worker, args=(r, cfg, task_qs[r], res_q), daemon=True)
             for r in range(cfg["world_size"])]
    try:
        for p in procs: p.start()
        ready = _collect(res_q, procs, cfg["world_size"], timeout=3600)
        backends = {m[1]: m[2] for m in ready}
        if len(set(backends.values())) > 1:
            print(f"[driver] ranks disagree on backend {backends}; forcing eager on all",
                  flush=True)
            for r in range(cfg["world_size"]): task_qs[r].put(("backend", "eager"))
            backends = {m[1]: m[2] for m in _collect(res_q, procs, cfg["world_size"])}
        cfg["backend"] = sorted(set(backends.values()))[0]
        startup = time.monotonic() - t_start
        print(f"[driver] {cfg['world_size']} workers ready on {cfg['backend']} "
              f"({startup:.0f}s into the session)", flush=True)
        cfg["startup_seconds"] = startup
        # Push the effective backend somewhere READABLE WHILE THE RUN IS ALIVE. It is
        # decided after wandb.init() captured the config, and the per-round log excludes
        # non-numeric fields, so the first real run gave no way to tell from outside
        # whether it was paying the ~2.9x eager penalty until it had finished.
        if wandb_run is not None:
            try:
                wandb_run.config.update({"backend": cfg["backend"],
                                         "startup_seconds": round(startup, 1)},
                                        allow_val_change=True)
                wandb_run.summary["backend"] = cfg["backend"]
            except Exception as e:
                print(f"[driver] could not publish backend to W&B: {e}", flush=True)
        return _rounds(cfg, spans, class_names, wandb_run, t_start, procs, task_qs, res_q)
    finally:
        # Without this a driver-side exception leaves two processes holding both GPUs, and
        # the next cell in the notebook fails with a CUDA OOM that names nothing.
        _shutdown(procs, task_qs)


def _rounds(cfg, spans, class_names, wandb_run, t_start, procs, task_qs, res_q):
    d = C.run_dir(cfg["run_name"])
    torch.manual_seed(cfg["seed"])                 # seed BEFORE building: w_G^0 is seeded
    model0 = build_model(cfg)
    fkeys, ikeys, n_params = layout(model0)
    w_global, i_global = flatten(model0, fkeys, ikeys)

    start = 1
    last = C.resolve_resume(cfg["run_name"], cfg)
    if last is not None:
        w = torch.load(d / "weights" / f"round_{last:03d}.pt", map_location="cpu",
                       weights_only=True)
        model0.load_state_dict(w["model"], strict=True)
        w_global, i_global = flatten(model0, fkeys, ikeys)
        start = last + 1
        print(f"[driver] resumed at round {start}")
    elif cfg.get("require_resume"):
        raise SystemExit("require_resume set and no checkpoint found")
    if start > cfg["rounds"]:
        print(f"[driver] nothing to do: {last} rounds already complete")
        return []

    n_total = sum(hi - lo for lo, hi in spans.values())
    # longest-first bounds the idle tail: sending the biggest client last strands a GPU
    order = sorted(spans, key=lambda c: spans[c][1] - spans[c][0], reverse=True)
    n_test = cfg["n_test"]
    shards = [(i * n_test // cfg["world_size"], (i + 1) * n_test // cfg["world_size"])
              for i in range(cfg["world_size"])]
    hist = []
    reserve = cfg.get("finalize_reserve_seconds", 600)
    elapsed = time.monotonic() - t_start
    if elapsed + reserve >= cfg["max_seconds"]:
        # Startup, prepack and compile already spent the budget. Beginning a round here
        # produces nothing and loses the session; say so instead.
        print(f"[driver] no round started: {elapsed/3600:.2f} h of the "
              f"{cfg['max_seconds']/3600:.2f} h budget is already gone", flush=True)
        return hist

    for rnd in range(start, cfg["rounds"] + 1):
        t0 = time.monotonic()
        for r in range(cfg["world_size"]):
            task_qs[r].put(("weights", w_global, i_global))
        _collect(res_q, procs, cfg["world_size"])

        pending, nxt, results = {}, 0, []
        for r in range(cfg["world_size"]):                 # prime both GPUs
            if nxt < len(order):
                c = order[nxt]; nxt += 1
                task_qs[r].put(("train", c, *spans[c], rnd, i_global)); pending[r] = c
        while len(results) < len(order):
            msg = _collect(res_q, procs, 1)[0]
            assert msg[0] == "train", msg[0]
            results.append(msg[1:])
            r = next(k for k, v in pending.items() if v == msg[1])
            if nxt < len(order):
                c = order[nxt]; nxt += 1
                task_qs[r].put(("train", c, *spans[c], rnd, i_global)); pending[r] = c
            else:
                pending.pop(r)

        updates = sorted([(cid, nk, fv, iv) for cid, nk, fv, iv, _ in results],
                         key=lambda t: t[0])            # NOT completion order
        stats = {cid: s for cid, _, _, _, s in results}
        check_updates(rnd, updates, stats, len(order), cfg.get("max_skips_per_client"))
        w_global, i_global = aggregate(updates, n_total)

        for r in range(cfg["world_size"]):
            task_qs[r].put(("weights", w_global, i_global))
        _collect(res_q, procs, cfg["world_size"])
        for r in range(cfg["world_size"]):
            task_qs[r].put(("eval", *shards[r]))
        ev = sorted(_collect(res_q, procs, cfg["world_size"]), key=lambda e: e[3])
        nf = sum(e[6] for e in ev)
        if nf:
            raise RuntimeError(f"round {rnd}: {nf} non-finite test logits; argmax would "
                               "have turned them into ordinary class labels")
        cm = sum(e[2] for e in ev).reshape(cfg["num_classes"], cfg["num_classes"]).numpy()
        assert cm.sum() == n_test, f"confusion covers {cm.sum()} of {n_test} rows"
        preds = torch.cat([e[5] for e in ev]).numpy()
        assert len(preds) == n_test, f"{len(preds)} predictions for {n_test} rows"
        met = metrics_from_confusion(cm)

        # ce_client_mean is the unweighted mean ACROSS CLIENTS of each client's mean over
        # its applied steps -- not the mean loss over training samples. At K=20 vs K=100 the
        # two differ, and comparing scenarios on the wrong one reads as a real effect.
        row = {"round": rnd, **met,
               "ce_client_mean": float(np.mean([s["ce"] for s in stats.values()])),
               "kd_client_mean": float(np.mean([s["kd"] for s in stats.values()])),
               "grad_norm": float(np.mean([s["gnorm"] for s in stats.values()])),
               "steps": int(sum(s["steps"] for s in stats.values())),
               "applied": int(sum(s["applied"] for s in stats.values())),
               "skipped": int(sum(s["skipped"] for s in stats.values())),
               "teacher_sec": float(sum(s["teacher_sec"] for s in stats.values())),
               "vram_train_gb": max(s["vram_gb"] for s in stats.values()),
               "vram_eval_gb": max(e[7] for e in ev),
               "backend": cfg.get("backend", "?"),
               "seconds": 0.0}

        # ---- commit. Marker absolutely last.
        unflatten_into(model0, w_global, i_global, fkeys, ikeys)
        C.save_round_weights(model0, rnd, cfg, met, cfg["run_name"])
        np.save(d / "confusion" / f"round_{rnd:03d}.npy", cm)
        np.save(d / "preds" / f"round_{rnd:03d}.u8.npy", preds)
        (d / "logs" / f"round_{rnd:03d}.json").write_text(
            json.dumps([stats[c] for c in sorted(stats)], indent=1))
        # `seconds` BEFORE the W&B call, not after: logging a dict whose `seconds` has not
        # been filled in yet sends 0.0, which is what the first real run actually reported.
        # The budget below does not depend on this field -- it compares absolute session
        # elapsed, so the commit tail is already inside the next round's elapsed, and the
        # finalize reserve covers the last one.
        row["seconds"] = time.monotonic() - t0
        if wandb_run is not None:
            wandb_run.log({k: v for k, v in row.items()
                           if k not in ("round", "backend")}, step=rnd)
        (d / "metrics" / f"round_{rnd:03d}.json").write_text(json.dumps(
            {**row, "per_class": per_class_from_confusion(cm, class_names)}, indent=2))
        C.append_history(d, row)
        C.mark_complete(d, rnd)
        hist.append(row)
        print(f"[r{rnd:03d}] f1_macro={met['f1_macro']:.6f} acc={met['accuracy']:.6f} "
              f"ce={row['ce_client_mean']:.4f} kd={row['kd_client_mean']:.4f} "
              f"skip={row['skipped']}/{row['steps']} "
              f"vram={row['vram_train_gb']:.2f}G {row['seconds']:.1f}s "
              f"| session {(time.monotonic()-t_start)/3600:.2f}h", flush=True)

        worst = max(h["seconds"] for h in hist)
        if (time.monotonic() - t_start) + worst * 1.15 + reserve > cfg["max_seconds"]:
            print(f"[driver] stopping after round {rnd}: the next round plus a "
                  f"{reserve/60:.0f} min finalize reserve would exceed the session budget "
                  f"({cfg['max_seconds']/3600:.2f} h)", flush=True)
            break

    return hist


def write_manifest(cfg, class_names, spans, y_true_src=None, extra=None):
    """Everything needed to say what these numbers are, written once, next to them.

    Also the SECOND data gate. `data_id` is cheap and pre-decode, so it guards the resume
    before the parquet pass but can only see the feature order, class order and scaler.
    `content_id` is computed after the decode from the row counts and the class histogram --
    the things that actually change when the partition or the file contents change -- and is
    compared here against what the resumed checkpoint was trained on. Continuing on top of
    different data is not a warning; it makes the whole run meaningless."""
    d = C.run_dir(cfg["run_name"])
    mf = d / "reports" / "manifest.json"
    m = {"fingerprint": C.fingerprint(cfg),
         "cfg": {k: v for k, v in cfg.items()},
         "class_names": list(class_names),
         "n_clients": len(spans), "n_train": sum(h - l for l, h in spans.values()),
         "client_rows": {str(c): spans[c][1] - spans[c][0] for c in sorted(spans)},
         "torch": torch.__version__, "cuda": torch.version.cuda,
         "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if extra: m.update(extra)

    if mf.is_file():
        old = json.loads(mf.read_text())
        for k in ("content_id", "data_id"):
            a, b = old.get(k), m.get(k)
            if a is not None and b is not None and a != b:
                raise RuntimeError(
                    f"{k} changed: this run's checkpoints were trained on {a}, the data "
                    f"mounted now is {b}. Resuming across that is not a continuation.")
        m["sessions"] = int(old.get("sessions", 1)) + 1
        m["first_written"] = old.get("first_written", old.get("written"))
    else:
        m["sessions"], m["first_written"] = 1, m["written"]

    # y_true travels with the run. Without it, a downloaded run directory cannot check its
    # own predictions against its own confusion matrices -- the cache it came from is in
    # /kaggle/temp and is gone with the session.
    if y_true_src is not None:
        dst = d / "reports" / "y_true.u8.npy"
        if not dst.is_file():
            shutil.copyfile(y_true_src, dst)
        m["y_true"] = "reports/y_true.u8.npy"
    mf.write_text(json.dumps(m, indent=2))
    return mf
