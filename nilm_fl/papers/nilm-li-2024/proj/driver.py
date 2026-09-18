"""One persistent worker per GPU, spawned once for the whole run.

Both GPUs hold the entire partition and the whole test set, so any client can train on
whichever GPU is free (longest-first dispatch) and evaluation splits the N personalized
models between the two GPUs. Each worker also holds a resident copy of EVERY client's
personalized weights w_s^k (N x 1.58 MB) and of the aggregated proxy w_bar_r, kept in
sync by the driver after each round, so a train task carries only a client id and an
eval task only a list of ids.

Aggregation is re-sorted by client id so float addition order never depends on which
worker finished first, and every client re-seeds the default generator from (seed, round,
client) so its update does not depend on the schedule either. Different clients are
different models: they never form a process group.
"""
import json, math, shutil, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp

from proj.model import build_model, N_PARAMS
from proj.nilm import (layout, flatten, unflatten_into, client_update, aggregate, amp,
                       lr_at, make_optimizer, mutual_loss, ACC_KEYS)
from proj.evaluate import fold_bn, load_folded, eval_model
from proj.metrics import metrics_from_confusion, per_class_from_confusion, METRIC_KEYS
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


def _verdict(ref_z, got_z, ref_g, got_g, n_rows, label):
    """Decisive-row argmax agreement plus bounded logit/gradient deltas. Counts are
    integers: torch.mean on CUDA returns 0.99999994 for a perfect match."""
    dz = (got_z - ref_z).abs().max().item()
    gn = ref_g.norm().item()
    dg = (got_g - ref_g).norm().item() / (gn + 1e-12)
    flip_all = int((got_z.argmax(1) != ref_z.argmax(1)).sum())
    top2 = ref_z.topk(2, dim=1).values
    decisive = (top2[:, 0] - top2[:, 1]) > max(10 * dz, 1e-3)
    n_dec = int(decisive.sum())
    flip_dec = int((got_z.argmax(1) != ref_z.argmax(1))[decisive].sum())
    # `flip_dec == 0` over an EMPTY decisive set says nothing at all.
    if n_dec < n_rows // 10:
        raise RuntimeError(f"{label}: cannot certify, only {n_dec} of {n_rows} rows have "
                           f"a margin above {max(10 * dz, 1e-3):.2e}")
    if flip_dec or not math.isfinite(dz) or not math.isfinite(dg) or dz > 5e-2 or dg > 5e-2:
        raise RuntimeError(f"{label}: mismatch dlogit={dz} dgrad_rel={dg} "
                           f"flips {flip_dec}/{n_dec} decisive, {flip_all} of all")
    return f"max|dlogit|={dz:.2e} rel|dgrad|={dg:.2e} flips {flip_all}/{n_rows} ({flip_dec}/{n_dec} decisive)"


def _compile_train(Se, Re, cfg, dev, xb, yb, rank=0):
    """reduce-overhead captures forward+backward into a CUDA graph. torch.compile is lazy,
    so a try around the call catches nothing -- run the production mutual step and compare
    against eager from the SAME state, with Dropout off on BOTH sides (Inductor
    functionalises RNG, so the two masks can never coincide). Warm-up captures the graphs
    at the production p, so restoring p reuses those entries."""
    if not cfg["compile"]:
        return Se, Re
    mods = list(Se.modules()) + list(Re.modules())
    drops = [m for m in mods if isinstance(m, nn.Dropout)]
    keep = [m.p for m in drops]
    snap = [{k: v.detach().clone() for k, v in m.state_dict().items()} for m in (Se, Re)]
    rng = torch.get_rng_state()
    crng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def restore():
        with torch.no_grad():
            for m, s in zip((Se, Re), snap):
                sd = m.state_dict()
                for k, v in s.items():
                    sd[k].copy_(v)                  # copy_ keeps addresses -> graph stays valid
        torch.set_rng_state(rng)
        if crng is not None: torch.cuda.set_rng_state_all(crng)
        Se.zero_grad(set_to_none=True); Re.zero_grad(set_to_none=True)
        Se.train(); Re.train()

    def probe(Sm, Rm):
        """The production step: both forwards under autocast, Eq. (17)-(19) in fp32, one
        backward. Returns the two logit blocks and the two flat gradients."""
        restore()
        if xb.is_cuda:
            torch.compiler.cudagraph_mark_step_begin()
        with amp(cfg):
            z_s = Sm(xb)
            z_r = Rm(xb)
        z_s, z_r = z_s.float(), z_r.float()
        loss, _ = mutual_loss(z_s, z_r, yb)
        loss.backward()
        z_s, z_r = z_s.clone(), z_r.clone()
        g_s = torch.cat([p.grad.reshape(-1).float().clone() for p in Se.parameters()])
        g_r = torch.cat([p.grad.reshape(-1).float().clone() for p in Re.parameters()])
        Se.zero_grad(set_to_none=True); Re.zero_grad(set_to_none=True)
        return z_s, g_s, z_r, g_r

    try:
        Sc = torch.compile(Se, mode="reduce-overhead")     # CUDA graphs: the 2.9x on T4
        Rc = torch.compile(Re, mode="reduce-overhead")
        for _ in range(3):                                  # warm up + capture at production p
            probe(Sc, Rc)
        for m in drops: m.p = 0.0
        try:
            ref = probe(Se, Re)
            got = probe(Sc, Rc)
        finally:
            for m, p_ in zip(drops, keep): m.p = p_
        restore()
        n = xb.shape[0]
        s1 = _verdict(ref[0], got[0], ref[1], got[1], n, "w_s")
        s2 = _verdict(ref[2], got[2], ref[3], got[3], n, "w_r")
        print(f"[rank{rank}] compile OK | w_s {s1} | w_r {s2}", flush=True)
        return Sc, Rc
    except Exception as e:                        # sm_75 Triton is the documented risk
        for m, p_ in zip(drops, keep): m.p = p_   # never leave the model with dropout off
        restore()
        print(f"[rank{rank}] compile DISABLED -> eager: {e}", flush=True)
        return Se, Re


def _compile_eval(Te, cfg, dev, xt, rank=0):
    """The folded eval template, compiled at the fixed eval batch. Certified against the
    eager folded template on real test rows: fp16 cannot be bit-equal, so the criterion is
    the decisive-row argmax rule with a delta ceiling."""
    if not cfg["compile"]:
        return Te
    try:
        Tc = torch.compile(Te, mode="reduce-overhead")
        with torch.inference_mode():
            for _ in range(3):
                with amp(cfg):
                    Tc(xt).float()
            with amp(cfg):
                ref = Te(xt).float().clone()
                got = Tc(xt).float().clone()
        z = torch.zeros(1, device=dev)
        s = _verdict(ref, got, z, z, xt.shape[0], "eval")
        print(f"[rank{rank}] eval compile OK | {s}", flush=True)
        return Tc
    except Exception as e:
        print(f"[rank{rank}] eval compile DISABLED -> eager: {e}", flush=True)
        return Te


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
    if cuda:
        torch.cuda.set_device(dev)
        torch.backends.cudnn.benchmark = True
    # S, R and the folded eval template share ONE code object (DAGSNet.forward), and Dynamo
    # caches per code object: 2 S variants (dropout on/off for the gate) + 2 R variants +
    # the eval template + mode toggles can pass the default recompile limit of 8. Past the
    # limit Dynamo runs the new variant EAGERLY without raising, so the eval template would
    # silently lose CUDA graphs (measured in a sibling project: "compiled" eval 0.7x eager).
    for name in ("recompile_limit", "cache_size_limit"):
        if hasattr(torch._dynamo.config, name):
            setattr(torch._dynamo.config, name, 64)
    torch.manual_seed(cfg["seed"] + rank)
    cache = Path(cfg["cache"])
    X = _resident(cache / "train_X.f16.npy", dev)
    Y = _resident(cache / "train_y.u8.npy", dev)
    TX = _resident(cache / "test_X.f16.npy", dev)
    TY = _resident(cache / "test_y.u8.npy", dev)

    Se = build_model(cfg).to(dev).train()                 # personalized model w_s (per task)
    Re = build_model(cfg).to(dev).train()                 # proxy w_r (per task)
    fk, ik, nP = layout(Se)
    assert nP == N_PARAMS, nP
    B = cfg["batch"]
    # Probe on real rows: random N(0,1) has none of the heavy tails of the z-scored
    # features, and a kernel that is wrong only at large magnitude would pass on noise.
    Sc, Rc = _compile_train(Se, Re, cfg, dev, X[:B].float(), Y[:B].long(), rank)
    Te = fold_bn(build_model(cfg).to(dev))            # folded STRUCTURE; weights per model
    Tc = _compile_eval(Te, cfg, dev, TX[:cfg["eval_batch"]].float(), rank)
    scaler_probe = torch.amp.GradScaler("cuda", enabled=cuda)
    if cuda:
        scaler_probe.scale(torch.zeros(1, device=dev))  # force _scale to exist
        assert scaler_probe._scale is not None, "GradScaler._scale gone; skips would read as 0"
    res_q.put(("ready", rank, "eager" if Sc is Se else "compiled",
               "eager" if Tc is Te else "compiled"))

    CW = CI = GW = GI = None
    while True:
        task = task_q.get()
        kind = task[0]
        if kind == "stop":
            return
        if kind == "backend":
            # Both ranks gate independently, so one can compile and the other fall back.
            # A round whose clients were trained on two different backends is not a round
            # anyone can reproduce; the driver forces the lower common denominator.
            if task[1] == "eager": Sc, Rc = Se, Re
            if task[2] == "eager": Tc = Te
            res_q.put(("backend_ok", rank, "eager" if Sc is Se else "compiled",
                       "eager" if Tc is Te else "compiled"))
            continue
        # Payloads cross the process boundary as numpy arrays: a torch tensor on a
        # multiprocessing queue is shared through /dev/shm, which a container may cap at
        # 64 MB, and the resident client table is 160 MB at 100 clients.
        if kind == "init":
            CW, CI = torch.from_numpy(task[1]).to(dev), torch.from_numpy(task[2]).to(dev)
            GW, GI = torch.from_numpy(task[3]).to(dev), torch.from_numpy(task[4]).to(dev)
            res_q.put(("init_ok", rank))
            continue
        if kind == "set_clients":
            cids = task[1]
            fvs, ivs = torch.from_numpy(task[2]).to(dev), torch.from_numpy(task[3]).to(dev)
            for j, c in enumerate(cids):
                CW[c].copy_(fvs[j]); CI[c].copy_(ivs[j])
            res_q.put(("set_ok", rank))
            continue
        if kind == "set_global":
            GW.copy_(torch.from_numpy(task[1]).to(dev)); GI.copy_(torch.from_numpy(task[2]).to(dev))
            res_q.put(("set_ok", rank))
            continue
        if kind == "eval":
            cids, want_preds, with_global = task[1], task[2], task[3]
            t0 = time.monotonic()
            if cuda: torch.cuda.reset_peak_memory_stats(dev)
            cms, nfs, preds = [], [], []
            for c in cids:
                unflatten_into(Se, CW[c], CI[c], fk, ik)
                load_folded(Te, Se)
                cm, nf, p = eval_model(Tc, Te, TX, TY, cfg, want_preds)
                cms.append(cm.cpu()); nfs.append(nf)
                if want_preds: preds.append(p.cpu())
            g = None
            if with_global:                                 # the aggregated proxy w_bar_r
                unflatten_into(Re, GW, GI, fk, ik)
                load_folded(Te, Re)
                cm, nf, p = eval_model(Tc, Te, TX, TY, cfg, want_preds)
                g = (cm.cpu().numpy(), nf, p.cpu().numpy() if want_preds else None)
            res_q.put(("eval", rank, list(cids),
                       torch.stack(cms).numpy() if cms else None, nfs,
                       torch.stack(preds).numpy() if preds else None, g,
                       (torch.cuda.max_memory_allocated(dev) / 2**30) if cuda else 0.0,
                       time.monotonic() - t0))
            continue
        if kind == "train":
            cid, lo, hi, rnd = task[1], task[2], task[3], task[4]
            t0 = time.monotonic()
            if cuda: torch.cuda.reset_peak_memory_stats(dev)
            unflatten_into(Se, CW[cid], CI[cid], fk, ik)      # w_s^k from the previous round
            unflatten_into(Re, GW, GI, fk, ik)                # w_r^k <- w_bar_r from the server
            # AdamW re-created per client per round (owner's decision): w_bar_r arrives
            # fresh from the server, and w_s's moments are reset with it. fused=True
            # collapses the step into one multi-tensor kernel. The rate is the round's
            # value of the schedule (proj.nilm.lr_at), the same for both models.
            lr = lr_at(cfg, rnd)
            opt = make_optimizer(Se, Re, lr, cfg, fused=cuda)
            scaler = torch.amp.GradScaler("cuda", enabled=cuda)
            if cuda:
                scaler.scale(torch.zeros(1, device=dev))
            # Every stochastic input to this client derives from (seed, round, client):
            # the default generator drives Dropout, `g` drives the shuffles.
            s = cfg["seed"] * 1_000_003 + rnd * 10_007 + cid
            torch.manual_seed(s)
            g = torch.Generator(device=dev); g.manual_seed(s)
            acc, n = client_update(Sc, Se, Rc, Re, opt, scaler, X, Y, lo, hi, cfg, g)
            sv, si = flatten(Se, fk, ik)
            rv, ri = flatten(Re, fk, ik)
            a = acc.cpu().tolist()
            sk = int(a[len(ACC_KEYS)]); ap = n - sk                # NOT max(1, .): 0 must stay 0
            d_ = max(1, ap)
            st = {"round": rnd, "cid": cid, "n_k": hi - lo, "rank": rank, "seed": s,
                  "lr": lr, "steps": n, "applied": ap, "skipped": sk,
                  "nonfinite": int(a[len(ACC_KEYS) + 1]),
                  "sec": time.monotonic() - t0,
                  "vram_gb": (torch.cuda.max_memory_allocated(dev) / 2**30 if cuda else 0.0)}
            st.update({k: a[j] / d_ for j, k in enumerate(ACC_KEYS)})   # means over applied steps
            res_q.put(("train", cid, hi - lo, sv.cpu().numpy(), si.cpu().numpy(),
                       rv.cpu().numpy(), ri.cpu().numpy(), st))


def check_updates(rnd, results, stats, n_clients, max_skips=None):
    """Every reason a round must not be aggregated, in one pure function so it can be
    tested without two GPUs and a spawned worker.

    Skipped steps are NOT a failure: each client starts a fresh GradScaler at 2**16 and
    spends a few steps calibrating. What must be rejected is a client that applied no
    step, one that skipped far more than calibration explains, one whose APPLIED steps
    carried a non-finite gradient, and non-finite weights."""
    if sorted(stats) != list(range(n_clients)):
        raise RuntimeError(f"round {rnd}: reported {sorted(stats)}, expected 0..{n_clients - 1}")
    bad = [cid for cid, _, sv, _, rv, _, _ in results
           if not (np.isfinite(sv).all() and np.isfinite(rv).all())]
    if bad:
        raise RuntimeError(f"round {rnd}: non-finite weights from clients {bad}")
    for c in sorted(stats):
        st = stats[c]
        if st["applied"] + st["skipped"] != st["steps"]:
            raise RuntimeError(f"round {rnd}: client {c} applied+skipped != steps")
    dead = [c for c in sorted(stats) if stats[c]["applied"] == 0]
    if dead:
        raise RuntimeError(f"round {rnd}: clients {dead} applied zero steps; "
                           "they would contribute unchanged weights")
    diverged = [c for c in sorted(stats) if stats[c]["nonfinite"]]
    if diverged:
        raise RuntimeError(f"round {rnd}: clients {diverged} APPLIED a step whose "
                           "gradient was not finite")
    if max_skips is not None:
        over = [c for c in sorted(stats) if stats[c]["skipped"] > max_skips]
        if over:
            raise RuntimeError(
                f"round {rnd}: clients {over} skipped more steps than the warm-up "
                f"budget ({max_skips}): "
                + ", ".join(f"{c}={stats[c]['skipped']}" for c in over))


def _collect(res_q, procs, n, timeout=7200):
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


def _broadcast(task_qs, res_q, procs, msg):
    for q in task_qs: q.put(msg)
    return _collect(res_q, procs, len(task_qs))


def run(cfg, spans, class_names, wandb_run=None, t_origin=None):
    """t_origin is a time.monotonic() reading from when the SESSION started, not from when
    this call did. Worker spawn, the resident copy and compilation are minutes the 12 h cap
    charges for, and a deadline that started here would happily begin a round the session
    cannot finish."""
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
        bt = {m[1]: m[2] for m in ready}; be = {m[1]: m[3] for m in ready}
        if len(set(bt.values())) > 1 or len(set(be.values())) > 1:
            print(f"[driver] ranks disagree on backend train={bt} eval={be}; forcing eager",
                  flush=True)
            force = ("backend", "eager" if len(set(bt.values())) > 1 else "keep",
                     "eager" if len(set(be.values())) > 1 else "keep")
            acks = _broadcast(task_qs, res_q, procs, force)
            bt = {m[1]: m[2] for m in acks}; be = {m[1]: m[3] for m in acks}
        cfg["backend"] = sorted(set(bt.values()))[0]
        cfg["backend_eval"] = sorted(set(be.values()))[0]
        startup = time.monotonic() - t_start
        print(f"[driver] {cfg['world_size']} workers ready: train {cfg['backend']}, "
              f"eval {cfg['backend_eval']} ({startup:.0f}s into the session)", flush=True)
        cfg["startup_seconds"] = startup
        # Push the effective backend somewhere READABLE WHILE THE RUN IS ALIVE: a running
        # Kaggle kernel's stdout cannot be downloaded.
        if wandb_run is not None:
            try:
                wandb_run.config.update({"backend": cfg["backend"],
                                         "backend_eval": cfg["backend_eval"],
                                         "startup_seconds": round(startup, 1)},
                                        allow_val_change=True)
                wandb_run.summary["backend"] = cfg["backend"]
                wandb_run.summary["backend_eval"] = cfg["backend_eval"]
            except Exception as e:
                print(f"[driver] could not publish backend to W&B: {e}", flush=True)
        return _rounds(cfg, spans, class_names, wandb_run, t_start, procs, task_qs, res_q)
    finally:
        # Without this a driver-side exception leaves two processes holding both GPUs, and
        # the next cell in the notebook fails with a CUDA OOM that names nothing.
        _shutdown(procs, task_qs)


def _stats(values):
    v = np.asarray(values, dtype=np.float64)
    return float(v.mean()), float(v.std()), float(v.min()), float(v.max())


def _rounds(cfg, spans, class_names, wandb_run, t_start, procs, task_qs, res_q):
    d = C.run_dir(cfg["run_name"])
    N, W = cfg["n_clients"], cfg["world_size"]
    torch.manual_seed(cfg["seed"])                 # seed BEFORE building: w^0 seeded
    M0 = build_model(cfg)
    fk, ik, nP = layout(M0)
    f0v, f0i = flatten(M0, fk, ik)
    # Every model starts from the SAME seeded initialization (owner's decision): the N
    # personalized models AND the proxy are copies of w^0.
    CW = f0v.unsqueeze(0).repeat(N, 1).contiguous()
    CI = f0i.unsqueeze(0).repeat(N, 1).contiguous()
    GW, GI = f0v.clone(), f0i.clone()

    start = 1
    last = C.resolve_resume(cfg["run_name"], cfg)
    if last is not None:
        R, Ss, _ = C.load_weights(d / "weights" / f"round_{last:03d}.pt", build_model, N_PARAMS)
        GW, GI = flatten(R, fk, ik)
        for c, m in Ss.items():
            CW[c], CI[c] = flatten(m, fk, ik)
        start = last + 1
        print(f"[driver] resumed at round {start}")
    elif cfg.get("require_resume"):
        raise SystemExit("require_resume set and no checkpoint found")
    if start > cfg["rounds"]:
        print(f"[driver] nothing to do: {last} rounds already complete")
        return []
    _broadcast(task_qs, res_q, procs, ("init", CW.numpy(), CI.numpy(), GW.numpy(), GI.numpy()))

    n_test = cfg["n_test"]
    preds_rounds = set(cfg.get("preds_rounds", [cfg["rounds"]]))
    hist = []
    reserve = cfg.get("finalize_reserve_seconds", 600)
    elapsed = time.monotonic() - t_start
    if elapsed + reserve >= cfg["max_seconds"]:
        print(f"[driver] no round started: {elapsed/3600:.2f} h of the "
              f"{cfg['max_seconds']/3600:.2f} h budget is already gone", flush=True)
        return hist

    for rnd in range(start, cfg["rounds"] + 1):
        t0 = time.monotonic()
        # Algorithm 1: every household trains in every round. Longest-first bounds the
        # idle tail: sending the biggest client last strands a GPU.
        order = sorted(range(N), key=lambda c: spans[c][1] - spans[c][0], reverse=True)
        pending, nxt, results = {}, 0, []
        for r in range(W):                                  # prime both GPUs
            if nxt < len(order):
                c = order[nxt]; nxt += 1
                task_qs[r].put(("train", c, *spans[c], rnd)); pending[r] = c
        while len(results) < len(order):
            msg = _collect(res_q, procs, 1)[0]
            assert msg[0] == "train", msg[0]
            results.append(msg[1:])
            r = next(k for k, v in pending.items() if v == msg[1])
            if nxt < len(order):
                c = order[nxt]; nxt += 1
                task_qs[r].put(("train", c, *spans[c], rnd)); pending[r] = c
            else:
                pending.pop(r)
        t_train = time.monotonic() - t0

        results.sort(key=lambda t: t[0])                    # NOT completion order
        stats = {cid: s for cid, _, _, _, _, _, s in results}
        check_updates(rnd, results, stats, N, cfg.get("max_skips_per_client"))
        GW, GI = aggregate([(cid, torch.from_numpy(rv), torch.from_numpy(ri))
                            for cid, _, _, _, rv, ri, _ in results])
        cids = [t[0] for t in results]
        svs = np.stack([t[2] for t in results]); sis = np.stack([t[3] for t in results])
        for j, c in enumerate(cids):
            CW[c].copy_(torch.from_numpy(svs[j])); CI[c].copy_(torch.from_numpy(sis[j]))
        _broadcast(task_qs, res_q, procs, ("set_clients", cids, svs, sis))
        _broadcast(task_qs, res_q, procs, ("set_global", GW.numpy(), GI.numpy()))

        # ---- evaluate every personalized model w_s^k and the aggregated proxy w_bar_r on
        # the full test set. Client c is always evaluated on worker c % W (the two workers
        # are separate processes whose cuDNN algorithm choice can differ on a few fp16
        # rows; a fixed assignment keeps each client's series on one GPU). The proxy goes
        # to the last worker, which holds the fewer clients when N is odd.
        want_preds = rnd in preds_rounds
        eval_split = [[c for c in range(N) if c % W == r] for r in range(W)]
        t1 = time.monotonic()
        for r in range(W):
            task_qs[r].put(("eval", eval_split[r], want_preds, r == W - 1))
        ev = _collect(res_q, procs, W)
        t_eval = time.monotonic() - t1
        fresh, g = {}, None
        preds = np.empty((N, n_test), dtype=np.uint8) if want_preds else None
        nf_total = 0
        for e in ev:
            for j, c in enumerate(e[2]):
                fresh[c] = e[3][j]; nf_total += e[4][j]
                if want_preds: preds[c] = e[5][j]
            if e[6] is not None:
                g = e[6]; nf_total += g[1]
        assert sorted(fresh) == list(range(N)) and g is not None, f"evaluated {sorted(fresh)}"
        if nf_total:
            raise RuntimeError(f"round {rnd}: {nf_total} non-finite test logits; argmax "
                               "would have turned them into ordinary class labels")
        cm = np.stack([fresh[c] for c in range(N)])
        gcm = g[0]
        assert (cm.sum(axis=(1, 2)) == n_test).all() and gcm.sum() == n_test, \
            "a confusion matrix misses test rows"
        per_client = []
        for c in range(N):
            m = metrics_from_confusion(cm[c])
            per_client.append({"cid": c, **m,
                               "per_class": per_class_from_confusion(cm[c], class_names)})
        gm = metrics_from_confusion(gcm)
        row = {"round": rnd, "evaluated": N, "lr": lr_at(cfg, rnd)}
        for k in METRIC_KEYS:                                # mean over ALL N clients
            mean, std, lo, hi = _stats([pc[k] for pc in per_client])
            row[k] = mean
            row[f"{k}_std"], row[f"{k}_min"], row[f"{k}_max"] = std, lo, hi
        for k in METRIC_KEYS:                                # the aggregated proxy
            row[f"global_{k}"] = gm[k]
        # *_client_mean is the unweighted mean ACROSS CLIENTS of each client's mean over
        # its applied steps -- not the mean over training samples.
        for k in ACC_KEYS:
            row[f"{k}_client_mean"] = float(np.mean([s[k] for s in stats.values()]))
        row.update({
            "steps": int(sum(s["steps"] for s in stats.values())),
            "skipped": int(sum(s["skipped"] for s in stats.values())),
            "train_sec": t_train, "eval_sec": t_eval,
            "vram_train_gb": max(s["vram_gb"] for s in stats.values()),
            "vram_eval_gb": max(e[7] for e in ev),
            "backend": cfg.get("backend", "?"), "seconds": 0.0})
        mean_metrics = {k: row[k] for k in METRIC_KEYS}

        # ---- commit. Marker absolutely last.
        unflatten_into(M0, GW, GI, fk, ik)
        global_sd = C.cpu_sd(M0)
        clients_sd = {}
        for c in range(N):
            unflatten_into(M0, CW[c], CI[c], fk, ik)
            clients_sd[c] = C.cpu_sd(M0)
        C.save_round_weights(global_sd, clients_sd, rnd, cfg, mean_metrics, gm, d)
        C.atomic_np_save(d / "confusion" / f"round_{rnd:03d}.npy", cm)
        C.atomic_np_save(d / "confusion" / f"global_{rnd:03d}.npy", gcm)
        if want_preds:
            C.atomic_np_save(d / "preds" / f"round_{rnd:03d}.u8.npy", preds)
            C.atomic_np_save(d / "preds" / f"global_{rnd:03d}.u8.npy", g[2])
        (d / "logs" / f"round_{rnd:03d}.json").write_text(
            json.dumps({"clients": [stats[c] for c in sorted(stats)]}, indent=1))
        # `seconds` BEFORE the W&B call and the JSON: the budget below compares absolute
        # session elapsed, so the commit tail lands in the next round's elapsed and the
        # finalize reserve covers the last one.
        row["seconds"] = time.monotonic() - t0
        if wandb_run is not None:
            # W&B is a monitor, never a dependency: in a resumed session (50c s2, 16-09)
            # the service died at startup and every later log call went nowhere. The
            # round's artifacts below are the record; a W&B failure must not touch them.
            try:
                wandb_run.log({k: v for k, v in row.items()
                               if k not in ("round", "backend")}, step=rnd)
            except Exception as e:
                print(f"[driver] W&B log failed at round {rnd}: {e}", flush=True)
        (d / "metrics" / f"round_{rnd:03d}.json").write_text(json.dumps(
            {**row, "clients": per_client,
             "global": {**gm, "per_class": per_class_from_confusion(gcm, class_names)}},
            indent=1))
        C.append_history(d, row, [{"round": rnd, **{k: v for k, v in pc.items()
                                                    if k != "per_class"}}
                                  for pc in per_client])
        C.mark_complete(d, rnd)
        hist.append(row)
        print(f"[r{rnd:03d}] f1_macro mean={row['f1_macro']:.6f} "
              f"std={row['f1_macro_std']:.4f} min={row['f1_macro_min']:.4f} "
              f"acc={row['accuracy']:.6f} | proxy f1={row['global_f1_macro']:.6f} "
              f"| lr={row['lr']:.2e} ce_s={row['ce_s_client_mean']:.4f} "
              f"ce_r={row['ce_r_client_mean']:.4f} kl_s={row['kl_s_client_mean']:.4f} "
              f"skip={row['skipped']}/{row['steps']} "
              f"train={t_train:.0f}s eval={t_eval:.0f}s vram={row['vram_train_gb']:.2f}G "
              f"{row['seconds']:.1f}s | session {(time.monotonic()-t_start)/3600:.2f}h",
              flush=True)

        worst = max(h["seconds"] for h in hist)
        if (time.monotonic() - t_start) + worst * 1.15 + reserve > cfg["max_seconds"]:
            print(f"[driver] stopping after round {rnd}: the next round plus a "
                  f"{reserve/60:.0f} min finalize reserve would exceed the session budget "
                  f"({cfg['max_seconds']/3600:.2f} h)", flush=True)
            break

    return hist


def write_manifest(cfg, class_names, spans, y_true_src=None, extra=None):
    """Everything needed to say what these numbers are, written once, next to them.

    Also the SECOND data gate: `content_id` is computed after the decode from the row
    counts and the class histogram and compared against what the resumed checkpoint was
    trained on. Continuing on top of different data is not a warning."""
    d = C.run_dir(cfg["run_name"])
    mf = d / "reports" / "manifest.json"
    m = {"fingerprint": C.fingerprint(cfg),
         "cfg": {k: v for k, v in cfg.items()},
         "class_names": list(class_names),
         "n_clients": len(spans), "n_train": sum(h - l for l, h in spans.values()),
         "client_rows": {str(c): spans[c][1] - spans[c][0] for c in sorted(spans)},
         "n_params": N_PARAMS,
         "torch": torch.__version__, "cuda": torch.version.cuda,
         "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if extra: m.update(extra)

    # A handoff bundle carries the ids of the data its rounds were trained on; a resumed
    # session has no earlier manifest, so this is where that gate fires for it.
    old = C.read_handoff(d) if not mf.is_file() else None
    if old is not None:
        for k in ("content_id", "data_id"):
            a, b = old.get(k), m.get(k)
            if a is not None and b is not None and a != b:
                raise RuntimeError(
                    f"{k} changed: the handoff's rounds were trained on {a}, the data "
                    f"mounted now is {b}. Resuming across that is not a continuation.")
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

    # y_true travels with the run: a downloaded run directory must be able to check its
    # own predictions against its own confusion matrices.
    if y_true_src is not None:
        dst = d / "reports" / "y_true.u8.npy"
        if not dst.is_file():
            shutil.copyfile(y_true_src, dst)
        m["y_true"] = "reports/y_true.u8.npy"
    mf.write_text(json.dumps(m, indent=2))
    return mf
