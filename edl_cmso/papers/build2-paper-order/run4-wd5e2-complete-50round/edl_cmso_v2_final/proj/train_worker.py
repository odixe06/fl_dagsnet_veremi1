import os, sys, json, time, math
from datetime import timedelta
import numpy as np, torch, torch.nn as nn, torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast, GradScaler

sys.path.insert(0, "/kaggle/working")
from proj import ckpt as C
from proj.model import build_model, load_extractor_state
from proj.metrics import save_round_artifacts

TIMEOUT_MIN = 60          # overwritten from cfg in main_worker before ddp_setup runs


def ddp_setup(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    # set_device BEFORE init: otherwise every rank builds its NCCL communicator on cuda:0.
    torch.cuda.set_device(rank)
    # PyTorch's NCCL default is 10 min. Rank 1 sits in the round barrier for the whole of
    # rank 0's 10.7M-row eval + artifact write; if that ever exceeds the timeout the
    # watchdog aborts the run. 60 min is generous enough to never be the cause of a kill.
    dist.init_process_group("nccl", rank=rank, world_size=world,
                            timeout=timedelta(minutes=int(TIMEOUT_MIN)))


def load_shard(path_X, path_y, rank, world, device):
    """This rank's slice of train, resident on its GPU as fp16.

    Build 2 loads ALL 66 columns: selection happens on the fused channels inside the
    model, not on the input, so nothing is dropped here. Both ranks take exactly n_per
    rows so the epoch length — and therefore the number of all_reduce calls — is
    identical; a ragged split deadlocks the first collective after the short rank ends."""
    X = np.load(path_X, mmap_mode="r"); Y = np.load(path_y, mmap_mode="r")
    n_per = X.shape[0] // world
    lo, hi = rank * n_per, (rank + 1) * n_per
    chunks_x, chunks_y = [], []
    for i in range(lo, hi, 4_000_000):                       # stream: bounded host RAM
        j = min(i + 4_000_000, hi)
        chunks_x.append(torch.from_numpy(np.asarray(X[i:j])).to(device))
        chunks_y.append(torch.from_numpy(np.asarray(Y[i:j]).astype(np.int64)).to(device))
    return torch.cat(chunks_x), torch.cat(chunks_y)


def load_test_resident(path_X, path_y, device):
    """Rank 0 keeps the whole test set on its own GPU as fp16, loaded once for the run.
    The eval loop used to fancy-index a memmap and copy host->device once per batch,
    every round, on 4 vCPU already shared with the other rank — that, not the model, was
    what made evaluation slow. 10,761,343 x 66 fp16 = 1.32 GB."""
    X = np.load(path_X, mmap_mode="r"); Y = np.load(path_y, mmap_mode="r")
    n = X.shape[0]
    chunks = []
    for i in range(0, n, 4_000_000):
        j = min(i + 4_000_000, n)
        chunks.append(torch.from_numpy(np.asarray(X[i:j])).to(device))
    return torch.cat(chunks), np.asarray(Y).astype(np.int8)


@torch.inference_mode()
def evaluate_full(model, Xte, prob_buf, cfg):
    """Rank 0 only. The ENTIRE test set, exactly once, in file order — no
    DistributedSampler, so no padded or dropped tail. Probabilities are written into a
    resident GPU buffer; they only reach the host on the rounds that are actually saved."""
    model.eval()
    n, B = Xte.shape[0], cfg["eval_batch"]
    preds = torch.empty(n, dtype=torch.int8, device=Xte.device)
    for i in range(0, n, B):
        j = min(i + B, n)
        with autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
            logits = model(Xte[i:j].float())
        pr = torch.softmax(logits.float(), dim=1)
        prob_buf[i:j] = pr.to(torch.float16)
        preds[i:j] = pr.argmax(1).to(torch.int8)
    model.train()
    return preds.cpu().numpy()


def main_worker(rank, world, cfg):
    global TIMEOUT_MIN
    TIMEOUT_MIN = cfg.get("nccl_timeout_min", 60)
    ddp_setup(rank, world)
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0
    torch.manual_seed(cfg["seed"] + rank); np.random.seed(cfg["seed"] + rank)
    torch.backends.cudnn.benchmark = True                    # fixed input shapes

    sel_ch = np.asarray(cfg["sel_ch"], dtype=np.int64)       # FUSED channels, not columns

    Xs, Ys = load_shard(cfg["train_X"], cfg["train_y"], rank, world, device)
    if is_main:
        print(f"[rank0] shard {tuple(Xs.shape)} fp16 = "
              f"{Xs.element_size()*Xs.nelement()/2**30:.2f} GB resident", flush=True)

    model = build_model(cfg, cfg["n_features"], sel_ch.tolist()).to(device)

    # THE CONTRACT OF BUILD 2. The CMSO mask was chosen against the fused map emitted by
    # §4.8 at initialisation. Training must therefore begin from that exact projection —
    # a freshly seeded §4.8 here would leave the mask describing a network that never
    # existed. On resume the checkpoint supplies everything and this is skipped.
    if not (C.run_dir(cfg["run_name"]) / "checkpoints" / "last.pt").exists():
        sd = torch.load(cfg["init_state"], map_location="cpu")
        n = load_extractor_state(model, sd)
        if is_main:
            print(f"[rank0] §4.8 restored from extractor_init.pt ({n} tensors) — the mask "
                  f"and the network now refer to the same projection", flush=True)

    if is_main:
        print(f"[rank0] params: {sum(p.numel() for p in model.parameters()):,}   "
              f"fused {model.fused_dim} -> {len(sel_ch)} channels into DAGSNet", flush=True)
    model = DDP(model, device_ids=[rank], output_device=rank,
                gradient_as_bucket_view=True, find_unused_parameters=False)

    # torch.compile — measured, not assumed. DDP FIRST, then compile: that is the order
    # DDPOptimizer needs in order to break the graph on gradient-bucket boundaries.
    # Default mode, not "reduce-overhead": CUDA graphs bought a further 5% locally but
    # they interact badly with DDP + GradScaler, and this run has no quota to spend on a
    # crash. A trial step below proves the whole fwd+bwd path really compiles; if anything
    # at all goes wrong we drop back to eager and the run still happens. Both ranks run
    # the identical trial, so DDP stays in step either way.
    if cfg.get("compile_model", False):
        eager = model
        try:
            model = torch.compile(model)
            xt = torch.zeros(cfg["batch_per_gpu"], cfg["n_features"], device=device)
            yt = torch.zeros(cfg["batch_per_gpu"], dtype=torch.long, device=device)
            with autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
                lt = model(xt)
            torch.nn.functional.cross_entropy(lt.float(), yt).backward()
            model.zero_grad(set_to_none=True)
            if is_main:
                print("[rank0] torch.compile active (trial fwd+bwd passed)", flush=True)
        except Exception as e:
            model = eager
            model.zero_grad(set_to_none=True)
            if is_main:
                print(f"[rank0] torch.compile failed ({type(e).__name__}: {e}) — "
                      f"running eager", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = GradScaler("cuda", enabled=cfg["amp"])

    # Table 1: "0.001 with adaptive decay". The schedule is unstated (deviation 10):
    # 1 warmup round, then cosine to zero across the remaining rounds.
    def lr_at(rnd):
        w = cfg["warmup_rounds"]
        if rnd < w: return (rnd + 1) / max(w, 1)
        prog = (rnd - w) / max(cfg["rounds"] - w, 1)
        return 0.5 * (1 + math.cos(math.pi * prog))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    crit = nn.CrossEntropyLoss()

    # rank 0 alone imports from /kaggle/input — concurrent copies race. After the barrier
    # every rank loads the identical file out of working/.
    if cfg["resume"] and is_main:
        C.resolve_resume(cfg["run_name"])
    dist.barrier()
    start_round = C.load_for_resume(model, opt, scaler, sched, cfg,
                                    cfg["run_name"], device) if cfg["resume"] else 0

    B = cfg["batch_per_gpu"]
    steps = Xs.shape[0] // B
    if is_main:
        print(f"[rank0] {steps:,} steps/epoch/rank  (global batch "
              f"{B * world * cfg['grad_accum']})", flush=True)

    # rank 0 alone evaluates, so only rank 0 pays for the resident test set
    Xte = y_true = prob_buf = None
    if is_main:
        Xte, y_true = load_test_resident(cfg["test_X"], cfg["test_y"], device)
        prob_buf = torch.empty((Xte.shape[0], cfg["num_classes"]),
                               dtype=torch.float16, device=device)
        print(f"[rank0] test {tuple(Xte.shape)} fp16 = "
              f"{Xte.element_size()*Xte.nelement()/2**30:.2f} GB resident "
              f"+ {prob_buf.element_size()*prob_buf.nelement()/2**30:.2f} GB prob buffer",
              flush=True)

    # ---- observability: rank 0 owns the single W&B run ------------------------
    # Kaggle publishes a kernel's output only when it stops, so without this the run is
    # invisible for its whole 8 h. The JSONL heartbeat is written too and is committed
    # with the output, so the evidence survives a network or account problem.
    HEART = C.run_dir(cfg["run_name"]) / "logs" / "heartbeat.jsonl"
    WB = None
    if is_main:
        try:
            import wandb
            WB = wandb.init(entity=cfg["wandb_entity"], project=cfg["wandb_project"],
                            id=cfg["wandb_run_id"], name=cfg["run_name"], config=cfg,
                            resume="allow", mode="online",
                            dir=str(C.run_dir(cfg["run_name"])))
            WB.define_metric("progress/global_step")
            WB.define_metric("*", step_metric="progress/global_step")
            print(f"[rank0] W&B run: {WB.url}", flush=True)
        except Exception as e:
            print(f"[rank0] W&B unavailable ({type(e).__name__}: {e}) — "
                  f"the JSONL heartbeat still records everything", flush=True)

    def emit(rec, payload=None, gstep=None):
        with HEART.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print("  " + json.dumps(rec), flush=True)   # flush: a killed kernel loses buffers
        if WB is not None and payload is not None:
            WB.log(payload, step=gstep)

    def attn_stats(core, xb):
        """|logit|max and softmax entropy inside vit.0 — the quantities that predicted
        every divergence this project has had. fp32, no grad, one small slice."""
        blk = core.vit[0]
        with torch.no_grad(), autocast("cuda", enabled=False):
            xw = core.dwt(xb[:512].float()); _, z = core.patch(xw); h = blk.n1(z)
            qkv = torch.nn.functional.linear(h, blk.att.in_proj_weight, blk.att.in_proj_bias)
            d = z.shape[-1]; H = blk.att.num_heads; hd = d // H
            q, k, _ = qkv.chunk(3, -1)
            q = q.view(z.shape[0], -1, H, hd).transpose(1, 2)
            k = k.view(z.shape[0], -1, H, hd).transpose(1, 2)
            lg = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
            pr = lg.softmax(-1)
            return float(lg.abs().max()), float(-(pr * (pr + 1e-12).log()).sum(-1).mean())

    LOG_EVERY = cfg.get("heartbeat_every", 250)
    outcome = "running"

    deadline = cfg.get("deadline_ts", float("inf"))
    best_f1 = -1.0
    hist_p = C.run_dir(cfg["run_name"]) / "metrics" / "history.csv"
    if hist_p.exists():
        import csv as _csv
        with open(hist_p) as f:
            best_f1 = max([float(r["f1_macro"]) for r in _csv.DictReader(f)] or [-1.0])

    for rnd in range(start_round, cfg["rounds"]):
        t0 = time.time(); model.train()
        running = torch.zeros(2, device=device)
        skipped = win_skip = win_ok = 0
        aborted = False
        step_offset = rnd * steps

        # ---------------- ROUND BODY: one epoch block ----------------
        for ep in range(cfg["epochs_per_round"]):
            g = torch.Generator(device=device)
            g.manual_seed(cfg["seed"] * 100003 + rnd * 1009 + ep)   # same stream on both ranks
            perm = torch.randperm(Xs.shape[0], generator=g, device=device)
            opt.zero_grad(set_to_none=True)
            for s in range(steps):
                idx = perm[s * B:(s + 1) * B]
                xb = Xs[idx].float(); yb = Ys[idx]
                with autocast("cuda", dtype=torch.float16, enabled=cfg["amp"]):
                    logits = model(xb)
                # Cross-entropy in fp32, OUTSIDE autocast. A softmax over 16 half-precision
                # logits can round a class probability to exactly 0, and log(0) is -inf —
                # one of the two routes build 2 had to a non-finite loss. Costs nothing:
                # the tensor is (B, 16).
                loss = crit(logits.float(), yb) / cfg["grad_accum"]
                scaler.scale(loss).backward()
                if (s + 1) % cfg["grad_accum"] == 0:
                    scaler.unscale_(opt)
                    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["clip"])
                    # GradScaler discards a non-finite step and reports NOTHING: a run can
                    # throw away most of its batches while printing a falling loss. The
                    # scale halving is the only observable signal, so read it and count.
                    prev = scaler.get_scale()
                    scaler.step(opt); scaler.update()
                    if scaler.get_scale() < prev:
                        skipped += 1; win_skip += 1
                    else:
                        win_ok += 1
                    opt.zero_grad(set_to_none=True)
                running += torch.stack([loss.detach() * cfg["grad_accum"],
                                        torch.ones((), device=device)])

                want_abort = False
                if is_main and (s + 1) % LOG_EVERY == 0:
                    gstep = step_offset + s + 1
                    skip_pct = 100.0 * win_skip / max(win_ok + win_skip, 1)
                    gn = float(gnorm)
                    core = model.module if hasattr(model, "module") else model
                    core = getattr(core, "_orig_mod", core)      # unwrap torch.compile
                    lgmax, ent = attn_stats(core, xb)
                    wqkv = float(core.vit[0].att.in_proj_weight.norm())
                    with torch.no_grad():
                        mw = float(max(p.abs().max() for p in model.parameters()))
                    rec = {"round": rnd, "step": s + 1, "of": steps, "global_step": gstep,
                           "loss": round(float(loss) * cfg["grad_accum"], 5),
                           "grad_norm": round(gn, 5) if math.isfinite(gn) else "nonfinite",
                           "Wqkv_norm": round(wqkv, 3), "abs_logit_max": round(lgmax, 2),
                           "softmax_entropy": round(ent, 4),
                           "skipped_in_window": win_skip, "skip_pct": round(skip_pct, 2),
                           "max_abs_weight": round(mw, 4),
                           "scaler_scale": scaler.get_scale(),
                           "lr": opt.param_groups[0]["lr"],
                           "elapsed_s": round(time.time() - t0, 1)}
                    emit(rec, {"progress/round": rnd, "progress/round_step": s + 1,
                               "progress/global_step": gstep,
                               "train/loss": rec["loss"],
                               "train/grad_norm": gn if math.isfinite(gn) else float("nan"),
                               "train/Wqkv_norm": wqkv, "train/abs_logit_max": lgmax,
                               "train/softmax_entropy": ent, "train/skip_pct": skip_pct,
                               "train/max_abs_weight": mw,
                               "train/scaler_scale": scaler.get_scale(),
                               "train/lr": opt.param_groups[0]["lr"],
                               "train/global_batch": B * world * cfg["grad_accum"]}, gstep)
                    # Two in-round tripwires, both with a grace period. The skip rate
                    # catches a run learning nothing; |logit|max catches the attention
                    # collapse that killed runs 1 and 3 — and that one stays FINITE, so
                    # the loss tripwire below would never see it coming.
                    if s + 1 >= cfg["abort_after"]:
                        if skip_pct > cfg["abort_skip_pct"]:
                            emit({"event": "abort", "round": rnd, "step": s + 1,
                                  "skip_pct": round(skip_pct, 2),
                                  "reason": "GradScaler discarded most of the window"})
                            want_abort = True
                        elif lgmax > cfg["abort_logit_max"]:
                            emit({"event": "abort", "round": rnd, "step": s + 1,
                                  "abs_logit_max": lgmax, "Wqkv_norm": wqkv,
                                  "reason": "attention logits past the collapse threshold"})
                            want_abort = True
                    win_ok = win_skip = 0

                # THE DECISION IS COLLECTIVE, AND EVERY RANK REACHES IT.
                # Only rank 0 computes these statistics, but if it broke out of the loop
                # alone, rank 1's next backward() would block forever inside DDP's
                # gradient all-reduce and the kernel would burn its budget in a hang.
                if (s + 1) % LOG_EVERY == 0:
                    flag = torch.tensor([1.0 if (is_main and want_abort) else 0.0],
                                        device=device)
                    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                    if flag.item() > 0:
                        aborted = True
                        break
            if aborted:
                break
        sched.step()
        # -------------------------------------------------------------

        dist.all_reduce(running, op=dist.ReduceOp.SUM)
        train_loss = (running[0] / running[1]).item()

        if aborted:
            if is_main:
                (C.run_dir(cfg["run_name"]) / "logs" / "diverged.json").write_text(
                    json.dumps({"round": int(rnd), "reason": "in-round tripwire",
                                "skipped_steps": int(skipped),
                                "train_loss": str(train_loss)}, indent=2))
                print(f"[abort] round {rnd:03d}: in-round tripwire fired. last.pt still "
                      f"holds round {rnd - 1}.", flush=True)
            outcome = "aborted"
            break

        # --- NaN tripwire -------------------------------------------------------
        # train_loss comes out of an all_reduce(SUM), so both ranks hold the SAME value
        # and can break together without another collective — no deadlock risk.
        #
        # Attempt 1 of this build went non-finite in round 25 and the loop kept going for
        # 20 more rounds: 4.1 h of GPU spent, 20 NaN checkpoints written, last.pt
        # overwritten with NaN weights so the run could not even be resumed. Nothing
        # checked. Breaking BEFORE eval and before save_round leaves the last valid
        # checkpoint intact at last.pt.
        if not math.isfinite(train_loss):
            if is_main:
                print(f"[diverged] round {rnd:03d}: train_loss = {train_loss}. Stopping "
                      f"now; last.pt still holds round {rnd - 1}, the last valid one.",
                      flush=True)
                (C.run_dir(cfg["run_name"]) / "logs" / "diverged.json").write_text(
                    json.dumps({"round": int(rnd), "train_loss": str(train_loss),
                                "last_valid_round": int(rnd) - 1,
                                "skipped_steps": int(skipped),
                                "reason": "non-finite training loss"}, indent=2))
            outcome = "diverged"
            break
        # ------------------------------------------------------------------------

        if is_main:
            y_pred = evaluate_full(model.module, Xte, prob_buf, cfg)
            from proj.metrics import compute_metrics
            m0 = compute_metrics(y_true, y_pred, cfg["num_classes"])
            f1m = m0["f1_macro"]
            is_best = f1m > best_f1; best_f1 = max(best_f1, f1m)
            save_prob = is_best or rnd == cfg["rounds"] - 1
            y_prob = prob_buf.cpu().numpy() if save_prob else None
            m = save_round_artifacts(
                rnd, y_true, y_pred, y_prob, cfg["num_classes"], cfg["class_names"],
                C.run_dir(cfg["run_name"]), metrics=m0, is_best=is_best,
                extra={"train_loss": train_loss, "lr": opt.param_groups[0]["lr"],
                       "seconds": round(time.time() - t0, 1),
                       "skipped_steps": int(skipped),
                       "samples_per_sec": round(steps * B * world * cfg["epochs_per_round"]
                                                / max(time.time() - t0, 1e-9)),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)},
                save_prob=save_prob)
            C.save_round(model, opt, scaler, sched, rnd, cfg, m, cfg["run_name"])
            print(f"[round {rnd:03d}] loss {train_loss:.4f}  acc {m['accuracy']:.4f}  "
                  f"F1_mac {m['f1_macro']:.4f}  F1_wtd {m['f1_weighted']:.4f}  "
                  f"{m['seconds']}s  {m['samples_per_sec']:,}/s  peak {m['peak_gb']}GB"
                  + ("  <- best" if is_best else ""), flush=True)
            if WB is not None:
                WB.log({"progress/completed_round": rnd,
                        "progress/global_step": (rnd + 1) * steps,
                        "round/duration_s": m["seconds"],
                        "round/skipped_steps": int(skipped),
                        "round/train_loss": train_loss,
                        **{f"eval/{k}": v for k, v in m0.items()}},
                       step=(rnd + 1) * steps)
            del y_pred, y_prob
            torch.cuda.reset_peak_memory_stats()

        dist.barrier()      # after the atomic write: a kill during eval leaves a valid last.pt

        # --- session budget: only begin a round we can finish -------------------
        # Both ranks must break together or the next collective deadlocks, so the
        # decision is made by all_reduce, never from each rank's own clock.
        probe = torch.tensor([time.time() - t0], device=device)
        dist.all_reduce(probe, op=dist.ReduceOp.MAX)
        round_s = probe.item()
        over = torch.tensor([1.0 if time.time() + 1.15 * round_s > deadline else 0.0],
                            device=device)
        dist.all_reduce(over, op=dist.ReduceOp.MAX)
        if over.item() > 0 and rnd + 1 < cfg["rounds"]:
            if is_main:
                left = (deadline - time.time()) / 3600
                print(f"[budget] stopping cleanly after round {rnd:03d}: next round needs "
                      f"~{round_s/60:.1f} min, {left*60:.1f} min left in the session budget. "
                      f"Re-push with this run's output attached to resume at round {rnd+1}.",
                      flush=True)
                (C.run_dir(cfg["run_name"]) / "logs" / "stopped_early.json").write_text(
                    json.dumps({"last_round": int(rnd), "next_round": int(rnd) + 1,
                                "reason": "session wall-clock budget",
                                "round_seconds": round(round_s, 1)}, indent=2))
            outcome = "budget"
            break

    if is_main and WB is not None:
        WB.summary["outcome"] = outcome
        WB.finish()
    dist.destroy_process_group()
