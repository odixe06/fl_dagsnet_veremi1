"""Lightweight federated learning for NILM (Li et al.) — the federated mutual-learning
part only: Eq. (17)-(19) and the round loop of Algorithm 1. The NAS search (§III.B) is
deliberately absent; every model is the same DAGSNet (proj/model.py).

Round t, EVERY client k (Algorithm 1 iterates over all households):
  personalized model w_s^k (stays on the device) and proxy w_r^k <- w_bar_r^{t-1}
  one local epoch, per batch (x, y):
      z_s = f_s(w_s, x),  z_r = f_r(w_r, x)
      l_s = CE(z_s, y),   l_r = CE(z_r, y)                              Eq. (18)
      l_d = l(y_s, y_r) / (l_s + l_r)                                   Eq. (19)
      L_s = l_s + l_d,    L_r = l_r + l_d                               Eq. (17)
      w_s <- w_s - eta grad_{w_s} L_s ;  w_r <- w_r - eta grad_{w_r} L_r
  upload w_r^k
Server:  w_bar_r^t = (1/K) sum_k w_r^k   (unweighted mean, as Algorithm 1 writes it)

Choices the paper leaves open for a CLASSIFIER, fixed by the owner on 2026-09-15 and
reported with every number:
  * l(y_s, y_r) is the Deep-Mutual-Learning KL on softmax outputs, one direction per
    model: w_s minimises KL(p_r || p_s) with p_r detached, w_r minimises KL(p_s || p_r)
    with p_s detached (the paper's L2 is for a regression output). Temperature 1.
  * the adaptive weight 1 / (l_s + l_r) is a stop-gradient scalar: the paper calls it
    "the weight of the distillation loss", so it is not differentiated through. A
    clamp at 1e-6 guards the division when both CE terms are exactly 0.
  * both gradients come from ONE backward of L_s + L_r: with the cross terms detached,
    grad_{w_s} L_r = grad_{w_r} L_s = 0, so this is exactly the two updates of
    Algorithm 1. One AdamW holds both parameter sets (Adam is per-parameter, so this
    equals two optimizers at the same rate); the gradient norm is clipped per model.
  * AdamW (owner, as in the sibling rebuilds; the paper names no optimizer), re-created
    per client per round, so no optimizer state exists at a round boundary and a
    checkpoint is weights alone.
  * the learning rate follows a per-ROUND cosine schedule (`lr_at`), constant within a
    round and shared by both models: the owner's single LR policy across the sibling
    rebuilds, because every per-client DAGSNet on this data peaks on the global test
    after ~2 local epochs and drifts under a constant rate.
  * every client trains in every round (C = 100 %), as Algorithm 1 loops over all K.
  * the final local fine-tuning step of the paper's framework (§III.A step 4) is
    skipped: w_s is trained on local labels in every round already.
"""
import contextlib
import math
import torch
import torch.nn.functional as F


def amp(cfg):
    """fp16 autocast on CUDA; a no-op on CPU so the same code runs in the local
    simulation. Never bf16: the T4 is sm_75 and falls back to a slow emulation path."""
    if cfg.get("device", "cuda") == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


# --------------------------------------------------------------------- flat layout
# One flat float vector + one int vector per model. A DAGSNet state_dict has 192 entries;
# torch.multiprocessing gives each tensor its own shared-memory fd, so 100 clients a round
# would exhaust the process fd limit. Parameters come FIRST so vec[:n_params] is exactly
# the learnable block; BN running stats follow as buffers.
def layout(model):
    pnames = {n for n, _ in model.named_parameters()}
    sd = model.state_dict()
    fkeys = [k for k in sd if k in pnames]
    fkeys += [k for k in sd if k not in pnames and sd[k].is_floating_point()]
    ikeys = [k for k in sd if not sd[k].is_floating_point()]
    n_params = sum(sd[k].numel() for k in fkeys if k in pnames)
    return fkeys, ikeys, n_params


def flatten(model, fkeys, ikeys):
    sd = model.state_dict()
    fv = torch.cat([sd[k].reshape(-1).float() for k in fkeys])
    iv = torch.stack([sd[k].reshape(-1).long().squeeze() for k in ikeys]) if ikeys \
        else torch.zeros(0, dtype=torch.long)
    return fv, iv


def unflatten_into(model, fv, iv, fkeys, ikeys):
    sd = model.state_dict()
    o = 0
    for k in fkeys:
        t = sd[k]; n = t.numel()
        t.copy_(fv[o:o + n].view_as(t)); o += n           # copy_ keeps addresses -> CUDA graph valid
    for j, k in enumerate(ikeys):
        sd[k].copy_(iv[j].view_as(sd[k]))
    return model


# --------------------------------------------------------------------- learning rate
LR_SCHEDULES = ("constant", "cosine")


def lr_at(cfg, rnd):
    """Learning rate of round `rnd` (1-based), held constant within the round and used for
    both w_s and w_r. A pure function of (cfg, rnd): a resumed session applies exactly
    the value the original session would have.

      constant : cfg['lr'] every round
      cosine   : lr_min + (lr - lr_min)/2 * (1 + cos(pi * (rnd-1) / (rounds-1)))
                 -- cfg['lr'] at round 1, cfg['lr_min'] at round cfg['rounds']; the
                 formula the sibling rebuilds share.
    """
    sched = cfg.get("lr_schedule", "constant")
    if sched == "constant":
        return float(cfg["lr"])
    if sched == "cosine":
        T = int(cfg["rounds"])
        if not 1 <= rnd <= T:
            raise ValueError(f"round {rnd} outside 1..{T}: the cosine schedule is undefined")
        if T == 1:
            return float(cfg["lr"])
        lo, hi = float(cfg["lr_min"]), float(cfg["lr"])
        return lo + 0.5 * (hi - lo) * (1.0 + math.cos(math.pi * (rnd - 1) / (T - 1)))
    raise ValueError(f"lr_schedule {sched!r} not in {LR_SCHEDULES}")


# --------------------------------------------------------------------- the loss
# Order of the per-step values accumulated on the device (see `client_update`).
ACC_KEYS = ("loss_s", "loss_r", "ce_s", "ce_r", "kl_s", "kl_r", "gnorm_s", "gnorm_r")
DEN_EPS = 1e-6


def mutual_loss(z_s, z_r, y):
    """Eq. (17)-(19) for one batch, from fp32 logits. Returns the joint objective whose
    single backward yields grad_{w_s} L_s and grad_{w_r} L_r, plus the parts for logging.

        ce_s = CE(z_s, y), ce_r = CE(z_r, y)                                Eq. (18)
        w    = 1 / sg(ce_s + ce_r)                                          Eq. (19) weight
        kl_s = KL(sg(p_r) || p_s)   -- what the personalized model absorbs from the proxy
        kl_r = KL(sg(p_s) || p_r)   -- what the proxy absorbs from the personalized model
        L_s  = ce_s + w * kl_s,  L_r = ce_r + w * kl_r                      Eq. (17)
    """
    ce_s = F.cross_entropy(z_s, y)
    ce_r = F.cross_entropy(z_r, y)
    log_ps, log_pr = F.log_softmax(z_s, dim=1), F.log_softmax(z_r, dim=1)
    # kl_div(input=log q, target=p) = sum p (log p - log q); batchmean = mean over rows
    kl_s = F.kl_div(log_ps, log_pr.detach(), log_target=True, reduction="batchmean")
    kl_r = F.kl_div(log_pr, log_ps.detach(), log_target=True, reduction="batchmean")
    w = 1.0 / (ce_s + ce_r).detach().clamp_min(DEN_EPS)
    loss_s = ce_s + w * kl_s
    loss_r = ce_r + w * kl_r
    return loss_s + loss_r, (loss_s, loss_r, ce_s, ce_r, kl_s, kl_r)


# --------------------------------------------------------------------- client update
def make_optimizer(Se, Re, lr, cfg, fused):
    """One AdamW over both parameter sets. Adam's state is per parameter, so this is the
    same update as two optimizers at the same rate, and the GradScaler skip decision is
    atomic for the step: either both models apply it or neither does."""
    return torch.optim.AdamW([{"params": list(Se.parameters())},
                              {"params": list(Re.parameters())}],
                             lr=lr, weight_decay=cfg["weight_decay"], fused=fused)


def client_update(Sc, Se, Rc, Re, opt, scaler, X, Y, lo, hi, cfg, gen):
    """`local_epochs` passes over rows [lo, hi) of the resident tensors for one client.
    Se/Re are the eager modules that own the parameters (w_s, w_r); Sc/Rc are their
    compiled aliases (or the same objects when not compiled). The tail batch is a
    different shape and would recompile the CUDA graph once per client, so it runs on
    the eager modules: same weights, same math.

    Returns (acc, n_steps): `acc` is a device tensor of len(ACC_KEYS) + 2 sums over
    APPLIED steps (the last two entries are the skipped-step count and the count of
    applied steps whose gradient norm was not finite), read once by the caller.
    Dropout reads torch's default generator, which the worker re-seeds from
    (seed, round, client) before calling this; `gen` drives only the shuffles."""
    B, clip, dev = cfg["batch"], cfg["clip"], X.device
    paramsS, paramsR = list(Se.parameters()), list(Re.parameters())
    Se.train(); Re.train()
    acc = torch.zeros(len(ACC_KEYS) + 2, device=dev)
    zeros = torch.zeros(len(ACC_KEYS), device=dev)
    n = 0
    for _ in range(cfg["local_epochs"]):
        perm = lo + torch.randperm(hi - lo, generator=gen, device=dev)
        for i in range(0, hi - lo, B):
            # Two forward graphs and their backward run per step. CUDA-graph trees decide
            # which pool memory is dead from the iteration boundary, so declare it
            # explicitly instead of letting the second forward look like a new iteration.
            if X.is_cuda:
                torch.compiler.cudagraph_mark_step_begin()
            idx = perm[i:i + B]
            full = idx.numel() == B
            xb = X[idx].float()
            yb = Y[idx].long()
            with amp(cfg):
                z_s = (Sc if full else Se)(xb)
                z_r = (Rc if full else Re)(xb)
            loss, parts = mutual_loss(z_s.float(), z_r.float(), yb)   # loss in fp32
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                                   # grads now in true units
            gn_s = torch.nn.utils.clip_grad_norm_(paramsS, clip)
            gn_r = torch.nn.utils.clip_grad_norm_(paramsR, clip)
            prev = scaler._scale.clone() if scaler.is_enabled() else None
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            # A skipped step overflowed: its grad-norm is inf and its loss may be nan.
            # torch.where, not multiplication -- inf*0 is nan.
            applied = (scaler._scale >= prev) if prev is not None \
                else torch.ones((), dtype=torch.bool, device=dev)
            vals = torch.stack([p.detach() for p in parts] + [gn_s, gn_r])
            acc[:len(ACC_KEYS)] += torch.where(applied, vals, zeros)
            acc[len(ACC_KEYS)] += (~applied).float()
            acc[len(ACC_KEYS) + 1] += ((~torch.isfinite(gn_s) | ~torch.isfinite(gn_r))
                                       & applied).float()
            n += 1
    return acc, n


def expected_steps(n_k, cfg):
    return cfg["local_epochs"] * math.ceil(n_k / cfg["batch"])


# --------------------------------------------------------------------- aggregation
def aggregate(updates):
    """Algorithm 1: w_bar_r = (1/K) sum_k w_r^k, every client weighted equally (owner's
    choice to follow the paper rather than FedAvg's n_k weighting). `updates` must
    already be sorted by client id -- float addition order decides the result, and it
    must not be set by a completion race."""
    K = len(updates)
    acc = None
    for cid, fv, iv in updates:
        acc = fv / K if acc is None else acc.add_(fv, alpha=1.0 / K)
    # num_batches_tracked is an int counter, not an averageable quantity; with the default
    # BatchNorm momentum=0.1 it is unused at inference. Take the max so it stays monotone.
    ints = torch.stack([iv for _, _, iv in updates]).amax(dim=0) if updates[0][2].numel() \
        else updates[0][2]
    return acc, ints
