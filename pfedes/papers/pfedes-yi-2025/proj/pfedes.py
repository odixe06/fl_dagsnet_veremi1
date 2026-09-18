"""pFedES client update and aggregation — Yi et al., Eq. (4)-(11).

Round t, client k in S^t (K = C*N clients drawn without replacement, seeded by round):
  Step 1  freeze G(theta^{t-1}), train F_k:
            x_hat = G(x)                                              Eq. (4)
            l_w   = mu * CE(F_k(x_hat), y) + (1 - mu) * CE(F_k(x), y)  Eq. (5)-(6)
            w_k^t <- w_k^{t-1} - eta_w * grad l_w                      Eq. (7)
  Step 2  freeze F_k(w_k^t), train G:
            l_theta = CE(F_k(G(x)), y)                                 Eq. (8)-(9)
            theta_k^t <- theta^{t-1} - eta_theta * grad l_theta        Eq. (10)
Server:   theta^t = sum_{k in S^t} (n_k / sum_{j in S^t} n_j) * theta_k^t   Eq. (11)

Implementation choices that are NOT in the paper and must be reported:
  * The two forward passes of Step 1 run as ONE forward on the concatenated batch
    [x_hat; x] (2B rows). The loss is exactly Eq. (6) -- each half gets its own CE mean --
    but F_k's BatchNorm sees the union of enhanced and original rows in one batch. Two
    separate calls of a CUDA-graph-compiled module before backward are not allowed (the
    second replay overwrites the first's outputs), and the launch-bound DAGSNet makes the
    concatenated forward ~1.7x cheaper than two calls.
  * "Frozen" is implemented as eval() + no parameter gradients: the frozen module uses its
    running BatchNorm statistics and no dropout, so x_hat in Step 1 and F_k in Step 2 are
    deterministic functions of the input.
  * Eq. (11) normalises over the SELECTED clients. The paper writes n_k / n with n the total
    over all N clients (Preliminaries); with C < 100 % that literal form shrinks theta by
    the factor C every round.
  * Optimizer is AdamW (owner's decision, not the paper's SGD), re-created per client per
    round: theta arrives fresh from the server each round and the owner chose to reset the
    F_k moments as well, so no optimizer state is ever persisted.
  * The learning rate follows a per-ROUND schedule (`lr_at`), constant within a round and
    shared by eta_w and eta_theta. The paper uses a constant 0.01; the owner's unified
    choice across the sibling rebuilds (2026-09-13) is a cosine from cfg['lr'] at round 1
    to cfg['lr_min'] at the last round, because every personalized F_k on this data peaks
    on the global test after ~2 local epochs and then drifts under a constant rate.
"""
import contextlib
import math
import numpy as np
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


# --------------------------------------------------------------------- client sampling
def n_selected(n_clients, frac):
    """K = |C * N|, at least 1. frac = 1.0 selects everyone."""
    k = int(round(frac * n_clients))
    if not 1 <= k <= n_clients:
        raise ValueError(f"participation {frac} of {n_clients} clients gives K={k}")
    return k


def select_clients(n_clients, frac, seed, rnd):
    """The paper's 'randomly selects K clients among N'. A pure function of (seed, round):
    a resumed session draws the same set the original would have, and the draw does not
    consume the training RNG."""
    k = n_selected(n_clients, frac)
    if k == n_clients:
        return list(range(n_clients))
    g = np.random.default_rng(seed * 1_000_003 + rnd * 10_007)
    return sorted(int(c) for c in g.choice(n_clients, size=k, replace=False))


# --------------------------------------------------------------------- learning rate
LR_SCHEDULES = ("constant", "cosine")


def lr_at(cfg, rnd):
    """Learning rate of round `rnd` (1-based), held constant within the round and used for
    both eta_w and eta_theta. A pure function of (cfg, rnd): a resumed session applies
    exactly the value the original session would have.

      constant : cfg['lr'] every round
      cosine   : lr_min + (lr - lr_min)/2 * (1 + cos(pi * (rnd-1) / (rounds-1)))
                 -- cfg['lr'] at round 1, cfg['lr_min'] at round cfg['rounds']; the
                 formula the afpha rebuild uses, so the sibling projects share one schedule.
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


# --------------------------------------------------------------------- client update
def _freeze(module, frozen):
    """frozen: eval() so BatchNorm uses running stats and Dropout is off, and no gradient
    is accumulated into its parameters (the backward still flows THROUGH it to its input,
    which is what Step 2 needs). Trainable: the reverse."""
    module.train(not frozen)
    for p in module.parameters():
        p.requires_grad_(not frozen)


def _epoch(Fc, Fe, Gc, Ge, opt, scaler, params, X, Y, lo, hi, cfg, gen, step_fn, accs):
    """One pass over the client's rows. `step_fn(Fm, Gm, xb, yb)` returns (loss, extra)
    for the phase being trained; `accs` are device-side accumulators updated per step.
    The tail batch is a different shape and would recompile the CUDA graph once per
    client, so it runs on the eager modules: same weights, same math."""
    B, clip, dev = cfg["batch"], cfg["clip"], X.device
    zero = torch.zeros((), device=dev)
    perm = lo + torch.randperm(hi - lo, generator=gen, device=dev)
    n = 0
    for i in range(0, hi - lo, B):
        # One training step invokes up to four captured graphs (G forward, F forward and
        # backward, and in Step 2 G's backward). CUDA-graph trees decide which pool memory
        # is dead from the iteration boundary, so declare it explicitly instead of letting
        # the no_grad G forward be mistaken for a new iteration mid-step.
        if X.is_cuda:
            torch.compiler.cudagraph_mark_step_begin()
        idx = perm[i:i + B]
        full = idx.numel() == B
        xb = X[idx].float()
        yb = Y[idx].long()
        loss, extra = step_fn(Fc if full else Fe, Gc if full else Ge, xb, yb)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)                                   # grads now in true units
        gn = torch.nn.utils.clip_grad_norm_(params, clip)
        prev = scaler._scale.clone() if scaler.is_enabled() else None
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        # A skipped step overflowed: its grad-norm is inf and its loss may be nan.
        # torch.where, not multiplication -- inf*0 is nan.
        applied = (scaler._scale >= prev) if prev is not None \
            else torch.ones((), dtype=torch.bool, device=dev)
        accs["nonfin"] += (~torch.isfinite(gn) & applied).float()
        accs["loss"] += torch.where(applied, loss.detach(), zero)
        accs["extra"] += torch.where(applied, extra.detach(), zero)
        accs["gn"] += torch.where(applied, gn, zero)
        accs["skips"] += (~applied).float()
        n += 1
    return n


def client_update(Fc, Fe, Gc, Ge, optF, optG, scF, scG, X, Y, lo, hi, cfg, gen):
    """Step 1 then Step 2 for one client. Fe/Ge are the eager modules that own the
    parameters; Fc/Gc are their compiled aliases (or the same objects when not compiled).

    Returns two dicts of device-side accumulators (read once by the caller) and the step
    counts. Dropout reads torch's default generator, which the worker re-seeds from
    (seed, round, client) before calling this; `gen` drives only the shuffles."""
    mu, dev = cfg["mu"], X.device
    B = cfg["batch"]
    paramsF = list(Fe.parameters())
    paramsG = list(Ge.parameters())

    def new_accs():
        return {k: torch.zeros((), device=dev) for k in ("loss", "extra", "gn", "skips",
                                                         "nonfin")}

    # ---- Step 1: freeze G, train F_k on [x_hat; x]                       Eq. (4)-(7)
    _freeze(Ge, True); _freeze(Fe, False)

    def step1(Fm, Gm, xb, yb):
        with torch.no_grad(), amp(cfg):
            xhat = Gm(xb)
        with amp(cfg):
            z = Fm(torch.cat([xhat.float(), xb], dim=0))
        z = z.float()
        n = xb.shape[0]
        l1 = F.cross_entropy(z[:n], yb)                      # enhanced data, Eq. (5)
        l2 = F.cross_entropy(z[n:], yb)                      # original data, Eq. (5)
        return mu * l1 + (1.0 - mu) * l2, l2                 # Eq. (6); l2 reported separately

    acc1 = new_accs()
    n1 = sum(_epoch(Fc, Fe, Gc, Ge, optF, scF, paramsF, X, Y, lo, hi, cfg, gen, step1, acc1)
             for _ in range(cfg["local_epochs"]))

    # ---- Step 2: freeze F_k(w_k^t), train G                             Eq. (8)-(10)
    _freeze(Fe, True); _freeze(Ge, False)

    def step2(Fm, Gm, xb, yb):
        with amp(cfg):
            z = Fm(Gm(xb).float())
        loss = F.cross_entropy(z.float(), yb)                # Eq. (9)
        return loss, loss

    acc2 = new_accs()
    n2 = sum(_epoch(Fc, Fe, Gc, Ge, optG, scG, paramsG, X, Y, lo, hi, cfg, gen, step2, acc2)
             for _ in range(cfg["proxy_epochs"]))

    _freeze(Fe, False); _freeze(Ge, False)                  # leave both trainable
    return acc1, n1, acc2, n2


def expected_steps(n_k, cfg):
    per_epoch = math.ceil(n_k / cfg["batch"])
    return cfg["local_epochs"] * per_epoch, cfg["proxy_epochs"] * per_epoch


# --------------------------------------------------------------------- aggregation
def aggregate(updates):
    """Eq. (11) over the selected set: theta = sum_k (n_k / n_S) theta_k. `updates` must
    already be sorted by client id -- float addition order decides the result, and it must
    not be set by a completion race."""
    n_s = sum(n_k for _, n_k, _, _ in updates)
    acc = None
    for cid, n_k, fv, iv in updates:
        w = n_k / n_s
        acc = fv * w if acc is None else acc.add_(fv, alpha=w)
    # num_batches_tracked is an int counter, not an averageable quantity; with the default
    # BatchNorm momentum=0.1 it is unused at inference. Take the max so it stays monotone.
    ints = torch.stack([iv for _, _, _, iv in updates]).amax(dim=0) if updates[0][3].numel() \
        else updates[0][3]
    return acc, ints
