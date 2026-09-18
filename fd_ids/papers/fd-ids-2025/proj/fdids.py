"""FD-IDS client update and aggregation — Zhang et al., Sensors 2025, Eq. (2)-(6), Algorithm 1.

L_distill = lambda*L_hard + (1-lambda)*L_soft + beta*L_proximal          Eq. (6)
  L_hard      = CrossEntropy(student logits, integer labels)
  L_soft      = T^2 * KL( softmax(Z_t/T) || softmax(Z_s/T) )             Eq. (4)
  L_proximal  = (mu/2) * || w_k - w_G^t ||^2                             Eq. (3)
w_G^{t+1} = sum_k (n_k/n) * w_k^{t+1}                                    Eq. (2)
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
# One flat float vector + one int vector per client. A DAGSNet state_dict has 192 entries;
# torch.multiprocessing gives each tensor its own shared-memory fd, so 100 clients a round
# would exhaust the process fd limit. Parameters come FIRST so vec[:n_params] is exactly
# the block the proximal term applies to -- BN running stats are buffers, not parameters.
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


# --------------------------------------------------------------------- teacher (KD)
@torch.inference_mode()
def teacher_logits(teacher, X, lo, hi, cfg, batch=16384):
    """Z_t for one client's rows, computed once per round.

    The teacher is w_G^t, frozen for the whole round and in eval() mode, so its output is a
    function of the input row alone -- that is what makes computing it once here the same
    quantity the training step would have computed. It is NOT bit-identical: this batches at
    16384 and stores fp16, and a different batch size reduces in a different order. Measured
    on CPU fp32, batch 16 vs one big batch: max|dZt| = 1.2e-4, which moves the KD loss by
    <1e-3 and the student gradient by <1e-3 relative (tests/test_teacher.py). Treat those as
    the tolerance, not zero."""
    out = torch.empty((hi - lo, teacher.head[-1].out_features), dtype=torch.float16,
                      device=X.device)
    for i in range(lo, hi, batch):
        j = min(i + batch, hi)
        with amp(cfg):
            out[i - lo:j - lo] = teacher(X[i:j].float()).half()
    return out


# --------------------------------------------------------------------- client update
def client_update(compiled, eager, opt, scaler, X, Y, lo, hi, w_global, n_params,
                  ZT, cfg, gen):
    """Algorithm 1 lines 13-21, E local epochs. Returns device-side accumulators; the
    caller reads them once, after the client finishes.

    Dropout and any other global-RNG consumer read torch's default generator, which the
    worker re-seeds from (seed, round, client) before calling this. `gen` is separate and
    drives only the shuffle, so neither depends on which GPU took the client or on how many
    clients ran before it."""
    lam, beta, mu, T = cfg["lam"], cfg["beta"], cfg["mu"], cfg["temperature"]
    B, clip = cfg["batch"], cfg["clip"]
    dev = X.device
    params = [p for p in eager.parameters()]
    anchor = w_global[:n_params]
    # Views into the anchor and the parameter storages, built ONCE per client. Adam updates
    # p.data in place, so these stay valid for every step; .grad does not, because
    # zero_grad(set_to_none=True) replaces it each step.
    pdata, aview, o0 = [], [], 0
    for p in params:
        pdata.append(p.data)
        aview.append(anchor[o0:o0 + p.numel()].view_as(p)); o0 += p.numel()

    ce_acc = torch.zeros((), device=dev); kd_acc = torch.zeros((), device=dev)
    gn_acc = torch.zeros((), device=dev); skips = torch.zeros((), device=dev)
    nonfin = torch.zeros((), device=dev)
    zero = torch.zeros((), device=dev)
    E = cfg["local_epochs"]                       # Algorithm 1 line 15
    nsteps = E * math.ceil((hi - lo) / B)

    for _e in range(E):
        perm = lo + torch.randperm(hi - lo, generator=gen, device=dev)
        for i in range(0, hi - lo, B):
            idx = perm[i:i + B]
            xb = X[idx].float()
            yb = Y[idx].long()
            zt = ZT[idx - lo]
            # The tail batch is a different shape and would recompile the CUDA graph once per
            # client. Run that one step on the eager module: same weights, same math.
            mod = compiled if idx.numel() == B else eager
            with amp(cfg):
                zs = mod(xb)
            zs = zs.float()
            l_hard = F.cross_entropy(zs, yb)                                      # Eq. (6) hard
            l_soft = (T * T) * F.kl_div(F.log_softmax(zs / T, dim=1),             # Eq. (4)
                                        F.log_softmax(zt.float() / T, dim=1),
                                        reduction="batchmean", log_target=True)
            loss = lam * l_hard + (1.0 - lam) * l_soft
            scaler.scale(loss).backward()
            scaler.unscale_(opt)                                   # grads now in true units
            # Eq. (3) as its gradient: d/dw [beta*(mu/2)*||w-w_G||^2] = beta*mu*(w-w_G).
            # Building it in the autograd graph would cost ~400 kernels over 118 tensors.
            # The closed form is exact, and the foreach pair is two fused launches instead of
            # a torch.cat over 99 parameters plus 99 slice-add_ calls: measured 16.88 ->
            # 15.16 ms/step with max|delta| exactly 0 against the loop it replaces. That
            # matters more once the model is compiled -- this runs OUTSIDE the compiled
            # graph, so it does not shrink with it. PARAMETERS only.
            torch._foreach_add_(
                [p.grad for p in params],
                torch._foreach_sub(pdata, aview), alpha=beta * mu)
            gn = torch.nn.utils.clip_grad_norm_(params, clip)
            prev = scaler._scale.clone() if scaler.is_enabled() else None
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            # A skipped step overflowed: its grad-norm is inf and its loss may be nan.
            # torch.where, not multiplication -- inf*0 is nan.
            applied = (scaler._scale >= prev) if prev is not None \
                else torch.ones((), dtype=torch.bool, device=dev)
            # Non-finite ONLY counts on a step the scaler APPLIED. On a skipped step an
            # infinite grad-norm is the ordinary fp16 overflow the scaler exists to absorb:
            # it discarded the step and halved the scale, and with a fresh scaler per client
            # that happens on the first steps of nearly every client. On an applied step the
            # value did reach the weights -- and it can, because the proximal term is added
            # after unscale_, so scaler.step's overflow check never saw it.
            nonfin += (~torch.isfinite(gn) & applied).float()
            ce_acc += torch.where(applied, l_hard.detach(), zero)
            kd_acc += torch.where(applied, l_soft.detach(), zero)
            gn_acc += torch.where(applied, gn, zero)
            skips += (~applied).float()
    return ce_acc, kd_acc, gn_acc, skips, nonfin, nsteps


# --------------------------------------------------------------------- aggregation
def aggregate(updates, n_total):
    """Eq. (2): w_G = sum_k (n_k/n) w_k. `updates` must already be sorted by client id --
    float addition order decides the result, and it must not be set by a completion race."""
    acc = None
    for cid, n_k, fv, iv in updates:
        w = n_k / n_total
        acc = fv * w if acc is None else acc.add_(fv, alpha=w)
    # num_batches_tracked is an int counter, not an averageable quantity; with the default
    # BatchNorm momentum=0.1 it is unused at inference. Take the max so it stays monotone.
    ints = torch.stack([iv for _, _, _, iv in updates]).amax(dim=0) if updates[0][3].numel() \
        else updates[0][3]
    return acc, ints
