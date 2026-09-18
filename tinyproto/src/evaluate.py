"""Per-client evaluation on the fixed global test set.

In TinyProto there is no global model: every client keeps its own weights, so a round's
evaluation is C independent full-test passes. At 100 clients that is the dominant cost of the
whole run, which is why this module has two paths and a calibration step that picks between
them from a measurement rather than a guess.

* `eval_sequential` — one client at a time. Always correct, always available.
* `eval_vmapped`    — G clients at once over `torch.vmap` on stacked parameters. DAGSNet is a
  pile of tiny convolutions on an 11-position axis, so a single model leaves the GPU almost
  idle; stacking turns those into grouped convolutions with G× the channels and amortizes the
  test-data reads across G clients.

Both paths run on a **BatchNorm-folded copy** of each client's model. In eval mode BatchNorm is
an affine map with constant coefficients, so folding it into the preceding convolution is exact
(not an approximation), and it buys three things: `vmap` works under fp16 autocast (batched
`F.batch_norm` rejects the mixed dtypes autocast produces), ~31 fewer kernel launches per
forward, and one uniform parameter set to stack.

Folding is redone every round after the weights change; it costs 31 small tensor ops per client.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.func import functional_call, stack_module_state


# ---------------------------------------------------------------------------
# BatchNorm folding
# ---------------------------------------------------------------------------

@torch.no_grad()
def fold_bn(model: nn.Module) -> nn.Module:
    """Return an eval-mode copy with every BatchNorm folded into its preceding Conv1d.

        y = gamma·(conv(x) − mean)/sqrt(var + eps) + beta
          = conv'(x) + b'   with   w' = w·gamma/sqrt(var+eps),  b' = beta − gamma·mean/sqrt(var+eps)

    Exact for `model.eval()`; meaningless for `model.train()`, which uses batch statistics.
    Every BatchNorm in DAGSNet sits inside a `cbr` block, i.e. `Sequential(Conv1d, BN, ReLU)`,
    so the pattern match below covers all 31 of them. The assertion at the end is what stops a
    future architecture change from silently leaving a BatchNorm unfolded.
    """
    m = copy.deepcopy(model).eval()
    for seq in m.modules():
        if not (isinstance(seq, nn.Sequential) and len(seq) >= 2
                and isinstance(seq[0], nn.Conv1d) and isinstance(seq[1], nn.BatchNorm1d)):
            continue
        conv, bn = seq[0], seq[1]
        inv = torch.rsqrt(bn.running_var + bn.eps)
        w = conv.weight * (bn.weight * inv).view(-1, 1, 1)
        b = bn.bias - bn.weight * bn.running_mean * inv
        if conv.bias is not None:
            b = b + conv.bias * bn.weight * inv
        new = nn.Conv1d(conv.in_channels, conv.out_channels, conv.kernel_size[0],
                        stride=conv.stride[0], padding=conv.padding[0], bias=True,
                        device=w.device, dtype=w.dtype)
        new.weight.copy_(w)
        new.bias.copy_(b)
        seq[0] = new
        seq[1] = nn.Identity()
    left = [n for n, mod in m.named_modules() if isinstance(mod, nn.BatchNorm1d)]
    assert not left, f"BatchNorm survived folding at {left}; the cbr pattern changed"
    return m.eval()


class _Both(nn.Module):
    """`forward_both` as a plain `forward`, because `functional_call` only drives `forward`."""

    def __init__(self, m: nn.Module):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m.forward_both(x)


def make_meta_base(build_fn) -> nn.Module:
    """The stateless module `functional_call` drives, on the meta device.

    It must have the FOLDED structure: folding adds a bias to every conv and deletes the
    BatchNorm parameters, so a base built from the unfolded model leaves those keys unmatched
    and `functional_call` silently keeps meta tensors -- surfacing much later as
    "Tensor on device meta is not on the expected device cuda:0".
    """
    return _Both(fold_bn(build_fn())).to("meta").eval()


# ---------------------------------------------------------------------------
# prediction rules
# ---------------------------------------------------------------------------

def _proto_scores(feats: torch.Tensor, c_local: torch.Tensor,
                  bias: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """argmax of this equals argmin_j ‖h − c_L[j]‖₂ (Eq. 12).

    `bias` is −½‖c_L[j]‖², precomputed once per round rather than per batch. Absent classes are
    pushed to −inf: a client with no prototype for a class can never predict it.
    """
    s = feats @ c_local.transpose(-1, -2) + bias
    return s.masked_fill(~present, float("-inf"))


def _accumulate(cm: torch.Tensor, y: torch.Tensor, pred: torch.Tensor, C: int) -> None:
    """Integer confusion accumulation. bincount keeps shard/batch splits exactly
    order-independent, unlike any float reduction."""
    cm += torch.bincount(y * C + pred, minlength=C * C)


# ---------------------------------------------------------------------------
# sequential path
# ---------------------------------------------------------------------------

@torch.inference_mode()
def eval_sequential(models: list[nn.Module], protos: torch.Tensor, present: torch.Tensor,
                    X: torch.Tensor, Y: torch.Tensor, *, batch: int, C: int,
                    amp: bool = True, want_preds: bool = False):
    """models[k] is client k's BN-folded eval module. protos: (n, C, d). present: (n, C) bool.

    Returns (cm_proto (n, C, C) int64, cm_clf (n, C, C) int64, preds or None).
    """
    n = len(models)
    dev = X.device
    cm_p = torch.zeros(n, C * C, dtype=torch.int64, device=dev)
    cm_c = torch.zeros(n, C * C, dtype=torch.int64, device=dev)
    bias = (-0.5 * protos.pow(2).sum(-1)).unsqueeze(1)              # (n, 1, C)
    pres = present.unsqueeze(1)                                     # (n, 1, C)
    out_p = np.empty((n, len(Y)), dtype=np.uint8) if want_preds else None
    out_c = np.empty((n, len(Y)), dtype=np.uint8) if want_preds else None

    for k, m in enumerate(models):
        for i in range(0, X.shape[0], batch):
            xb = X[i:i + batch].float()
            yb = Y[i:i + batch].long()
            with autocast("cuda", dtype=torch.float16, enabled=amp):
                h, logits = m.forward_both(xb)
            if not torch.isfinite(h).all() or not torch.isfinite(logits).all():
                raise RuntimeError("non-finite evaluation output")
            h = h.float()
            pp = _proto_scores(h, protos[k], bias[k], pres[k]).argmax(1)
            pc = logits.float().argmax(1)
            _accumulate(cm_p[k], yb, pp, C)
            _accumulate(cm_c[k], yb, pc, C)
            if want_preds:
                out_p[k, i:i + xb.shape[0]] = pp.to(torch.uint8).cpu().numpy()
                out_c[k, i:i + xb.shape[0]] = pc.to(torch.uint8).cpu().numpy()
    preds = {"proto": out_p, "clf": out_c} if want_preds else None
    return cm_p.view(n, C, C), cm_c.view(n, C, C), preds


# ---------------------------------------------------------------------------
# vmapped path
# ---------------------------------------------------------------------------

class VmapEvaluator:
    """Evaluate G clients per pass with one stacked forward.

    Build once per round from the folded modules; `stack_module_state` copies, so the stacked
    tensors are independent of the source modules afterwards.
    """

    def __init__(self, folded: list[nn.Module], group: int, meta_base: nn.Module,
                 half: bool = True):
        # autocast does NOT reach the batched weight/bias that vmap hands to F.conv1d -- it
        # raises "Input type (c10::Half) and bias type (float) should be the same". So the
        # stacked parameters are cast once, here, and the forward runs with autocast disabled.
        # That makes this path fp16 in exactly the places the sequential autocast path is.
        self.group = group
        self.base = meta_base
        self.half = half
        self.chunks = []
        for lo in range(0, len(folded), group):
            wraps = [_Both(m).eval() for m in folded[lo:lo + group]]
            p, b = stack_module_state(wraps)
            if half:
                p = {k: v.half() for k, v in p.items()}
                b = {k: (v.half() if v.dtype.is_floating_point else v) for k, v in b.items()}
            self.chunks.append((lo, lo + len(wraps), p, b))

    @torch.inference_mode()
    def run(self, protos: torch.Tensor, present: torch.Tensor, X: torch.Tensor, Y: torch.Tensor,
            *, batch: int, C: int, amp: bool = True, want_preds: bool = False):
        n = protos.shape[0]
        dev = X.device
        cm_p = torch.zeros(n, C * C, dtype=torch.int64, device=dev)
        cm_c = torch.zeros(n, C * C, dtype=torch.int64, device=dev)
        bias = (-0.5 * protos.pow(2).sum(-1)).unsqueeze(1)
        pres = present.unsqueeze(1)
        out_p = np.empty((n, len(Y)), dtype=np.uint8) if want_preds else None
        out_c = np.empty((n, len(Y)), dtype=np.uint8) if want_preds else None
        base = self.base

        for lo, hi, params, buffers in self.chunks:
            g = hi - lo
            for i in range(0, X.shape[0], batch):
                xb = X[i:i + batch]
                xb = xb.half() if self.half else xb.float()
                yb = Y[i:i + batch].long()
                with autocast("cuda", enabled=False):
                    h, logits = torch.vmap(
                        lambda p, b: functional_call(base, (p, b), (xb,)),
                        in_dims=(0, 0))(params, buffers)
                h = h.float()                                        # (g, B, d)
                if not torch.isfinite(h).all() or not torch.isfinite(logits).all():
                    raise RuntimeError("non-finite vmapped evaluation output")
                pp = _proto_scores(h, protos[lo:hi], bias[lo:hi], pres[lo:hi]).argmax(-1)
                pc = logits.float().argmax(-1)                       # (g, B)
                for k in range(g):
                    _accumulate(cm_p[lo + k], yb, pp[k], C)
                    _accumulate(cm_c[lo + k], yb, pc[k], C)
                if want_preds:
                    out_p[lo:hi, i:i + xb.shape[0]] = pp.to(torch.uint8).cpu().numpy()
                    out_c[lo:hi, i:i + xb.shape[0]] = pc.to(torch.uint8).cpu().numpy()
        preds = {"proto": out_p, "clf": out_c} if want_preds else None
        return cm_p.view(n, C, C), cm_c.view(n, C, C), preds


# ---------------------------------------------------------------------------
# calibration: choose the path from a measurement, and prove it agrees
# ---------------------------------------------------------------------------

def verify_fold(model: nn.Module, folded: nn.Module, x: torch.Tensor, amp: bool = False) -> dict:
    """Folding must be exact. Reported, not assumed.

    `amp=True` repeats the comparison inside autocast. Evaluation actually runs under autocast,
    so an fp32-only result certifies a precision the run never uses. Under fp16 the values cannot
    be bit-equal, so read `amp_argmax_disagreements` -- whether the decision moved -- rather than
    the raw delta.
    """
    model = model.eval()
    with torch.inference_mode():
        h0, l0 = model.forward_both(x)
        h1, l1 = folded.forward_both(x)
        out = {"max_abs_dlogit": float((l0 - l1).abs().max()),
               "max_abs_dfeat": float((h0 - h1).abs().max()),
               "argmax_disagreements": int((l0.argmax(1) != l1.argmax(1)).sum())}
        if amp and x.is_cuda:
            with torch.autocast("cuda", dtype=torch.float16, enabled=True):
                a0, b0 = model.forward_both(x)
                a1, b1 = folded.forward_both(x)
            out.update(amp_max_abs_dlogit=float((b0.float() - b1.float()).abs().max()),
                       amp_max_abs_dfeat=float((a0.float() - a1.float()).abs().max()),
                       amp_argmax_disagreements=int((b0.argmax(1) != b1.argmax(1)).sum()))
    return out


def decisive_agreement(ref: torch.Tensor, got: torch.Tensor, delta: float) -> tuple[int, int]:
    """(disagreements, decisive_rows) among rows whose top-2 gap clearly exceeds the observed
    numerical difference.

    Requiring 100% argmax agreement on near-tied rows rejects a correct build; requiring it on
    decisive rows is the real test. Counted as integers -- `torch.mean` on CUDA multiplies by a
    rounded reciprocal and returns 0.99999994 for a perfect match at many row counts, so a
    float-mean acceptance test fails at random on GPU (perf-federated.md §5).
    """
    top2 = ref.topk(2, dim=-1).values
    decisive = (top2[..., 0] - top2[..., 1]) > max(10 * delta, 1e-3)
    bad = int(((got.argmax(-1) != ref.argmax(-1)) & decisive).sum())
    return bad, int(decisive.sum())


def time_paths(models: list[nn.Module], protos: torch.Tensor, present: torch.Tensor,
               X: torch.Tensor, Y: torch.Tensor, meta_base: nn.Module, *,
               C: int, batch: int, groups=(1, 4, 8, 16, 25), amp: bool = True,
               log=print) -> dict:
    """Time both paths on a real slice of the real test set and return the measurement.

    `groups` is swept because the best G is a property of the device and the batch, not
    something to inherit from another machine.
    """
    folded = [fold_bn(m) for m in models]
    torch.cuda.synchronize()
    t0 = time.time()
    cm_p_ref, cm_c_ref, _ = eval_sequential(folded, protos, present, X, Y,
                                            batch=batch, C=C, amp=amp)
    torch.cuda.synchronize()
    seq_s = time.time() - t0
    rows = X.shape[0] * len(models)
    out = {"batch": batch, "n_clients": len(models), "test_rows": int(X.shape[0]),
           "sequential_seconds": seq_s, "sequential_rows_per_s": rows / seq_s, "vmap": {}}
    log(f"  sequential: {seq_s:.2f}s  ({rows / seq_s:,.0f} client-rows/s)")

    for g in groups:
        if g > len(models):
            continue
        try:
            ev = VmapEvaluator(folded, g, meta_base, half=amp)
            torch.cuda.synchronize()
            t0 = time.time()
            cm_p, cm_c, _ = ev.run(protos, present, X, Y, batch=batch, C=C, amp=amp)
            torch.cuda.synchronize()
            dt = time.time() - t0
            out["vmap"][g] = {
                "seconds": dt, "rows_per_s": rows / dt, "speedup": seq_s / dt,
                "cm_proto_mismatch": int((cm_p - cm_p_ref).abs().sum()),
                "cm_clf_mismatch": int((cm_c - cm_c_ref).abs().sum()),
                "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
            }
            log(f"  vmap G={g:>3}: {dt:.2f}s  ({rows / dt:,.0f} client-rows/s)  "
                f"{seq_s / dt:.2f}x  cm mismatch proto/clf "
                f"{out['vmap'][g]['cm_proto_mismatch']}/{out['vmap'][g]['cm_clf_mismatch']}")
            del ev
        except Exception as e:                       # noqa: BLE001 - report and keep going
            out["vmap"][g] = {"error": f"{type(e).__name__}: {e}"}
            log(f"  vmap G={g:>3}: FAILED {type(e).__name__}: {str(e)[:160]}")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return out
