"""Per-client evaluation of the personalized model f_s(w_s^k, x) on the fixed global test
set, plus the aggregated proxy f_r(w_bar_r, x).

The lightweight-FL NILM method keeps a personalized model on every device (only its proxy
is uploaded), so a round's evaluation is N + 1 independent full-test passes -- at 100
clients the dominant cost of the run.

Two exact speed-ups, both measured on 2xT4 in this repository's sibling projects:
  * BatchNorm folding: in eval mode BN is an affine map with constant coefficients, so it
    folds into the preceding Conv1d exactly (max|dlogit| 4.8e-07 measured), removing 31
    kernel launches per forward.
  * One folded TEMPLATE module per worker, compiled once with CUDA graphs. Client weights
    are folded and copied INTO the template with load_state_dict (in-place copy_), so
    parameter addresses never change and the captured graph stays valid for every client.
"""
import copy
import torch
import torch.nn as nn


@torch.no_grad()
def fold_bn(model):
    """Return an eval-mode copy with every BatchNorm folded into its preceding Conv1d.

        y = gamma*(conv(x) - mean)/sqrt(var + eps) + beta
          = conv'(x) + b'   with   w' = w*gamma/sqrt(var+eps),  b' = beta - gamma*mean/sqrt(var+eps)

    Exact for model.eval(); meaningless for model.train(). Every BatchNorm in DAGSNet sits
    inside a `cbr` block, i.e. Sequential(Conv1d, BN, ReLU), and the assertion at the end
    is what stops a future architecture change from silently leaving one unfolded."""
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
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def load_folded(template, model):
    """Fold `model` and copy the result into `template` in place (same folded structure).
    strict=True: a key mismatch means the template was built from a different architecture."""
    template.load_state_dict(fold_bn(model).state_dict(), strict=True)
    return template


@torch.inference_mode()
def eval_model(compiled, eager, TX, TY, cfg, want_preds=False):
    """Confusion matrix of one folded model over the whole resident test set.

    Counts stay on the device: a per-batch .item() would sync ~650 times a pass, and the
    argmax of a row of NaN is 0 -- a perfectly ordinary class index -- so non-finite logits
    are counted explicitly instead of trusted. The tail batch runs eagerly (different shape
    would recompile the graph), same weights, same math."""
    C, EB, n = cfg["num_classes"], cfg["eval_batch"], TX.shape[0]
    dev = TX.device
    cm = torch.zeros(C * C, dtype=torch.long, device=dev)
    nonfin = torch.zeros((), dtype=torch.long, device=dev)
    preds = torch.empty(n, dtype=torch.uint8, device=dev) if want_preds else None
    ac = torch.autocast("cuda", dtype=torch.float16) if dev.type == "cuda" \
        else torch.autocast("cpu", enabled=False)
    for i in range(0, n, EB):
        j = min(i + EB, n)
        m = compiled if j - i == EB else eager
        # The previous batch's logits are still referenced by `z` when the next replay
        # starts; without an explicit step boundary CUDA-graph trees treat the call as
        # part of the same iteration and record a NEW graph node instead of replaying
        # (measured locally: 15k rows/s "compiled" vs 242k eager).
        if dev.type == "cuda":
            torch.compiler.cudagraph_mark_step_begin()
        with ac:
            z = m(TX[i:j].float())
        nonfin += (~torch.isfinite(z)).sum()
        p = z.argmax(1)
        cm += torch.bincount(TY[i:j].long() * C + p, minlength=C * C)
        if want_preds:
            preds[i:j] = p.to(torch.uint8)
    return cm.view(C, C), int(nonfin.item()), preds
