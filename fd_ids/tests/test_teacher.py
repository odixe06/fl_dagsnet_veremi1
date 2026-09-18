"""The cached teacher must be the same teacher.

teacher_logits computes Z_t once per client at batch 16384 and stores it as fp16. The
docstring calls that "exact", and it is exact in the sense that matters -- the teacher is
frozen in eval() for the whole round, so its output depends only on the input row. What it
is not is bit-identical: the cache quantises to fp16 and a large batch fuses differently
from the training batch. This measures the gap that is actually there, on the loss and on
the gradient the student receives, rather than asserting it away.
"""
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/fd-ids-2025"))
from proj.model import build_model, CFG
from proj.fdids import layout, flatten, teacher_logits, client_update

CFGD = {**CFG, "n_features": 66, "batch": 64, "local_epochs": 1, "lam": 0.5, "beta": 0.1,
        "mu": 0.01, "temperature": 3.0, "clip": 1.0, "device": "cpu", "lr": 1e-3}
N, T = 512, 3.0


def kd_grad(student, x, y, zt):
    student.zero_grad(set_to_none=True)
    zs = student(x).float()
    l = 0.5 * F.cross_entropy(zs, y) + 0.5 * (T * T) * F.kl_div(
        F.log_softmax(zs / T, 1), F.log_softmax(zt.float() / T, 1),
        reduction="batchmean", log_target=True)
    l.backward()
    g = torch.cat([p.grad.reshape(-1).clone() for p in student.parameters()])
    student.zero_grad(set_to_none=True)
    return float(l.detach()), g


def main():
    torch.manual_seed(0)
    X = torch.randn(N, 66)
    Y = torch.randint(0, 16, (N,))
    teacher = build_model(CFGD).eval()
    for p in teacher.parameters(): p.requires_grad_(False)

    # 1. the cache is invariant to the batch size it was computed at
    ref = teacher_logits(teacher, X, 0, N, CFGD, batch=N)
    worst = 0.0
    for b in (16, 64, 300, 4096):
        got = teacher_logits(teacher, X, 0, N, CFGD, batch=b)
        worst = max(worst, (got.float() - ref.float()).abs().max().item())
    # NOT zero, and the docstring used to imply it was: a different batch size reduces in a
    # different order. Below the fp16 storage resolution, so it cannot compound.
    print(f"   teacher cache across batch 16..4096: max|dZt| = {worst:.3e}")
    assert worst < 1e-3, worst

    # 2. cached fp16 vs an online fp32 teacher: the gap on loss and on student gradient
    student = build_model(CFGD).train()
    torch.manual_seed(1)
    x, y = X[:64], Y[:64]
    with torch.no_grad():
        online = teacher(x).float()
    cached = ref[:64]
    dz = (cached.float() - online).abs().max().item()
    torch.manual_seed(2); l_on, g_on = kd_grad(student, x, y, online)
    torch.manual_seed(2); l_ca, g_ca = kd_grad(student, x, y, cached)
    dl = abs(l_on - l_ca)
    dg = (g_on - g_ca).norm().item() / (g_on.norm().item() + 1e-12)
    print(f"   fp16 cache vs online teacher: max|dZt|={dz:.3e}  "
          f"|dloss|={dl:.3e}  rel|dgrad|={dg:.3e}")
    # fp16 has ~3 decimal digits; logits of order 1 round at ~1e-3. State the tolerance
    # rather than discover it later from a number that looked wrong.
    assert dz < 1e-2 and dl < 1e-3 and dg < 1e-3, (dz, dl, dg)

    # 3. the teacher and the anchor must not move during a client update
    fk, ik, npar = layout(student)
    w0, i0 = flatten(student, fk, ik)
    anchor = w0[:npar].clone()
    t_before = torch.cat([p.reshape(-1).clone() for p in teacher.parameters()])
    ZT = teacher_logits(teacher, X, 0, N, CFGD, batch=256)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    g = torch.Generator(); g.manual_seed(4)
    client_update(student, student, opt, torch.amp.GradScaler("cpu", enabled=False),
                  X, Y.to(torch.uint8), 0, N, w0, npar, ZT, CFGD, g)
    t_after = torch.cat([p.reshape(-1).clone() for p in teacher.parameters()])
    assert torch.equal(t_before, t_after), "the teacher moved during the client update"
    assert torch.equal(anchor, w0[:npar]), "the proximal anchor was mutated in place"
    now = torch.cat([p.detach().reshape(-1) for p in student.parameters()])
    assert not torch.equal(now, anchor), "the student did not move at all"
    print(f"   teacher and anchor unchanged; student moved by "
          f"{(now - anchor).norm().item():.4f}")

    print("\nALL 3 TEACHER CHECKS PASSED")


if __name__ == "__main__":
    main()
