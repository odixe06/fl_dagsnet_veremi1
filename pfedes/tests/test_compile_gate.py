"""The compile gates must certify a CORRECT compiler that has its own RNG stream, reject a
wrong one, fall back on an exception, and restore dropout / trainability on every exit.
CPU, torch.compile replaced by wrappers -- the real Inductor path is tests/test_gpu_local.py."""
import sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/pfedes-yi-2025"))
from proj.model import build_model, build_proxy, CFG
from proj.evaluate import fold_bn
from proj import driver

CFGD = {**CFG, "n_features": 66, "mu": 0.5, "clip": 1.0, "device": "cpu", "compile": True,
        "eval_batch": 64}
B = 64


class IndepRNG(torch.nn.Module):
    """Same maths, own RNG draw -- exactly how Inductor's dropout differs from eager's."""
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): torch.rand(1); return self.m(x)


class Wrong(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x):
        out = self.m(x).clone(); out[:, 0] += 0.3; return out


class Boom(torch.nn.Module):
    def __init__(self, m): super().__init__()
    def forward(self, x): raise RuntimeError("triton exploded")


def gate(wrapper, dropout=0.1):
    torch.manual_seed(0)
    cfg = {**CFGD, "dropout": dropout}
    Fe, Ge = build_model(cfg), build_proxy(cfg)
    xb = torch.randn(B, 66) * 2; yb = torch.randint(0, 16, (B,))
    real = torch.compile
    torch.compile = lambda m, **kw: wrapper(m)
    try:
        Fc, Gc = driver._compile_train(Fe, Ge, cfg, torch.device("cpu"), xb, yb)
    finally:
        torch.compile = real
    ps = [m.p for m in list(Fe.modules()) + list(Ge.modules()) if isinstance(m, torch.nn.Dropout)]
    trainable = Fe.training and Ge.training and all(
        p.requires_grad for p in list(Fe.parameters()) + list(Ge.parameters()))
    return (Fc is not Fe) and (Gc is not Ge), ps, trainable


def gate_eval(wrapper):
    torch.manual_seed(0)
    Te = fold_bn(build_model(CFGD))
    xt = torch.randn(B, 66) * 2
    real = torch.compile
    torch.compile = lambda m, **kw: wrapper(m)
    try:
        Tc = driver._compile_eval(Te, CFGD, torch.device("cpu"), xt)
    finally:
        torch.compile = real
    return Tc is not Te


def main():
    n = 0
    ok, ps, tr = gate(IndepRNG)
    assert ok, "a numerically identical compiler was rejected -- the dropout-mask bug is back"
    assert ps == [0.1] * 4 and tr, (ps, tr)
    print("1. train gate: correct compiler with its own RNG: CERTIFIED, p restored, trainable"); n += 1
    for p in (0.1, 0.0):
        ok, ps, tr = gate(Wrong, dropout=p)
        assert not ok and ps == [p] * 4 and tr, (ok, ps, tr)
        print(f"2. train gate: wrong compiler at dropout={p}: REJECTED, p restored, trainable"); n += 1
    ok, ps, tr = gate(Boom)
    assert not ok and ps == [0.1] * 4 and tr
    print("3. train gate: compiler that raises: eager fallback, p restored, trainable"); n += 1
    assert gate_eval(IndepRNG), "eval gate rejected a correct compiler"
    assert not gate_eval(Wrong), "eval gate certified a compiler that shifts logits"
    assert not gate_eval(Boom), "eval gate did not fall back on an exception"
    print("4. eval gate: certified / rejected / fallback"); n += 3
    print(f"\nALL {n} COMPILE-GATE CHECKS PASSED")


if __name__ == "__main__":
    main()
