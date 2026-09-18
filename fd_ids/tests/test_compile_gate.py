"""The compile gate must certify a CORRECT compiler that has its own RNG stream.

`_compile` compares one production-shaped step of the eager model against the same step of
the compiled one, and `restore()` puts torch's RNG back before each so the two see the same
state. That is enough for BatchNorm and for Dropout in eager -- but Inductor functionalises
RNG and draws its own Philox offsets, so the compiled side's dropout mask cannot be made to
match. The gate then measured the difference between two dropout masks and reported it as
compiler error: max|dlogit| 6.07e-01 at p=0.1 against 7.32e-04 at p=0, on sm_86, where
Triton is not in question. Every T4 run silently fell back to eager and paid ~2x for it.

The fault injection here is a compiler that is numerically identical to the model but
consumes the global RNG stream independently. Against the unfixed gate it is rejected with
"cannot certify"; against the fixed one it is certified. The other three cases guard the
fix from becoming a rubber stamp: a genuinely wrong compiler must still be rejected, and
dropout must be back at its configured p on BOTH exits.
"""
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "papers/fd-ids-2025"))
from proj.model import build_model, CFG
from proj import driver

CFGD = {**CFG, "n_features": 66, "dropout": 0.1, "lam": 0.5, "beta": 0.1, "mu": 0.01,
        "temperature": 3.0, "clip": 1.0, "device": "cpu", "compile": True}
B = 128


class IndepRNG(torch.nn.Module):
    """Same maths, own RNG draw -- exactly how Inductor's dropout differs from eager's."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        torch.rand(1)                      # desynchronise the stream, then compute honestly
        return self.m(x)


class Wrong(torch.nn.Module):
    """A compiler that really is broken: a constant shift big enough to move argmax."""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        out = self.m(x).clone()
        out[:, 0] += 0.3
        return out


def gate(wrapper, dropout=0.1):
    """Run the real gate with `torch.compile` replaced by `wrapper`."""
    torch.manual_seed(0)
    model = build_model({**CFGD, "dropout": dropout})
    sample = torch.randn(B, 66)
    real = torch.compile
    torch.compile = lambda m, **kw: wrapper(m)
    try:
        out = driver._compile(model, {**CFGD, "dropout": dropout}, torch.device("cpu"), sample)
    finally:
        torch.compile = real
    ps = [m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    return out is not model, ps, model


def main():
    n = 0

    # 1. THE BUG: correct compiler, independent RNG, dropout on -> must be certified.
    ok, ps, _ = gate(IndepRNG)
    assert ok, "a numerically identical compiler was rejected -- the dropout-mask bug is back"
    print("1. correct compiler with its own RNG stream: CERTIFIED"); n += 1

    # 2. dropout is restored on the success path.
    assert ps == [0.1, 0.1], f"dropout left at {ps} after a successful gate"
    print("2. dropout restored to 0.1 after success"); n += 1

    # 3. A genuinely wrong compiler is still rejected, dropout or not.
    for p in (0.1, 0.0):
        ok, ps, _ = gate(Wrong, dropout=p)
        assert not ok, f"a compiler that shifts logits by 0.3 was certified at dropout={p}"
        assert ps == [p, p], f"dropout left at {ps} after a rejected gate"
        print(f"3. wrong compiler at dropout={p}: REJECTED, dropout restored"); n += 1

    # 4. The exception path also restores dropout.
    class Boom(torch.nn.Module):
        def __init__(self, m): super().__init__()
        def forward(self, x): raise RuntimeError("triton exploded")
    ok, ps, _ = gate(Boom)
    assert not ok and ps == [0.1, 0.1], f"exception path: certified={ok} dropout={ps}"
    print("4. compiler that raises: fell back to eager, dropout restored"); n += 1

    # 5. Certification must survive the model being left trainable and unchanged.
    ok, _, model = gate(IndepRNG)
    assert model.training, "gate left the model in eval mode"
    print("5. model left in train mode"); n += 1

    print(f"\nALL {n} COMPILE-GATE CHECKS PASSED")


if __name__ == "__main__":
    main()
