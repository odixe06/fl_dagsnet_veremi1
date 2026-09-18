"""Measure the cost of full-extractor fp32 versus selective autocast policy.

The TPU ``fp32ln`` run disables autocast around all of section 4.8. CUDA's bf16
autocast already implements the candidate policy we want to mirror on XLA: LayerNorm
and softmax stay fp32 while linear/matmul use bf16. This script compares those scopes on
the real model and real VeReMi rows. Peak host use stays below the loader's 1 GB budget.
"""

import argparse
import sys
import time
import types

import numpy as np
import torch

sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--rows", type=int, default=120_000)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=8)
    return parser.parse_args()


def dags_forward(model, fused):
    feats = [
        backbone(stem(fused))
        for stem, backbone in zip(
            model.stems,
            [model.dense, model.google, model.alex, model.squeeze],
        )
    ]
    return model.head(torch.cat([feat.mean(dim=-1) for feat in feats], dim=1))


def full_extractor_fp32(self, x):
    with torch.autocast(x.device.type, enabled=False):
        fused = self.fuse(x.float()).index_select(1, self.sel_ch)
    return dags_forward(self, fused)


def build(scope):
    torch.manual_seed(42)
    model, _ = R.build("cuda", load_init=True)
    if scope == "full_extractor_fp32":
        model.forward = types.MethodType(full_extractor_fp32, model)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
    return model, optimizer


def run(scope, x, y, args):
    model, optimizer = build(scope)
    criterion = torch.nn.CrossEntropyLoss()
    batches = len(x) // args.batch
    rng = np.random.default_rng(7)
    order = rng.permutation(len(x))
    elapsed = None
    finite = 0
    for step in range(args.warmup + args.steps):
        if step == args.warmup:
            torch.cuda.synchronize()
            started = time.perf_counter()
        offset = (step % batches) * args.batch
        idx = np.sort(order[offset : offset + args.batch])
        xb = torch.from_numpy(x[idx]).to("cuda")
        yb = torch.from_numpy(y[idx]).to("cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(xb)
        loss = criterion(logits.float(), yb)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finite += int(torch.isfinite(grad_norm))
        optimizer.step()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - started) / args.steps
    del model, optimizer
    torch.cuda.empty_cache()
    print(
        f"{scope:24s} {elapsed * 1e3:8.2f} ms/step  "
        f"{args.batch / elapsed:9,.0f} samples/s  finite={finite}/{args.steps + args.warmup}"
    )
    return elapsed


def compare_initial_logits(x, args):
    full, _ = build("full_extractor_fp32")
    selective, _ = build("selective_autocast")
    selective.load_state_dict(full.state_dict())
    full.eval()
    selective.eval()
    xb = torch.from_numpy(x[: args.batch]).to("cuda")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = full(xb).float()
        actual = selective(xb).float()
    print(
        "initial logits: "
        f"max_abs_diff={float((expected - actual).abs().max()):.6g}, "
        f"argmax_agreement={float((expected.argmax(1) == actual.argmax(1)).float().mean()):.4%}"
    )
    del full, selective
    torch.cuda.empty_cache()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active environment")
    x, y = R.load_train(args.rows, seed=7)
    print(f"real rows={len(x):,}, batch={args.batch}, host_X={x.nbytes / 2**20:.1f} MiB")
    compare_initial_logits(x, args)
    full = run("full_extractor_fp32", x, y, args)
    selective = run("selective_autocast", x, y, args)
    print(f"selective speedup: {full / selective:.3f}x")


if __name__ == "__main__":
    main()
