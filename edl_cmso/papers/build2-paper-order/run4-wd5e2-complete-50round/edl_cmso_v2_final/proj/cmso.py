"""Crayfish-Mother Swarm Optimizer, Eq. (29)-(37) of Khan et al. 2025.

Build 2 searches over the 262 FUSED channels of Eq. (28), not the 66 input columns.
Maximises fitness. Every hyperparameter here fills a gap the paper leaves unstated."""
import json, math, time
import numpy as np, torch, torch.nn as nn
from torch.amp import autocast, GradScaler
from sklearn.metrics import f1_score

from proj.model import SurrogateCNN


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


# Build 1 needed a floor of 2*patch_len because a small INPUT subset collapsed the token
# axis to k=1 and hung DDP. That failure mode is gone in build 2 — k is fixed at 11 — so
# this floor is only about not handing DAGSNet a degenerate one-channel feature map.
MIN_CHANNELS = 16

# ...and at least this many from EACH of Eq. (28)'s three terms.
#
# This is NOT a DDP guard. z_new reaches the loss only through F, but cat + index_select
# leave gat.W / a_src / a_dst in the autograd graph either way, so a GAT-free mask hands
# them a ZERO gradient, not None — measured, not assumed (smoke.py check 5).
# find_unused_parameters=False stays valid regardless.
#
# What a GAT-free mask would actually do is freeze the whole GAT branch at its random
# initialisation for all 50 rounds while still paying for its forward pass every step, and
# reduce Eq. (28) to a two-term concatenation. The same argument applies to the other two
# terms. Two per term keeps the fused map what the equation says it is, at a cost of at
# most 6 of 262 channels taken out of CMSO's discretion.
MIN_PER_TERM = 2


def binarise(pos, term_bounds=None, min_channels=MIN_CHANNELS, min_per_term=MIN_PER_TERM):
    """Deviation 20: S-shaped transfer + 0.5 threshold. Candidates below either floor are
    topped up by |pos| rank."""
    m = _sigmoid(pos) > 0.5
    for a, b in (term_bounds or []):
        if m[a:b].sum() < min_per_term:
            keep = a + np.argsort(-np.abs(pos[a:b]))[:min_per_term]
            m[a:b] = False
            m[keep] = True
    if m.sum() < min_channels:
        order = np.argsort(-np.abs(pos))
        extra = [i for i in order if not m[i]][:min_channels - int(m.sum())]
        m[extra] = True
    return m


class SurrogateFitness:
    """Trains a miniature DAGSNet on the masked fused map and returns macro-F1 on a
    held-out slice. Both slices are extracted from TRAIN ROWS ONLY. Results are cached by
    mask so duplicate candidates — common late in a converging swarm — cost nothing.

    Ftr/Fva are (rows, fused, k) fp16 tensors already resident on the GPU: the channel
    axis is dim 1 so a mask is a single index_select and the layout already matches
    Conv1d's (N, C, L)."""

    def __init__(self, Ftr, ytr, Fva, yva, num_classes, cfg, device):
        self.Ftr, self.ytr, self.Fva, self.yva = Ftr, ytr, Fva, yva
        self.C, self.cfg, self.device = num_classes, cfg, device
        self.cache, self.evals = {}, 0

    def __call__(self, mask):
        key = mask.tobytes()
        if key in self.cache:
            return self.cache[key]
        idx = torch.from_numpy(np.where(mask)[0]).to(self.device)
        g = torch.Generator(device=self.device); g.manual_seed(self.cfg["seed"])
        torch.manual_seed(self.cfg["seed"])

        net = SurrogateCNN(int(mask.sum()), self.C).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=3e-3)
        scaler = GradScaler("cuda", enabled=True)
        crit = nn.CrossEntropyLoss()
        B, N = self.cfg["cmso_surrogate_batch"], self.Ftr.shape[0]

        net.train()
        for _ in range(self.cfg["cmso_surrogate_steps"]):
            sel = torch.randint(0, N, (B,), device=self.device, generator=g)
            xb = self.Ftr[sel].index_select(1, idx).float(); yb = self.ytr[sel].long()
            with autocast("cuda", dtype=torch.float16):
                loss = crit(net(xb), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()

        net.eval(); preds = []
        with torch.inference_mode():
            for i in range(0, self.Fva.shape[0], 32768):
                xb = self.Fva[i:i+32768].index_select(1, idx).float()
                with autocast("cuda", dtype=torch.float16):
                    preds.append(net(xb).argmax(1).cpu())
        f1 = f1_score(self.yva.cpu().numpy(), torch.cat(preds).numpy(),
                      average="macro", labels=np.arange(self.C), zero_division=0)
        fit = float(f1) - self.cfg["cmso_size_penalty"] * mask.sum() / mask.size
        self.cache[key] = fit; self.evals += 1
        del net, opt, scaler; torch.cuda.empty_cache()
        return fit


def run_cmso(fitness_fn, dim, cfg, term_bounds=None, log=print):
    """Returns (best_mask, history). Eq. (29)-(37).

    term_bounds is the list of (start, stop) spans of Eq. (28)'s three concatenated terms;
    every candidate is forced to keep MIN_PER_TERM channels from each. See binarise."""
    rng = np.random.default_rng(cfg["seed"])
    N, T = cfg["cmso_pop"], cfg["cmso_iters"]
    lb, ub = -1.0, 1.0                                   # continuous search bounds
    bina = lambda p: binarise(p, term_bounds)

    X = lb + (ub - lb) * rng.random((N, dim))            # Eq. (29)-(30)
    masks = [bina(x) for x in X]
    fit = np.array([fitness_fn(m) for m in masks])
    gi = int(np.argmax(fit))
    X_G, f_G, m_G = X[gi].copy(), fit[gi], masks[gi].copy()
    hist = []

    for t in range(1, T + 1):
        temp = rng.random() * 15 + 20                    # Eq. (31)
        C1, mu, sg = cfg["cmso_C1"], cfg["cmso_mu"], cfg["cmso_sigma"]
        p = C1 * (1.0 / (math.sqrt(2 * math.pi) * sg)) * math.exp(-((temp - mu) ** 2) / (2 * sg * sg))  # Eq. (32)
        C2 = 2 - t / T                                   # Eq. (35)
        X_L = X[int(np.argmax(fit))]                     # best of the current population

        for i in range(N):
            if temp > 30:
                # ---- summer resort, COA exploration: Eq. (33)-(34) ----
                X_shade = (X_G + X_L) / 2.0
                if rng.random() < 0.5:                   # no competitor at the cave
                    cand = X[i] + C2 * rng.random() * (X_shade - X[i])
                else:                                    # competition: move on a rival
                    j = int(rng.integers(N))
                    cand = X[i] - X[j] + X_shade
            else:
                # ---- foraging, scaled by the intake model p ----
                cand = X[i] + p * rng.normal(0, 0.3, dim) * (X_G - X[i])

            # ---- MOA upbringing, Eq. (36) ----
            cand = cand + (1 - 2 * rng.random(dim)) * (ub - lb) / t
            mut = rng.random(dim) < cfg["cmso_mutation"]
            cand[mut] = lb + (ub - lb) * rng.random(int(mut.sum()))
            cand = np.clip(cand, lb, ub)

            m = bina(cand); f = fitness_fn(m)
            if f >= fit[i]:                              # Eq. (37), sign flipped: we maximise
                X[i], fit[i] = cand, f
                if f > f_G:
                    X_G, f_G, m_G = cand.copy(), f, m.copy()

        hist.append({"iter": t, "best_fitness": float(f_G),
                     "n_selected": int(m_G.sum()), "mean_fitness": float(fit.mean()),
                     "temp": float(temp)})
        log(f"[cmso] iter {t:3d}/{T}  best {f_G:.5f}  |S| {int(m_G.sum()):3d}/{dim}  "
            f"mean {fit.mean():.5f}  temp {temp:.1f}", flush=True)
    return m_G, hist
