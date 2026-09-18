"""Class-wise Prototype Sparsification — TinyProto §4.1, Definitions 1 and 2.

One binary mask per class, `s` ones out of `d`, allocated randomly and then optimized for
maximum inter-class Hamming separation. Masks are built once before round 1, recorded in
`config.json`, and never change (paper: "Mask vectors are distributed once at the beginning of
FL rounds and remain fixed").

    structured sparse prototype   c_bar = S(c; m) = c ⊙ m                        Eq. (7)
    compressed prototype          c_hat = C(c; m) = (c_i : m_i = 1) ∈ R^s        Eq. (8)

With K = 16 and d = 256, s = 50 means 16·50 = 800 > 256, so masks MUST overlap. The paper's
CIFAR-10 setting (K = 10, d = 500, s = 50) happens to admit a disjoint partition; ours does not.
The construction below therefore minimizes the worst pairwise overlap instead of reaching zero.

Hamming distance between two s-hot masks is 2(s − overlap), so maximizing the minimum pairwise
Hamming distance is exactly minimizing the maximum pairwise overlap.
"""
from __future__ import annotations

import numpy as np


def _overlaps(Mi: np.ndarray) -> np.ndarray:
    return Mi @ Mi.T


def _objective(counts: np.ndarray, off: np.ndarray) -> tuple[int, int]:
    """Lexicographic: worst pairwise overlap first, then total squared overlap (a tiebreaker
    that keeps flattening the distribution once the max stops moving)."""
    v = counts[off]
    return int(v.max()), int((v ** 2).sum())


def build_masks(d: int, K: int, s: int, seed: int = 42, iters: int = 20_000) -> np.ndarray:
    """(K, d) uint8 masks with exactly `s` ones per row.

    Two stages, both deterministic given `seed`:

    1. Balanced greedy allocation. Class j takes the `s` dimensions used by the fewest classes
       so far, ties broken by the seeded RNG. Flat dimension usage is what drives pairwise
       overlap down: sum of overlaps equals sum_dim C(usage_dim, 2), which a flat usage vector
       minimizes. For (16, 256, 50) this already beats the random expectation s²/d = 9.77.
    2. Local search. Take the worst-overlapping pair, propose several random moves of one of
       its shared dimensions into a dimension that mask does not use, and keep the best move
       that strictly improves the lexicographic objective. Purely greedy moves stall well above
       the heuristic target; this search does not prove global optimality.
    """
    if s > d:
        raise ValueError(f"CPS dimension s={s} exceeds feature dimension d={d}")
    rng = np.random.default_rng(seed)

    # --- stage 1: balanced greedy allocation --------------------------------
    Mi = np.zeros((K, d), dtype=np.int64)
    usage = np.zeros(d, dtype=np.int64)
    for j in range(K):
        order = np.lexsort((rng.random(d), usage))    # by usage, random within a usage level
        pick = order[:s]
        Mi[j, pick] = 1
        usage[pick] += 1

    # --- stage 2: local-search polish ---------------------------------------
    off = ~np.eye(K, dtype=bool)
    counts = _overlaps(Mi)
    best = _objective(counts, off)
    target = int(np.ceil(s * s / d))  # ceil(random expected overlap), not a lower bound

    n_props = 24                    # random (drop, add) proposals examined per iteration
    stall = 0
    for _ in range(iters):
        if best[0] <= target:
            break
        masked = np.where(off, counts, -1)
        j, k = np.unravel_index(int(masked.argmax()), masked.shape)
        move = None
        for src, other in ((j, k), (k, j)):
            shared = np.flatnonzero(Mi[src] & Mi[other])
            free = np.flatnonzero(Mi[src] == 0)
            if shared.size == 0 or free.size == 0:
                continue
            drops = rng.choice(shared, size=min(n_props, shared.size), replace=False)
            adds = rng.choice(free, size=min(n_props, free.size), replace=False)
            for drop, add in zip(drops, adds):
                delta = Mi[:, add] - Mi[:, drop]
                trial = counts + delta[:, None] * (np.arange(K) == src)[None, :] \
                              + delta[None, :] * (np.arange(K) == src)[:, None]
                trial[src, src] = s
                cand = _objective(trial, off)
                if cand < best and (move is None or cand < move[0]):
                    move = (cand, src, int(drop), int(add), trial)
        if move is None:
            stall += 1
            if stall >= 200:        # the worst pair cannot be improved by a single swap
                break
            continue
        stall = 0
        best, src, drop, add, counts = move
        Mi[src, drop], Mi[src, add] = 0, 1

    M = Mi.astype(np.uint8)
    assert (M.sum(axis=1) == s).all(), "mask cardinality drifted during optimization"
    return M


def mask_stats(M: np.ndarray) -> dict:
    """Achieved separation, recorded in config.json so the mask design is auditable."""
    K, d = M.shape
    s = int(M[0].sum())
    Mi = M.astype(np.int64)
    counts = Mi @ Mi.T
    off = ~np.eye(K, dtype=bool)
    ov = counts[off]
    ham = 2 * (s - ov)
    usage = Mi.sum(axis=0)
    return {
        "d": int(d), "K": int(K), "s": s,
        "compression_rate": float(1.0 - s / d),
        "overlap_min": int(ov.min()), "overlap_max": int(ov.max()),
        "overlap_mean": float(ov.mean()),
        "overlap_random_expectation": float(s * s / d),
        "overlap_search_target": int(np.ceil(s * s / d)),
        "hamming_min": int(ham.min()), "hamming_max": int(ham.max()),
        "hamming_mean": float(ham.mean()),
        "dims_used": int((usage > 0).sum()),
        "dim_usage_min": int(usage.min()), "dim_usage_max": int(usage.max()),
    }


def compress(c: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Eq. (8). (K, d) dense -> (K, s), each row gathered with its own class mask."""
    idx = mask_index(M)
    return np.take_along_axis(c, idx, axis=1)


def decompress(chat: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Eq. (7) applied to a compressed vector: (K, s) -> (K, d) structured sparse."""
    K, d = M.shape
    out = np.zeros((K, d), dtype=chat.dtype)
    np.put_along_axis(out, mask_index(M), chat, axis=1)
    return out


def mask_index(M: np.ndarray) -> np.ndarray:
    """(K, s) int64 of active dimension indices per class -- the gather/scatter form used on
    GPU, where boolean indexing would produce data-dependent shapes."""
    K, d = M.shape
    s = int(M[0].sum())
    idx = np.empty((K, s), dtype=np.int64)
    for j in range(K):
        idx[j] = np.flatnonzero(M[j])
    return idx
