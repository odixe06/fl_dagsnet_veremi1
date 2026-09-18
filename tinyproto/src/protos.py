"""Prototypes, APS scaling, and the TinyProto-FP regularizer — paper §3.1 and §4.2.

Everything here is deliberately explicit about which equation it implements and which choice
is this build's rather than the authors'. `papers/tinyproto-lee-2026/rebuild.md` carries the
same table; keep the two in sync.

    local prototype        c_L[i,j] = (1/n_ij) Σ_{(x,y)∈D_ij} f_i(θ_i; x)          Eq. (3)
    APS upload            n_ij · ĉ_L[i,j]   (compressed to s dims)                 Eq. (10)
    global prototype      ĉ_G[j] = (1/|N_j|) Σ_{i∈N_j} n_ij ĉ_L[i,j]               Eq. (10)
    regularizer           R_i = Σ_j ρ( ĉ_L[i,j], μ ĉ_G[j] )                        Eq. (11)
    local objective       CE + λ R_i                                               Eq. (5)
    prediction            ŷ = argmin_j ‖ f_i(θ_i; x) − c_L[i,j] ‖₂                 Eq. (12)
"""
from __future__ import annotations

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Step 1 — local prototype generation, Eq. (3)
# ---------------------------------------------------------------------------

class ProtoAccumulator:
    """Class-wise sums of feature vectors, on device, in fp32.

    Kept as (K, d) sums plus (K,) counts rather than running means so the result is exactly
    the arithmetic mean regardless of batch boundaries, and so an empty class stays exactly 0
    instead of NaN.
    """

    def __init__(self, K: int, d: int, device):
        self.sum = torch.zeros(K, d, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(K, dtype=torch.int64, device=device)

    @torch.no_grad()
    def update(self, feats: torch.Tensor, labels: torch.Tensor) -> None:
        self.sum.index_add_(0, labels, feats.float())
        self.cnt.index_add_(0, labels, torch.ones_like(labels, dtype=torch.int64))

    @torch.no_grad()
    def finish(self) -> tuple[torch.Tensor, torch.Tensor]:
        """(c_L (K, d) dense local prototypes, n (K,) counts). Absent classes stay exactly 0."""
        n = self.cnt
        c = torch.where(n[:, None] > 0, self.sum / n.clamp(min=1.0)[:, None],
                        torch.zeros_like(self.sum))
        return c, n


# ---------------------------------------------------------------------------
# Step 2 — server aggregation with APS, Eq. (10)
# ---------------------------------------------------------------------------

def aggregate_global(uploads: list[tuple[int, np.ndarray, np.ndarray]],
                     K: int, s: int) -> tuple[np.ndarray, np.ndarray]:
    """Server side of Eq. (10).

    `uploads` is [(client_id, scaled_compressed (K, s) float64, present (K,) bool)], and MUST be
    sorted by client id before it reaches here: floating-point addition is not associative, so
    letting GPU completion order decide the summation order would make the run irreproducible.

    Returns (ĉ_G (K, s) float64, |N_j| (K,) int64). A class no client holds gets an all-zero
    prototype and |N_j| = 0; callers must treat that as "no target this round", not as a zero
    target, otherwise the regularizer would pull those features to the origin.
    """
    ids = [u[0] for u in uploads]
    assert ids == sorted(ids), "uploads must be sorted by client id before aggregation"
    acc = np.zeros((K, s), dtype=np.float64)
    n_clients = np.zeros(K, dtype=np.int64)
    for _, scaled, present in uploads:
        acc[present] += scaled[present]
        n_clients += present.astype(np.int64)
    out = np.zeros_like(acc)
    nz = n_clients > 0
    out[nz] = acc[nz] / n_clients[nz][:, None]
    return out, n_clients


def scale_and_compress(c_local: np.ndarray, counts: np.ndarray,
                       mask_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Client side of Eq. (10): the payload actually put on the wire.

    Sends `n_ij · ĉ_L[i,j]` as one (K, s) block. n_ij is never transmitted separately -- that
    is exactly the privacy property APS buys: the server sees a scaled prototype and cannot
    read the class count out of it.
    """
    present = counts > 0
    compressed = np.take_along_axis(c_local.astype(np.float64), mask_idx, axis=1)
    return compressed * counts.astype(np.float64)[:, None], present


# ---------------------------------------------------------------------------
# Step 3 — the regularizer, Eq. (5) + Eq. (11)
# ---------------------------------------------------------------------------

class ProtoRegularizer:
    """μ-scaled, mask-restricted pull of each sample's feature toward its class prototype.

    Two implementation choices, both recorded in rebuild.md:

    * **Per-sample, not per-class.** Eq. (11) sums a distance over classes. FedProto's released
      code -- which this paper builds on and does not modify here -- applies the term per sample
      against the global prototype of that sample's label. Per-sample gives a gradient on every
      row of the batch instead of one on the batch's class means; it is the reference behaviour
      and the one that survives large batches.
    * **ρ = mean squared error over the s active dimensions**, i.e.
      `‖(h − μ ĉ_G) ⊙ m‖² / s`, not the raw Euclidean norm. With s = d this is exactly
      `nn.MSELoss()` on dense prototypes, so it reduces to FedProto's term when CPS is off, and
      it keeps λ = 1 meaningful across different s.

    Rows whose class has no global prototype this round contribute exactly zero and are excluded
    from the denominator, so round 1 (no prototypes yet) yields R_i = 0 without a NaN.
    """

    def __init__(self, sparse_global: torch.Tensor, has_global: torch.Tensor,
                 mask: torch.Tensor, mu: torch.Tensor, s: int):
        # sparse_global: (K, d) already masked and already multiplied by nothing else.
        # mu:            (K,) or scalar tensor -- per-class μ is supported for the ablation.
        self.target = (sparse_global * (mu[:, None] if mu.ndim else mu))     # (K, d)
        self.has_global = has_global.float()                                 # (K,)
        self.mask = mask                                                     # (K, d) float
        self.s = float(s)

    def __call__(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        tgt = self.target.index_select(0, labels)                # (B, d)
        msk = self.mask.index_select(0, labels)                  # (B, d)
        ok = self.has_global.index_select(0, labels)             # (B,)
        diff = (feats - tgt) * msk
        per = diff.pow(2).sum(dim=1) * ok
        return per.sum() / (ok.sum().clamp(min=1.0) * self.s)


# ---------------------------------------------------------------------------
# Eq. (12) — prototype-distance prediction
# ---------------------------------------------------------------------------

def proto_logits(feats: torch.Tensor, c_local: torch.Tensor,
                 present: torch.Tensor) -> torch.Tensor:
    """Scores whose argmax equals argmin_j ‖h − c_L[j]‖₂.

    ‖h − p‖² = ‖h‖² − 2h·p + ‖p‖². ‖h‖² is constant across j for a given row, so it drops out
    of the argmin; what remains is a single matmul plus a per-class bias. Classes the client
    has never seen have no prototype and are masked to −inf so they can never be predicted --
    that is inherent to personalized PBFL, not a bug, and it is why per-client recall on an
    absent class is 0.
    """
    scores = feats @ c_local.t() - 0.5 * c_local.pow(2).sum(dim=1)[None, :]
    return scores.masked_fill(~present[None, :], float("-inf"))


def communication_cost(present_per_client: np.ndarray, K: int, s: int, d: int) -> dict:
    """Table 2's cost formulation, Σ_i (K_i + K) × s parameters per round.

    K_i is the number of classes client i holds (its upload); K is the number of classes the
    server sends back (its download). The dense figure is the same formula with s replaced by
    d, i.e. what FedProto would have cost on this partition -- reported alongside because
    communication volume, not accuracy, is what the paper is actually about.
    """
    K_i = present_per_client.sum(axis=1).astype(np.int64)
    total = int((K_i + K).sum())
    return {
        "sum_Ki_plus_K": total,
        "params_per_round": total * int(s),
        "params_per_round_dense_fedproto": total * int(d),
        "compression_vs_fedproto": float(d) / float(s),
        "K_i_mean": float(K_i.mean()), "K_i_min": int(K_i.min()), "K_i_max": int(K_i.max()),
    }
