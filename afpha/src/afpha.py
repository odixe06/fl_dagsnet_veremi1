"""AFPHA: the approved Proposal A specification (papers/khan-2025-afpha/rebuild.md).

Every constant here completes a gap the paper leaves open. They are implementation
choices, not the authors' published hyperparameters, and the two-level averaging is
algebraically identical to sample-weighted FedAvg -- say so in any write-up.
"""
import math

import numpy as np
import torch

import flatpack

CLUSTER_SIZE = 5
CLUSTER_SEED = 42
SHUFFLE_SEED = 42
MU_BASE = 0.01
MU_EPS = 1e-12
LR_MAX = 1e-3
LR_MIN = 1e-5
GRAD_CLIP = 1.0
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
ADAM_WEIGHT_DECAY = 0.0


def build_clusters(num_clients, cluster_size=CLUSTER_SIZE, seed=CLUSTER_SEED):
    """Fixed clusters from one seeded permutation; a short final cluster is kept.

    Clusters are a logical grouping only -- nothing about client ids implies a
    geographic position or an RSU.
    """
    perm = np.random.default_rng(seed).permutation(num_clients)
    return [sorted(int(c) for c in perm[i:i + cluster_size])
            for i in range(0, num_clients, cluster_size)]


def lr_for_round(rnd, total_rounds):
    """Cosine over communication rounds, fixed within a round. Round index is 1-based.

    A single-round run has no schedule to descend, so it stays at LR_MAX; without this
    the denominator is zero. The configured run is 50 rounds and never takes that path,
    but short smoke tests do.
    """
    assert 1 <= rnd <= total_rounds
    if total_rounds == 1:
        return LR_MAX
    return LR_MIN + (LR_MAX - LR_MIN) / 2 * (1 + math.cos(math.pi * (rnd - 1) / (total_rounds - 1)))


def client_shuffle_seed(rnd, client_id, seed=SHUFFLE_SEED):
    """Deterministic per (round, client) permutation seed."""
    return (seed * 1_000_003 + rnd * 10_007 + client_id) % (2 ** 31 - 1)


def initial_mu(num_clients):
    return {int(i): MU_BASE for i in range(num_clients)}


def next_mu(drifts, rows):
    """mu_i,t+1 = 0.01 * (1 + d_i / (d_i + d_bar + 1e-12)), d_bar weighted by n_i/N.

    `drifts` and `rows` are dicts keyed by client id. The coefficient stays in
    [0.01, 0.02): a client that moved further from the new global is held tighter
    next round. If every drift is zero the rule degenerates to the base value.
    """
    total = float(sum(rows.values()))
    d_bar = sum(rows[i] / total * float(drifts[i]) for i in drifts)
    return {int(i): MU_BASE * (1.0 + float(drifts[i]) / (float(drifts[i]) + d_bar + MU_EPS))
            for i in drifts}


# ------------------------------------------------------------------------ aggregation

def aggregate(float_vecs, int_vecs, rows):
    """Sample-weighted average of flat state vectors.

    Callers pass a fixed, deterministic order -- never completion order. Float
    entries (parameters and BatchNorm running stats) are averaged by n_i/N;
    `num_batches_tracked` counters take the elementwise max.
    """
    assert len(float_vecs) == len(rows) and float_vecs
    total = float(sum(rows))
    weights = [float(r) / total for r in rows]
    return flatpack.weighted_mean(float_vecs, weights), flatpack.elementwise_max(int_vecs)


def hierarchical_aggregate(client_float, client_int, client_rows, clusters):
    """Cluster average, then server average -- the HFL half of AFPHA.

    Ordering follows `clusters` (sorted ids), so floating-point addition order
    never depends on which GPU happened to finish first. With both levels
    sample-weighted this equals sample-weighted FedAvg over all clients; the
    hierarchy is a system description, not an optimization difference.
    """
    c_float, c_int, c_rows = [], [], []
    for members in clusters:
        f, n = aggregate([client_float[i] for i in members],
                         [client_int[i] for i in members],
                         [client_rows[i] for i in members])
        c_float.append(f); c_int.append(n); c_rows.append(sum(client_rows[i] for i in members))
    gf, gn = aggregate(c_float, c_int, c_rows)
    return gf, gn, c_rows


def param_drift(client_float, global_float, n_params):
    """d_i = || w_i - w_global ||_2 over learnable parameters only, in float64.

    Parameters occupy the leading block of the flat layout (see flatpack.Template).
    """
    d = client_float[:n_params].to(torch.float64) - global_float[:n_params].to(torch.float64)
    return float(torch.linalg.vector_norm(d))
