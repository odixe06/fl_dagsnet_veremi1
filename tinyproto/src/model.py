"""DAGSNet — Khan et al. 2025 §4.10, Eq. (38)-(48). 395,024 learnable parameters.

Copied verbatim from `knowledge/ARCHITECTURE.md` §6 so that a state_dict written by either
file loads in the other with `strict=True`. The only addition is the feature/classifier split
that prototype-based FL needs: `forward_both` returns the penultimate activation alongside the
logits, without changing any parameter name or shape.

In:  (B, 66) z-scored features.   Out: (B, 16) logits (NOT softmaxed) and (B, 256) features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def cbr(i: int, o: int, k: int) -> nn.Sequential:
    """Conv -> BatchNorm -> ReLU. bias=False because the BatchNorm that follows has its own shift."""
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2, bias=False),
                         nn.BatchNorm1d(o), nn.ReLU(inplace=True))


class DenseNet1d(nn.Module):
    """Eq. (38)-(39): every layer sees the concatenation of all previous feature maps."""

    def __init__(self, cin: int, growth: int, layers: int):
        super().__init__()
        self.blocks = nn.ModuleList([cbr(cin + i * growth, growth, 3) for i in range(layers)])
        self.out_ch = cin + layers * growth

    def forward(self, x):
        for b in self.blocks:
            x = torch.cat([x, b(x)], dim=1)                     # Eq. (38)
        return x                                                # Eq. (39)


class Inception1d(nn.Module):
    """Eq. (40): four parallel branches 1x1 / 3x3 / 5x5 / pool, concatenated."""

    def __init__(self, cin: int, c: int):
        super().__init__()
        self.b1 = cbr(cin, c, 1)
        self.b3 = nn.Sequential(cbr(cin, c, 1), cbr(c, c, 3))
        self.b5 = nn.Sequential(cbr(cin, c, 1), cbr(c, c, 5))
        self.bp = nn.Sequential(nn.MaxPool1d(3, 1, 1), cbr(cin, c, 1))
        self.out_ch = 4 * c

    def forward(self, x):
        return torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)


class GoogleNet1d(nn.Module):
    """Eq. (40)-(41): stacked inception modules."""

    def __init__(self, cin: int, modules_n: int, c: int = 32):
        super().__init__()
        mods, ch = [], cin
        for _ in range(modules_n):
            m = Inception1d(ch, c)
            mods.append(m)
            ch = m.out_ch
        self.net = nn.Sequential(*mods)
        self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class AlexNet1d(nn.Module):
    """Eq. (42)-(44). ceil_mode=True: the position axis is only 11 long, pooling must not
    collapse it to 0."""

    def __init__(self, cin: int, ch: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            cbr(cin, ch, 3), nn.MaxPool1d(2, ceil_mode=True),
            cbr(ch, ch, 3), nn.MaxPool1d(2, ceil_mode=True),
            cbr(ch, ch, 3))
        self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class Fire1d(nn.Module):
    """Eq. (45)-(46): a 1x1 squeeze feeding two expand branches, 1x1 and 3x3."""

    def __init__(self, cin: int, sq: int, ex: int):
        super().__init__()
        self.squeeze = cbr(cin, sq, 1)                          # Eq. (46)
        self.e1 = cbr(sq, ex, 1)
        self.e3 = cbr(sq, ex, 3)
        self.out_ch = 2 * ex

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.e1(s), self.e3(s)], dim=1)       # Eq. (45)


class SqueezeNet1d(nn.Module):
    def __init__(self, cin: int, modules_n: int, sq: int = 32, ex: int = 48):
        super().__init__()
        mods, ch = [], cin
        for _ in range(modules_n):
            m = Fire1d(ch, sq, ex)
            mods.append(m)
            ch = m.out_ch
        self.net = nn.Sequential(*mods)
        self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class DAGSNet(nn.Module):
    """Four parallel backbones -> global average pool -> concat -> FC head.

    The head is kept as one `nn.Sequential` so parameter names stay `head.0`, `head.2`, `head.5`
    exactly as in ARCHITECTURE.md. Prototype-based FL splits it at the ReLU:

        feature extractor f_i  =  trunk + head[0:4]   ->  (B, 256), post-ReLU, non-negative
        classifier       g_i  =  head[4:6]            ->  (B, 16) logits

    That makes `head[3]` the penultimate layer, i.e. the layer whose class-mean activations are
    the prototypes of Eq. (3). Feature dimension d = 256.
    """

    FEATURE_DIM = 256

    def __init__(self, cfg: dict, n_features: int):
        super().__init__()
        self.patch_len = cfg["patch_len"]
        assert n_features % self.patch_len == 0
        self.k = n_features // self.patch_len                   # 11
        cin = self.patch_len                                    # 6 channels

        s = cfg["stem_ch"]
        self.stems = nn.ModuleList([cbr(cin, s, 1) for _ in range(4)])
        self.dense = DenseNet1d(s, cfg["dense_growth"], cfg["dense_layers"])
        self.google = GoogleNet1d(s, cfg["incep_modules"])
        self.alex = AlexNet1d(s)
        self.squeeze = SqueezeNet1d(s, cfg["fire_modules"])
        comb = self.dense.out_ch + self.google.out_ch + self.alex.out_ch + self.squeeze.out_ch

        self.head = nn.Sequential(                              # Eq. (48)
            nn.LayerNorm(comb), nn.Dropout(cfg["dropout"]),
            nn.Linear(comb, 256), nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout"]), nn.Linear(256, cfg["num_classes"]))

    # -- shared trunk -------------------------------------------------------
    def _trunk(self, x: torch.Tensor) -> torch.Tensor:
        # view THEN transpose: group 6 consecutive columns into one patch, and only then make
        # the patch the position axis. A direct view(B, 6, 11) mixes the columns up silently.
        Fm = x.view(x.shape[0], self.k, self.patch_len).transpose(1, 2)   # (B, 6, 11)
        feats = [gp(stem(Fm)) for stem, gp in
                 zip(self.stems, [self.dense, self.google, self.alex, self.squeeze])]
        pooled = [f.mean(dim=-1) for f in feats]                # global average pool
        return torch.cat(pooled, dim=1)                         # Eq. (47) -> (B, 544)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self._trunk(x))                        # Eq. (48) -> (B, 16)

    def forward_both(self, x: torch.Tensor):
        """(features, logits) from ONE trunk pass. Used by every FL code path: training needs
        both the prototype regularizer and the cross-entropy, evaluation needs both prediction
        rules, and neither is worth a second forward."""
        z = self._trunk(x)
        h = self.head[3](self.head[2](self.head[1](self.head[0](z))))     # (B, 256) post-ReLU
        return h, self.head[5](self.head[4](h))                           # (B, 16) logits


# Configuration exactly as ARCHITECTURE.md freezes it. Changing any value here makes the
# published DAGSNet state_dict unloadable -- that is the point.
CFG = {
    "patch_len": 6, "stem_ch": 96, "dense_growth": 32, "dense_layers": 3,
    "incep_modules": 2, "fire_modules": 3, "dropout": 0.1, "num_classes": 16,
}

N_PARAMS = 395_024
N_BUFFER_FLOATS = 3_295            # BatchNorm running_mean/var, excluding num_batches_tracked


def build_model(n_features: int = 66, cfg: dict | None = None) -> DAGSNet:
    """Fresh DAGSNet with PyTorch default initialization. Seed before calling for reproducibility."""
    model = DAGSNet(cfg or CFG, n_features=n_features)
    n = sum(p.numel() for p in model.parameters())
    assert n == N_PARAMS, f"architecture drift: {n:,} parameters instead of {N_PARAMS:,}"
    return model


# ---------------------------------------------------------------------------
# Flat packing: FL ships one vector per client, not a 192-entry state_dict.
# torch.multiprocessing backs every tensor with its own shared-memory file descriptor, so
# 100 clients x 192 tensors exhausts the process fd limit mid-run (perf-federated.md §6).
# ---------------------------------------------------------------------------

def param_keys(model: nn.Module) -> list[str]:
    return [k for k, _ in model.named_parameters()]


def buffer_keys(model: nn.Module) -> tuple[list[str], list[str]]:
    """(float buffers, int64 num_batches_tracked buffers), in state_dict order."""
    floats, ints = [], []
    for k, v in model.named_buffers():
        (ints if v.dtype in (torch.int64, torch.long) else floats).append(k)
    return floats, ints


class FlatPacker:
    """Bidirectional (state_dict <-> flat vectors) conversion with a fixed key order.

    Layout is frozen at construction from a reference module, so a vector packed on one process
    unpacks identically on another. `params` comes first, which keeps `vec[:n_params]` exactly
    the learnable block.
    """

    def __init__(self, model: nn.Module):
        sd = model.state_dict()
        self.p_keys = param_keys(model)
        self.b_keys, self.i_keys = buffer_keys(model)
        self.p_shapes = [tuple(sd[k].shape) for k in self.p_keys]
        self.b_shapes = [tuple(sd[k].shape) for k in self.b_keys]
        self.p_sizes = [sd[k].numel() for k in self.p_keys]
        self.b_sizes = [sd[k].numel() for k in self.b_keys]
        self.n_params = sum(self.p_sizes)
        self.n_buffers = sum(self.b_sizes)
        self.n_ints = len(self.i_keys)

    def pack(self, model: nn.Module):
        sd = model.state_dict()
        p = torch.cat([sd[k].reshape(-1).float() for k in self.p_keys])
        b = (torch.cat([sd[k].reshape(-1).float() for k in self.b_keys])
             if self.b_keys else torch.zeros(0))
        i = (torch.stack([sd[k].reshape(()).long() for k in self.i_keys])
             if self.i_keys else torch.zeros(0, dtype=torch.long))
        return p, b, i

    def to_state_dict(self, p: torch.Tensor, b: torch.Tensor, i: torch.Tensor) -> dict:
        sd, off = {}, 0
        for k, shp, n in zip(self.p_keys, self.p_shapes, self.p_sizes):
            sd[k] = p[off:off + n].reshape(shp).clone()
            off += n
        off = 0
        for k, shp, n in zip(self.b_keys, self.b_shapes, self.b_sizes):
            sd[k] = b[off:off + n].reshape(shp).clone()
            off += n
        for j, k in enumerate(self.i_keys):
            sd[k] = i[j].reshape(()).clone()
        return sd

    def load_into(self, model: nn.Module, p: torch.Tensor, b: torch.Tensor, i: torch.Tensor):
        """In-place copy_ so parameter storage addresses survive -- required for CUDA graphs."""
        with torch.no_grad():
            sd = model.state_dict()
            off = 0
            for k, n in zip(self.p_keys, self.p_sizes):
                sd[k].copy_(p[off:off + n].view_as(sd[k]))
                off += n
            off = 0
            for k, n in zip(self.b_keys, self.b_sizes):
                sd[k].copy_(b[off:off + n].view_as(sd[k]))
                off += n
            for j, k in enumerate(self.i_keys):
                sd[k].copy_(i[j])

    def manifest(self) -> dict:
        return {"param_keys": self.p_keys, "param_shapes": [list(s) for s in self.p_shapes],
                "buffer_keys": self.b_keys, "buffer_shapes": [list(s) for s in self.b_shapes],
                "int_buffer_keys": self.i_keys,
                "n_params": self.n_params, "n_buffers": self.n_buffers, "n_ints": self.n_ints}
