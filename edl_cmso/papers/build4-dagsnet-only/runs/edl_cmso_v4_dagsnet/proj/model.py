"""DAGSNet alone — Eq. (38)-(48) of Khan et al. 2025, with §4.8 and §4.9 removed.

Every class below is copied unchanged from build 2's model.py. The ONLY differences are
that EDLCMSO's dwt/patch/vit/gat/fuse are gone and the stem now takes patch_len channels
instead of |S| selected fused channels. Keeping the branches byte-identical is the whole
point: a difference in results then measures the extractor, not a different CNN.
"""
import torch, torch.nn as nn


def cbr(i, o, k):
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2, bias=False),
                         nn.BatchNorm1d(o), nn.ReLU(inplace=True))


class DenseNet1d(nn.Module):
    """Eq. (38)-(39): every layer consumes the concatenation of all preceding maps."""
    def __init__(self, cin, growth, layers):
        super().__init__()
        self.blocks = nn.ModuleList([cbr(cin + i * growth, growth, 3) for i in range(layers)])
        self.out_ch = cin + layers * growth

    def forward(self, x):
        for b in self.blocks:
            x = torch.cat([x, b(x)], dim=1)                     # Eq. (38)
        return x                                                # Eq. (39)


class Inception1d(nn.Module):
    """Eq. (40): parallel 1x1 / 3x3 / 5x5 / pool branches, concatenated."""
    def __init__(self, cin, c):
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
    def __init__(self, cin, modules_n, c=32):
        super().__init__()
        mods, ch = [], cin
        for _ in range(modules_n):
            m = Inception1d(ch, c); mods.append(m); ch = m.out_ch
        self.net = nn.Sequential(*mods); self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class AlexNet1d(nn.Module):
    """Eq. (42)-(44): conv+ReLU stack with max pooling. The position axis is only k long,
    so pooling is ceil_mode to avoid collapsing it to zero."""
    def __init__(self, cin, ch=128):
        super().__init__()
        self.net = nn.Sequential(
            cbr(cin, ch, 3), nn.MaxPool1d(2, ceil_mode=True),
            cbr(ch, ch, 3),  nn.MaxPool1d(2, ceil_mode=True),
            cbr(ch, ch, 3))
        self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class Fire1d(nn.Module):
    """Eq. (45)-(46): 1x1 squeeze feeding parallel 1x1 and 3x3 expands."""
    def __init__(self, cin, sq, ex):
        super().__init__()
        self.squeeze = cbr(cin, sq, 1)                          # Eq. (46)
        self.e1 = cbr(sq, ex, 1)
        self.e3 = cbr(sq, ex, 3)
        self.out_ch = 2 * ex

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.e1(s), self.e3(s)], dim=1)       # Eq. (45)


class SqueezeNet1d(nn.Module):
    def __init__(self, cin, modules_n, sq=32, ex=48):
        super().__init__()
        mods, ch = [], cin
        for _ in range(modules_n):
            m = Fire1d(ch, sq, ex); mods.append(m); ch = m.out_ch
        self.net = nn.Sequential(*mods); self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class DAGSNet(nn.Module):
    """Eq. (38)-(48) applied directly to the raw feature matrix.

    The 66 standardised columns become (B, patch_len, k) with k = 66 / patch_len = 11:
    the same 11-position grid PatchEmbed chunks the DWT output into, fed raw. Two other
    layouts were measured and rejected — (B,1,66) is 1.3x SLOWER than the full pipeline
    because every convolution then runs over 66 positions instead of 11, and (B,66,1)
    degenerates every convolution to 1x1, turning DAGSNet into an MLP."""

    def __init__(self, cfg, n_features):
        super().__init__()
        self.patch_len = cfg["patch_len"]
        assert n_features % self.patch_len == 0, \
            f"{n_features} features do not divide into patches of {self.patch_len}"
        self.k = n_features // self.patch_len                   # 11
        cin = self.patch_len                                    # 6 channels

        s = cfg["stem_ch"]
        self.stems = nn.ModuleList([cbr(cin, s, 1) for _ in range(4)])
        self.dense   = DenseNet1d(s, cfg["dense_growth"], cfg["dense_layers"])
        self.google  = GoogleNet1d(s, cfg["incep_modules"])
        self.alex    = AlexNet1d(s)
        self.squeeze = SqueezeNet1d(s, cfg["fire_modules"])
        comb = self.dense.out_ch + self.google.out_ch + self.alex.out_ch + self.squeeze.out_ch

        self.head = nn.Sequential(                              # Eq. (48)
            nn.LayerNorm(comb), nn.Dropout(cfg["dropout"]),
            nn.Linear(comb, 256), nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout"]), nn.Linear(256, cfg["num_classes"]))

    def forward(self, x):                                       # (B, n_features)
        Fm = x.view(x.shape[0], self.k, self.patch_len).transpose(1, 2)   # (B, 6, 11)
        feats = [gp(stem(Fm)) for stem, gp in
                 zip(self.stems, [self.dense, self.google, self.alex, self.squeeze])]
        pooled = [f.mean(dim=-1) for f in feats]                # global average pool
        return self.head(torch.cat(pooled, dim=1))              # Eq. (47) -> (48)


def build_model(cfg, n_features):
    return DAGSNet(cfg, n_features)
