"""EDL-CMSO feature extractor + DAGSNet. Equation numbers refer to Khan et al. 2025.

Build 2 differs from build 1 in exactly two places, both marked below:
  * Eq. (28) concatenates the RAW wavelet patch, not its embedding  -> fused = P + 2d
  * a CMSO channel mask sits between fusion and DAGSNet             -> the paper's order
"""
import math
import torch, torch.nn as nn, torch.nn.functional as F

# Everything the CMSO probe must reproduce bit-for-bit. The mask is selected on the
# features this half of the network emits at initialisation, so training has to start
# from these exact weights or the mask refers to a projection that no longer exists.
EXTRACTOR_PREFIXES = ("dwt.", "patch.", "vit.", "vit_norm.", "gat.")


class HaarDWT(nn.Module):
    """Eq. (19). Haar (db1), level 1, as a fixed non-learnable stride-2 conv so the
    transform stays on-GPU inside the graph. Returns [C_A || C_D]."""
    def __init__(self):
        super().__init__()
        s = 1.0 / math.sqrt(2.0)
        filt = torch.tensor([[[s, s]], [[-s, s]]])          # (2,1,2): lo=approx, hi=detail
        self.register_buffer("filt", filt)

    def forward(self, x):                                    # (B, n)
        if x.shape[1] % 2:
            x = F.pad(x, (0, 1))
        y = F.conv1d(x.unsqueeze(1), self.filt, stride=2)     # (B, 2, n/2)
        return torch.cat([y[:, 0], y[:, 1]], dim=1)           # (B, n)


class PatchEmbed(nn.Module):
    """Eq. (20)-(21). k patches of length P, linear -> d, plus a learned E_pos.

    Returns (x_patch, z'). x_patch is the RAW wavelet patch and is the X_wavelet term of
    Eq. (28) read literally — build 2's decision C. Build 1 returned the projection z
    instead, making all three fusion terms d-dimensional."""
    def __init__(self, n_in, patch, d):
        super().__init__()
        self.patch = patch
        self.k = math.ceil(n_in / patch)
        self.pad = self.k * patch - n_in
        self.proj = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.zeros(1, self.k, d))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, xw):
        if self.pad:
            xw = F.pad(xw, (0, self.pad))
        xp = xw.view(xw.size(0), self.k, self.patch)              # raw C_A||C_D patches
        return xp, self.proj(xp) + self.pos                       # Eq. (20)-(21)


class ViTBlock(nn.Module):
    """Eq. (22)-(25), pre-norm. nn.MultiheadAttention is exactly Eq. (22)-(24):
    scaled dot-product over Q=z'W_Q, K=z'W_K, V=z'W_V, heads concatenated then W_O."""
    def __init__(self, d, heads, mlp_ratio, drop):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, mlp_ratio * d), nn.GELU(),
                                 nn.Dropout(drop), nn.Linear(mlp_ratio * d, d), nn.Dropout(drop))

    def forward(self, z):
        h = self.n1(z)
        z = z + self.att(h, h, h, need_weights=False)[0]      # Eq. (22)-(24)
        return z + self.ffn(self.n2(z))                        # Eq. (25)


class GATLayer(nn.Module):
    """Eq. (26)-(27). The paper never defines N(i); Eq. (26)'s numerator is all-pairs,
    so the graph is fully connected over the k patch nodes.
    a^T[W_h z_i || W_h z_j] = a_src^T W_h z_i + a_dst^T W_h z_j — two matmuls, not k^2
    concatenations."""
    def __init__(self, d_in, d_out, heads, slope, drop):
        super().__init__()
        assert d_out % heads == 0
        self.h, self.dh = heads, d_out // heads
        self.W = nn.Linear(d_in, d_out, bias=False)            # W_h
        self.a_src = nn.Parameter(torch.empty(1, heads, 1, self.dh))
        self.a_dst = nn.Parameter(torch.empty(1, heads, 1, self.dh))
        nn.init.xavier_uniform_(self.a_src); nn.init.xavier_uniform_(self.a_dst)
        self.slope = slope
        self.drop = nn.Dropout(drop)

    def forward(self, z):                                       # (B,k,d_in)
        B, k, _ = z.shape
        h = self.W(z).view(B, k, self.h, self.dh).transpose(1, 2)          # (B,H,k,dh)
        e = (h * self.a_src).sum(-1).unsqueeze(-1) \
          + (h * self.a_dst).sum(-1).unsqueeze(-2)                          # (B,H,k,k)
        alpha = self.drop(torch.softmax(F.leaky_relu(e, self.slope), dim=-1))   # Eq. (26)
        out = (alpha @ h).transpose(1, 2).reshape(B, k, -1)                 # Eq. (27)
        return F.elu(out)                                                    # sigma


# ───────────────────────── DAGSNet backbones (1-D) ─────────────────────────

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
    """Eq. (42)-(44): conv+ReLU stack with max pooling. The token axis is only k long,
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


class EDLCMSO(nn.Module):
    """DWT -> ViT -> GAT -> fusion (Eq. 28) -> CMSO mask (Eq. 29-37) -> DAGSNet (Eq. 47-48).

    n_features is ALWAYS the full input width (66): build 2 selects fused channels, not
    input columns, so nothing upstream of Eq. (28) ever changes shape between runs.
    sel_ch = None builds the unmasked network, which is what the CMSO probe uses."""

    def __init__(self, cfg, n_features, sel_ch=None):
        super().__init__()
        d = cfg["d_model"]
        self.dwt = HaarDWT()
        # HaarDWT pads an odd-length input to even, so the coefficient vector it returns
        # is n + (n % 2) long — not n. PatchEmbed must be sized on THAT.
        dwt_len = n_features + (n_features % 2)
        self.patch = PatchEmbed(dwt_len, cfg["patch_len"], d)
        self.vit = nn.Sequential(*[ViTBlock(d, cfg["vit_heads"], cfg["vit_mlp_ratio"], cfg["dropout"])
                                   for _ in range(cfg["vit_layers"])])
        self.vit_norm = nn.LayerNorm(d)
        self.gat = GATLayer(d, d, cfg["gat_heads"], cfg["gat_slope"], cfg["dropout"])

        # Eq. (28) read literally: raw wavelet patch || z_final || z_new.
        self.fused_dim = cfg["patch_len"] + 2 * d
        if sel_ch is None:
            sel_ch = list(range(self.fused_dim))
        sel = torch.as_tensor(sel_ch, dtype=torch.long)
        assert sel.numel() > 0 and int(sel.max()) < self.fused_dim, "channel mask out of range"
        self.register_buffer("sel_ch", sel)

        s = cfg["stem_ch"]
        self.stems = nn.ModuleList([cbr(sel.numel(), s, 1) for _ in range(4)])
        self.dense  = DenseNet1d(s, cfg["dense_growth"], cfg["dense_layers"])
        self.google = GoogleNet1d(s, cfg["incep_modules"])
        self.alex   = AlexNet1d(s)
        self.squeeze= SqueezeNet1d(s, cfg["fire_modules"])
        comb = self.dense.out_ch + self.google.out_ch + self.alex.out_ch + self.squeeze.out_ch

        self.head = nn.Sequential(                               # Eq. (48)
            nn.LayerNorm(comb), nn.Dropout(cfg["dropout"]),
            nn.Linear(comb, 256), nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout"]), nn.Linear(256, cfg["num_classes"]))

    # §4.8 in one call — this is the half CMSO sees, and the half init_state pins down.
    def fuse(self, x):
        xw = self.dwt(x)                                         # Eq. (19)
        xp, zp = self.patch(xw)                                  # Eq. (20)-(21)
        z_final = self.vit_norm(self.vit(zp))                    # Eq. (22)-(25)
        z_new = self.gat(z_final)                                # Eq. (26)-(27)
        return torch.cat([xp, z_final, z_new], dim=-1).transpose(1, 2)   # Eq. (28) (B,fused,k)

    def forward(self, x):
        Fm = self.fuse(x).index_select(1, self.sel_ch)           # §4.9 acts here
        feats = [gp(stem(Fm)) for stem, gp in
                 zip(self.stems, [self.dense, self.google, self.alex, self.squeeze])]
        pooled = [f.mean(dim=-1) for f in feats]                 # global average pool
        return self.head(torch.cat(pooled, dim=1))               # Eq. (47) -> (48)


def build_model(cfg, n_features, sel_ch=None):
    return EDLCMSO(cfg, n_features, sel_ch)


def extractor_state(model):
    """Just §4.8's parameters — the ones the CMSO mask was selected against."""
    core = getattr(model, "module", model)
    return {k: v.detach().cpu().clone() for k, v in core.state_dict().items()
            if k.startswith(EXTRACTOR_PREFIXES)}


def load_extractor_state(model, sd):
    """Restore §4.8 exactly. DAGSNet keys are expected to be missing — its width depends
    on |S|, which was not known when the probe was built."""
    core = getattr(model, "module", model)
    own = set(core.state_dict())
    unknown = [k for k in sd if k not in own]
    assert not unknown, f"extractor_init.pt has keys this model does not: {unknown[:5]}"
    missing, unexpected = core.load_state_dict(sd, strict=False)
    assert not unexpected, unexpected
    assert not [k for k in missing if k.startswith(EXTRACTOR_PREFIXES)], \
        "extractor_init.pt did not cover every §4.8 parameter"
    return len(sd)


class SurrogateCNN(nn.Module):
    """Fitness evaluator for CMSO (deviation 19). Not part of the paper's model — it is a
    miniature DAGSNet: 1-D convolutions over the same k token positions, global pool,
    linear head. Structurally analogous to the real classifier so the fitness reflects
    what DAGSNet will see, but small enough that 1000 candidate subsets cost minutes
    rather than the weeks a full-wrapper evaluation would."""
    def __init__(self, cin, n_out, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, hidden, 3, padding=1, bias=False),
            nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Conv1d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm1d(hidden), nn.ReLU(inplace=True))
        self.head = nn.Linear(hidden, n_out)

    def forward(self, x):                                        # (B, cin, k)
        return self.head(self.net(x).mean(-1))
