"""DAGSNet — Khan et al. 2025 §4.10, Eq. (38)-(48). 395.024 tham số.
Vào: (B, 66) đặc trưng đã z-score.  Ra: (B, 16) logit (CHƯA softmax)."""
import torch
import torch.nn as nn


def cbr(i, o, k):
    """Conv → BatchNorm → ReLU. bias=False vì BatchNorm ngay sau đã có tham số dịch."""
    return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2, bias=False),
                         nn.BatchNorm1d(o), nn.ReLU(inplace=True))


class DenseNet1d(nn.Module):
    """Eq. (38)-(39): mỗi lớp nhận nối của toàn bộ feature map trước đó."""
    def __init__(self, cin, growth, layers):
        super().__init__()
        self.blocks = nn.ModuleList([cbr(cin + i * growth, growth, 3) for i in range(layers)])
        self.out_ch = cin + layers * growth

    def forward(self, x):
        for b in self.blocks:
            x = torch.cat([x, b(x)], dim=1)                     # Eq. (38)
        return x                                                # Eq. (39)


class Inception1d(nn.Module):
    """Eq. (40): bốn nhánh song song 1x1 / 3x3 / 5x5 / pool, nối lại."""
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
    """Eq. (40)-(41): các inception module xếp chồng."""
    def __init__(self, cin, modules_n, c=32):
        super().__init__()
        mods, ch = [], cin
        for _ in range(modules_n):
            m = Inception1d(ch, c); mods.append(m); ch = m.out_ch
        self.net = nn.Sequential(*mods); self.out_ch = ch

    def forward(self, x):
        return self.net(x)


class AlexNet1d(nn.Module):
    """Eq. (42)-(44). ceil_mode=True: trục vị trí chỉ dài 11, không được để pool co về 0."""
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
    """Eq. (45)-(46): squeeze 1x1 nuôi hai nhánh expand 1x1 và 3x3."""
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
    def __init__(self, cfg, n_features):
        super().__init__()
        self.patch_len = cfg["patch_len"]
        assert n_features % self.patch_len == 0
        self.k = n_features // self.patch_len                   # 11
        cin = self.patch_len                                    # 6 kênh

        s = cfg["stem_ch"]
        self.stems   = nn.ModuleList([cbr(cin, s, 1) for _ in range(4)])
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
        # view rồi MỚI transpose: gom 6 cột liên tiếp thành một patch, sau đó patch
        # mới trở thành trục vị trí. Làm view(B, 6, 11) thẳng sẽ trộn sai các cột.
        Fm = x.view(x.shape[0], self.k, self.patch_len).transpose(1, 2)   # (B, 6, 11)
        feats = [gp(stem(Fm)) for stem, gp in
                 zip(self.stems, [self.dense, self.google, self.alex, self.squeeze])]
        pooled = [f.mean(dim=-1) for f in feats]                # global average pool
        return self.head(torch.cat(pooled, dim=1))              # Eq. (47) -> (48)


# Cấu hình ĐÚNG như checkpoint round 5 đã huấn luyện. Đổi bất kỳ giá trị nào ở đây
# thì state_dict sẽ không nạp được — đó là chủ ý.
CFG = {
    "patch_len": 6, "stem_ch": 96, "dense_growth": 32, "dense_layers": 3,
    "incep_modules": 2, "fire_modules": 3, "dropout": 0.1, "num_classes": 16,
}


def build_model(cfg):
    """The rebuild recipe stored beside every weights file. cfg is the dict saved in the
    checkpoint, so a model rebuilt here matches the one that produced those weights."""
    return DAGSNet({k: cfg[k] for k in CFG}, n_features=cfg["n_features"])
