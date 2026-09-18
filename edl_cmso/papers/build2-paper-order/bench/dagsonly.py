"""How fast is DAGSNet without the §4.8 extractor, and how do the input layouts compare?

DAGSNet consumes (B, C, L): C channels over L positions. The full pipeline hands it
(B, 133, 11) — 133 CMSO-selected fused channels over 11 ViT tokens. Removing §4.8 and §4.9
means mapping 66 raw features into (B, C, L) directly, and there is more than one honest way:

  patch  (B, 6, 11)   the same 11-position grid PatchEmbed chunks the DWT output into,
                      fed raw. Only the stem's input width changes, so every branch keeps
                      its exact geometry -> the cleanest ablation of the extractor.
  seq    (B, 1, 66)   one channel, length 66: DAGSNet as a plain 1-D CNN over the feature
                      vector. 6x more positions, larger receptive field, more compute.
  chan   (B, 66, 1)   66 channels, length 1. Convolutions over a length-1 axis degenerate
                      to 1x1, so this quietly turns DAGSNet into an MLP. Measured for
                      completeness, not proposed.

Ratios measured here transfer GPU->GPU, so the 2xT4 projection uses the c50 measurement
(565.7 s/round at batch 4,096) scaled by the ratio, not this card's absolute times.

8 GB box: 120,000 rows only.
"""
import sys, time, json, torch, torch.nn as nn
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R

sys.path.insert(0, str(R.PKG))
import proj.model as M

dev, B, WARM, ITER = "cuda", 1024, 10, 40
CFG = R.CFG
C50_S_PER_ROUND, STEPS_PER_ROUND = 565.7, 43_045_415 // 4096


class DAGSNetOnly(nn.Module):
    """Eq. (38)-(48) only: the four branches, the fusion of their pooled outputs, the head.

    No DWT, no ViT, no GAT, no CMSO mask. Every branch below is the SAME class the full
    model uses, so a difference in results is the extractor's contribution and not a
    different CNN."""

    def __init__(self, cfg, n_features, layout):
        super().__init__()
        self.layout, self.n_features = layout, n_features
        self.patch_len = cfg["patch_len"]
        if layout == "patch":
            assert n_features % self.patch_len == 0, "patch layout needs n % patch_len == 0"
            self.k, cin = n_features // self.patch_len, self.patch_len
        elif layout == "seq":
            self.k, cin = n_features, 1
        elif layout == "chan":
            self.k, cin = 1, n_features
        else:
            raise ValueError(layout)

        s = cfg["stem_ch"]
        self.stems = nn.ModuleList([M.cbr(cin, s, 1) for _ in range(4)])
        self.dense = M.DenseNet1d(s, cfg["dense_growth"], cfg["dense_layers"])
        self.google = M.GoogleNet1d(s, cfg["incep_modules"])
        self.alex = M.AlexNet1d(s)
        self.squeeze = M.SqueezeNet1d(s, cfg["fire_modules"])
        comb = self.dense.out_ch + self.google.out_ch + self.alex.out_ch + self.squeeze.out_ch
        self.head = nn.Sequential(
            nn.LayerNorm(comb), nn.Dropout(cfg["dropout"]),
            nn.Linear(comb, 256), nn.ReLU(inplace=True),
            nn.Dropout(cfg["dropout"]), nn.Linear(256, cfg["num_classes"]))

    def shape_input(self, x):
        if self.layout == "patch":
            return x.view(x.shape[0], self.k, self.patch_len).transpose(1, 2)
        if self.layout == "seq":
            return x.unsqueeze(1)
        return x.unsqueeze(-1)

    def forward(self, x):
        Fm = self.shape_input(x)
        feats = [gp(stem(Fm)) for stem, gp in
                 zip(self.stems, [self.dense, self.google, self.alex, self.squeeze])]
        return self.head(torch.cat([f.mean(dim=-1) for f in feats], dim=1))


def timed(model, X, Y, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    crit, sc = nn.CrossEntropyLoss(), torch.amp.GradScaler("cuda")
    n = len(X) // B
    for i in range(WARM + ITER):
        if i == WARM:
            torch.cuda.synchronize(); t0 = time.time()
        j = (i % n) * B
        xb, yb = X[j:j+B], Y[j:j+B]
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(xb)
        sc.scale(crit(out.float(), yb)).backward()
        sc.step(opt); sc.update()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / ITER * 1000
    p = sum(q.numel() for q in model.parameters())
    print(f"  {tag:<34} {ms:8.2f} ms/step  {p:>9,} params")
    return ms, p


Xh, Yh = R.load_train(120_000, seed=7)
X = torch.from_numpy(Xh).to(dev)
Y = torch.from_numpy(Yh).to(dev)
print(f"rows {len(X):,}  batch {B}  (ms/step is this card; only RATIOS transfer)\n")

full = M.build_model(CFG, 66, CFG["sel_ch"]).to(dev)
base, pf = timed(full, X, Y, "full pipeline (B,133,11)")
del full; torch.cuda.empty_cache()

out = {"full": {"ms": base, "params": pf}}
for layout, label in (("patch", "DAGSNet only (B,6,11)  patch"),
                      ("seq",   "DAGSNet only (B,1,66)  seq"),
                      ("chan",  "DAGSNet only (B,66,1)  chan")):
    m = DAGSNetOnly(CFG, 66, layout).to(dev)
    ms, p = timed(m, X, Y, label)
    out[layout] = {"ms": ms, "params": p, "speedup": base / ms,
                   "t4_s_per_round": C50_S_PER_ROUND * ms / base,
                   "t4_h_50_rounds": 50 * C50_S_PER_ROUND * ms / base / 3600}
    del m; torch.cuda.empty_cache()

print(f"\n{'layout':<14}{'speedup':>9}{'s/round on 2xT4':>18}{'50 rounds':>12}{'params':>11}")
for k in ("patch", "seq", "chan"):
    r = out[k]
    print(f"{k:<14}{r['speedup']:8.2f}x{r['t4_s_per_round']:17.1f}"
          f"{r['t4_h_50_rounds']:11.2f} h{r['params']:11,}")
print(f"\nreference: full pipeline = {C50_S_PER_ROUND} s/round, "
      f"{50*C50_S_PER_ROUND/3600:.2f} h for 50 rounds (c50, measured on 2xT4)")
print(f"steps/round at batch 4,096: {STEPS_PER_ROUND:,}")
json.dump(out, open("/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench/dagsonly.json", "w"), indent=2)
