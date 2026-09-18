"""Experiment A — the prediction deviation 34 makes, tested on the real data.

If the raw Haar channels are the instability, then the stem weights that read them must
carry far more gradient per channel than the 130 LayerNorm'd ones. One forward/backward
on real rows settles it.
"""
import sys, torch, numpy as np
sys.path.insert(0, "/home/odixe/nckh/edl_cmso/papers/build2-paper-order/bench")
import realdata as R

dev = "cuda"
X, Y = R.load_train(200_000)
print(f"real train rows {X.shape}   |x| max {np.abs(X).max():.1f}   "
      f"labels {len(np.unique(Y))}")
is_wav = R.group_of_selected()
print(f"selected channels: {is_wav.sum()} raw-wavelet, {(~is_wav).sum()} ViT/GAT\n")

m, _ = R.build(dev)
crit = torch.nn.CrossEntropyLoss()
B = 4096
wav_g, oth_g, wav_a, oth_a = [], [], [], []
for it in range(6):
    xb = torch.from_numpy(X[it*B:(it+1)*B]).to(dev)
    yb = torch.from_numpy(Y[it*B:(it+1)*B]).to(dev)
    m.zero_grad(set_to_none=True)
    with torch.no_grad():
        F = m.fuse(xb).index_select(1, m.sel_ch)
        amp = F.abs().amax(dim=(0, 2)).cpu().numpy()
    crit(m(xb).float(), yb).backward()
    # stems[i].weight is (out_ch, in_ch=133, k); column j reads fused channel j
    g = torch.stack([s[0].weight.grad.abs().mean(dim=(0, 2)) for s in m.stems]).mean(0)
    g = g.cpu().numpy()
    wav_g.append(g[is_wav].mean());  oth_g.append(g[~is_wav].mean())
    wav_a.append(amp[is_wav].max()); oth_a.append(amp[~is_wav].max())

print(f"activation  max|value|   raw-wavelet {np.mean(wav_a):9.2f}   "
      f"ViT/GAT {np.mean(oth_a):7.2f}   ratio {np.mean(wav_a)/np.mean(oth_a):7.1f}x")
print(f"stem gradient per channel  raw-wavelet {np.mean(wav_g):.3e}   "
      f"ViT/GAT {np.mean(oth_g):.3e}   ratio {np.mean(wav_g)/np.mean(oth_g):7.1f}x")
