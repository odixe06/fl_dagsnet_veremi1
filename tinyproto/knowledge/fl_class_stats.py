"""n_{i,j} statistics per scenario -> data-driven mu candidates for APS."""
import json, numpy as np
from pathlib import Path
FL = Path("/home/odixe/nckh/dataset/fl_client/alpha05")
META = json.load(open("/home/odixe/nckh/tinyproto/knowledge/meta.json"))
CLASSES = META["class_names"]
out = {}
for n in (20, 50, 100):
    st = json.load(open(FL / f"{n}_client/client_stats.json"))
    M = np.array([[c["class_counts"][k] for k in CLASSES] for c in st["clients"]], np.int64)  # (M,16)
    present = M > 0
    # |N_j| = clients holding class j ; mean n_{i,j} over those clients
    Nj = present.sum(0)                                        # (16,)
    mean_nij = np.where(Nj > 0, M.sum(0) / np.maximum(Nj, 1), 0.0)
    nz = M[present]
    out[n] = {
        "clients": int(M.shape[0]),
        "K_i_min": int(present.sum(1).min()), "K_i_max": int(present.sum(1).max()),
        "K_i_mean": float(present.sum(1).mean()),
        "N_j_min": int(Nj.min()), "N_j_max": int(Nj.max()),
        "n_ij_nonzero_count": int(nz.size),
        "n_ij_min": int(nz.min()), "n_ij_median": float(np.median(nz)),
        "n_ij_mean": float(nz.mean()), "n_ij_max": int(nz.max()),
        "mean_nij_per_class_min": float(mean_nij.min()),
        "mean_nij_per_class_max": float(mean_nij.max()),
        "mean_nij_per_class_mean": float(mean_nij.mean()),
        # global-prototype scale factor:  chatG_j ~ c_j * mean_{i in N_j} n_{i,j}
        "mu_unit_scale": float(1.0 / mean_nij[mean_nij > 0].mean()),
        "mu_over_median": float(1.0 / np.median(nz)),
        # paper ratio: mu_paper ~ 0.12 / mean n_ij  (see derivation note)
        "mu_paper_ratio_0125": float(0.125 / mean_nij[mean_nij > 0].mean()),
        "per_class_mean_nij": {CLASSES[j]: float(mean_nij[j]) for j in range(16)},
        "per_class_Nj": {CLASSES[j]: int(Nj[j]) for j in range(16)},
        # communication cost per round: sum_i (K_i + K) * s
        "sum_Ki_plus_K": int((present.sum(1) + 16).sum()),
    }
    print(f"== {n} clients ==")
    print(f"  K_i (classes/client): min {out[n]['K_i_min']} max {out[n]['K_i_max']} mean {out[n]['K_i_mean']:.2f}")
    print(f"  |N_j|: min {out[n]['N_j_min']} max {out[n]['N_j_max']}")
    print(f"  n_ij (nonzero): min {out[n]['n_ij_min']:,} median {out[n]['n_ij_median']:,.0f} "
          f"mean {out[n]['n_ij_mean']:,.0f} max {out[n]['n_ij_max']:,}")
    print(f"  mean_{{i in N_j}} n_ij per class: min {out[n]['mean_nij_per_class_min']:,.0f} "
          f"max {out[n]['mean_nij_per_class_max']:,.0f} mean {out[n]['mean_nij_per_class_mean']:,.0f}")
    print(f"  mu candidates: unit-scale {out[n]['mu_unit_scale']:.3e} | "
          f"1/median {out[n]['mu_over_median']:.3e} | paper-ratio(0.125) {out[n]['mu_paper_ratio_0125']:.3e}")
    print(f"  sum_i(K_i+K) = {out[n]['sum_Ki_plus_K']}  -> cost/round at s=50: "
          f"{out[n]['sum_Ki_plus_K']*50/1e6:.4f} M params; at d=256 (FedProto): "
          f"{out[n]['sum_Ki_plus_K']*256/1e6:.4f} M")
Path("/tmp/claude-1000/-home-odixe-nckh-tinyproto/96aae4cb-f1a2-4517-b930-e0e4545a4884/scratchpad/aps_mu.json").write_text(json.dumps(out, indent=2))
