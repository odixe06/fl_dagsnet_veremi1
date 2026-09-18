"""Build a tiny parquet fixture with the real dataset's SHAPE (87 columns, hive client dirs,
already-standardized train, unstandardized test) so the whole FL loop can be exercised on a
4 GiB laptop GPU.

It is a fixture, not a sample of the real data: it proves the plumbing, the arithmetic and the
crash behaviour. Every timing and every score has to come from the real data on the T4s.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

META = json.loads((Path(__file__).resolve().parents[1] / "knowledge" / "meta.json").read_text())
SCALER = json.loads((Path(__file__).resolve().parents[1] / "knowledge" / "scaler.json").read_text())
FEATURES = META["feature_cols"]
CLASSES = META["class_names"]
EXTRA = ["receiver_id", "sender_id", "sender_alias", "message_id", "rcv_time_ns", "send_time_ns",
         "orig_split", "source_run", "t_rel_s", "label", "rcv_pos_x", "rcv_pos_y", "snd_pos_x",
         "snd_pos_y", "rcv_hed", "snd_hed", "rcv_profile", "snd_profile", "snd_dist_road_edge",
         "attack_type", "scenario"]


def _table(X: np.ndarray, y: np.ndarray) -> pa.Table:
    cols = {f: pa.array(X[:, i].astype(np.float32)) for i, f in enumerate(FEATURES)}
    n = len(y)
    cols.update({
        "receiver_id": pa.array(np.zeros(n, np.int32)), "sender_id": pa.array(np.zeros(n, np.int32)),
        "sender_alias": pa.array(np.zeros(n, np.int32)), "message_id": pa.array(np.zeros(n, np.int64)),
        "rcv_time_ns": pa.array(np.zeros(n, np.int64)), "send_time_ns": pa.array(np.zeros(n, np.int64)),
        "orig_split": pa.array(["train"] * n), "source_run": pa.array(["fixture"] * n),
        "t_rel_s": pa.array(np.zeros(n, np.float32)),
        "label": pa.array(y.astype(np.int8)),
        "rcv_pos_x": pa.array(np.zeros(n, np.float32)), "rcv_pos_y": pa.array(np.zeros(n, np.float32)),
        "snd_pos_x": pa.array(np.zeros(n, np.float32)), "snd_pos_y": pa.array(np.zeros(n, np.float32)),
        "rcv_hed": pa.array(np.zeros(n, np.float32)), "snd_hed": pa.array(np.zeros(n, np.float32)),
        "rcv_profile": pa.array(["normal"] * n), "snd_profile": pa.array(["normal"] * n),
        "snd_dist_road_edge": pa.array(np.zeros(n, np.float32)),
        "attack_type": pa.array([CLASSES[int(v)] for v in y]),
        "scenario": pa.array(["highway_7"] * n),
    })
    return pa.table({c: cols[c] for c in FEATURES + EXTRA})


def build(root: Path, n_clients: int = 4, rows_per_client: int = 6000,
          test_rows: int = 4000, seed: int = 0, alpha: float = 0.5) -> dict:
    """Learnable non-IID fixture: each class is a distinct Gaussian blob in the 66-dim space,
    so a real model reaches a real score and a broken loop is visible as chance accuracy."""
    rng = np.random.default_rng(seed)
    K = len(CLASSES)
    centers = rng.normal(0, 2.0, size=(K, len(FEATURES))).astype(np.float32)

    root = Path(root)
    (root / "train").mkdir(parents=True, exist_ok=True)
    (root / "test").mkdir(parents=True, exist_ok=True)
    props = rng.dirichlet(np.full(K, alpha), size=n_clients)

    counts = np.zeros((n_clients, K), dtype=np.int64)
    for cid in range(n_clients):
        y = rng.choice(K, size=rows_per_client, p=props[cid]).astype(np.int8)
        counts[cid] = np.bincount(y, minlength=K)
        X = centers[y] + rng.normal(0, 1.0, size=(rows_per_client, len(FEATURES))).astype(np.float32)
        X = (X - X.mean(0)) / (X.std(0) + 1e-8)          # train ships ALREADY standardized
        d = root / "train" / f"client_id={cid:03d}"
        d.mkdir(parents=True, exist_ok=True)
        half = rows_per_client // 2                       # two shards, like the real partition
        pq.write_table(_table(X[:half], y[:half]), d / "part-00000.parquet")
        pq.write_table(_table(X[half:], y[half:]), d / "part-00001.parquet")

    yt = rng.choice(K, size=test_rows).astype(np.int8)
    Xt = centers[yt] + rng.normal(0, 1.0, size=(test_rows, len(FEATURES))).astype(np.float32)
    mu = np.array([SCALER["features"][f]["mean"] for f in FEATURES], np.float32)
    sd = np.array([SCALER["features"][f]["std_used"] for f in FEATURES], np.float32)
    Xt_raw = Xt * sd + mu                                 # test ships UNSTANDARDIZED
    j = FEATURES.index("f_snd_spd")
    Xt_raw[:, j] = rng.normal(7.45, 1.0, size=test_rows)  # keep the audit's sentinel in range
    pq.write_table(_table(Xt_raw, yt), root / "test" / "part-00000.parquet")

    # Sidecars with the SAME names and keys the real Kaggle datasets carry, so the notebook's
    # sentinel resolution and cross-checks are exercised rather than bypassed.
    (root / "scaler.json").write_text(json.dumps(SCALER))
    (root / "label_mapping.json").write_text(json.dumps({
        "classes": CLASSES,
        "class_to_label": {c: i for i, c in enumerate(CLASSES)},
        "label_to_class": {str(i): c for i, c in enumerate(CLASSES)}}))
    (root / "feature_schema.json").write_text(json.dumps(
        {"feature_columns": FEATURES, "n_features": len(FEATURES)}))
    (root / "client_stats.json").write_text(json.dumps({
        "clients": [{"client_id": c, "rows": rows_per_client,
                     "class_counts": {CLASSES[j]: int(counts[c, j]) for j in range(K)},
                     "fedavg_weight": 1.0 / n_clients} for c in range(n_clients)],
        "totals": {"clients": n_clients, "rows": rows_per_client * n_clients}}))

    info = {"n_clients": n_clients, "rows_per_client": rows_per_client, "test_rows": test_rows,
            "total_rows": rows_per_client * n_clients,
            "class_counts": counts.tolist(), "classes_present_min": int((counts > 0).sum(1).min())}
    (root / "fixture.json").write_text(json.dumps(info, indent=2))
    return info


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--test-rows", type=int, default=4000)
    args = ap.parse_args()
    print(json.dumps(build(Path(args.root), args.clients, args.rows, args.test_rows), indent=2))
