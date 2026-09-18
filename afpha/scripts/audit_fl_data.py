"""Read-only, bounded-memory audit of AFPHA train partitions and the fixed test.

Run with conda run -n nckh python scripts/audit_fl_data.py [--full].
The default samples the first 64 rows of EVERY file; --full scans every feature/label.
Duplicate rows and cross-split flow overlap require a separate provenance audit.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def read_json(path):
    return json.loads(path.read_text())


def audit_split(files, features, full, scaler=None):
    counts = np.zeros(16, dtype=np.int64)
    low = np.full(len(features), np.inf)
    high = np.full(len(features), -np.inf)
    sums = np.zeros(len(features))
    squares = np.zeros(len(features))
    invalid = np.zeros(len(features), dtype=np.int64)
    rows = seen = 0
    schemas = {}
    for path in files:
        parquet = pq.ParquetFile(path)
        rows += parquet.metadata.num_rows
        schema = parquet.schema_arrow
        actual = [name for name in schema.names if name.startswith("f_")]
        if actual != features:
            raise ValueError(f"Feature order/schema mismatch: {path}")
        signature = str(schema)
        schemas[signature] = schemas.get(signature, 0) + 1
        for batch in parquet.iter_batches(batch_size=8192 if full else 64,
                                          columns=features + ["label"],
                                          use_threads=False):
            x = np.column_stack([batch.column(i).to_numpy()
                                 for i in range(len(features))]).astype(np.float64)
            y = batch.column(len(features)).to_numpy()
            if not np.issubdtype(y.dtype, np.integer) or np.any((y < 0) | (y >= 16)):
                raise ValueError(f"Invalid labels: {path}")
            bad = ~np.isfinite(x)
            invalid += bad.sum(axis=0)
            x[bad] = 0
            if scaler:
                x = (x - scaler[0]) / scaler[1]
            low = np.minimum(low, x.min(axis=0))
            high = np.maximum(high, x.max(axis=0))
            sums += x.sum(axis=0)
            squares += np.square(x).sum(axis=0)
            counts += np.bincount(y, minlength=16)
            seen += len(y)
            if not full:
                break
    if not seen:
        raise ValueError("Empty split")
    mean = sums / seen
    return {
        "files": len(files), "rows_from_parquet_metadata": rows, "rows_scanned": seen,
        "coverage": "full_features_and_labels" if full else "first_64_rows_per_file",
        "label_counts_scanned": counts.tolist(),
        "nonfinite_cells_scanned": int(invalid.sum()),
        "schema_variants": schemas,
        "features_in_model_scale": {
            name: {"min": float(low[i]), "max": float(high[i]), "mean": float(mean[i]),
                   "std": float(np.sqrt(max(0, squares[i] / seen - mean[i] ** 2))),
                   "invalid": int(invalid[i])}
            for i, name in enumerate(features)
        },
        "constant_features_scanned": [name for i, name in enumerate(features)
                                      if low[i] == high[i]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fl-root", type=Path,
                        default=Path("/home/odixe/nckh/dataset/fl_client/alpha05"))
    parser.add_argument("--centralized-root", type=Path,
                        default=Path("/home/odixe/nckh/dataset/centralized"))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path("papers/khan-2025-afpha/audit.json"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    features = read_json(args.centralized_root / "feature_schema.json")["feature_columns"]
    labels = read_json(args.centralized_root / "label_mapping.json")["classes"]
    model_meta = read_json(root / "architecture/meta.json")
    if features != model_meta["feature_cols"] or labels != model_meta["class_names"]:
        raise ValueError("Dataset and architecture metadata disagree")
    scaler = read_json(args.centralized_root / "scaler.json")
    if scaler["features"] != read_json(root / "architecture/scaler.json")["features"]:
        raise ValueError("Dataset and architecture scaler disagree")
    means = np.array([scaler["features"][f]["mean"] for f in features])
    stds = np.array([scaler["features"][f]["std_used"] for f in features])
    if len(features) != 66 or len(labels) != 16 or not np.all(stds > 0):
        raise ValueError("Invalid feature/label/scaler contract")
    result = {
        "feature_cols": features, "class_names": labels,
        "scaler_sha256": hashlib.sha256((args.centralized_root / "scaler.json").read_bytes()).hexdigest(),
        "normalization": "train already standardized; test transformed with supplied train scaler",
        "duplicate_rows_and_cross_split_flow_overlap": "not measured by this script",
        "identifier_policy": "only the frozen 66 f_* columns enter the model",
        "scenarios": {},
    }
    global_counts = None
    for clients in (20, 50, 100):
        directory = args.fl_root / f"{clients}_client"
        stats = read_json(directory / "client_stats.json")["clients"]
        config = read_json(directory / "manifest/partition_config.json")
        dirs = sorted((directory / "train").glob("client_id=*"))
        if len(dirs) != clients or config["n_clients"] != clients:
            raise ValueError(f"Wrong client count: {directory}")
        files = sorted((directory / "train").rglob("*.parquet"))
        expected_counts = np.zeros(16, dtype=np.int64)
        for entry in stats:
            local_files = sorted((directory / "train" /
                                  f"client_id={entry['client_id']:03d}").glob("*.parquet"))
            actual_rows = sum(pq.ParquetFile(f).metadata.num_rows for f in local_files)
            if actual_rows != entry["rows"] or len(local_files) != entry["files"]:
                raise ValueError(f"Client manifest mismatch: {entry['client_id']}")
            expected_counts += [entry["class_counts"].get(c, 0) for c in labels]
        report = audit_split(files, features, args.full)
        if report["rows_from_parquet_metadata"] != config["rows"]:
            raise ValueError("Partition total differs from manifest")
        if args.full and report["label_counts_scanned"] != expected_counts.tolist():
            raise ValueError("Scanned labels differ from client sidecars")
        if global_counts is not None and not np.array_equal(global_counts, expected_counts):
            raise ValueError("Scenario label totals differ")
        global_counts = expected_counts
        report["label_counts_from_sidecars"] = expected_counts.tolist()
        report["client_rows_min_max"] = [min(s["rows"] for s in stats),
                                          max(s["rows"] for s in stats)]
        result["scenarios"][str(clients)] = report
        print(json.dumps({"clients": clients, "files": len(files),
                          "rows": report["rows_from_parquet_metadata"],
                          "scanned": report["rows_scanned"],
                          "nonfinite": report["nonfinite_cells_scanned"]}), flush=True)
    test_files = sorted((args.centralized_root / "test").glob("*.parquet"))
    result["test"] = audit_split(test_files, features, args.full, (means, stds))
    if args.full:
        test_counts = np.array(result["test"]["label_counts_scanned"])
        if not np.array_equal(global_counts > 0, test_counts > 0):
            raise ValueError("Train union / test class coverage differs")
    if any(s["nonfinite_cells_scanned"] for s in result["scenarios"].values()) or result["test"]["nonfinite_cells_scanned"]:
        raise ValueError("Non-finite data found; resolve preprocessing before training")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "test_rows": result["test"]["rows_from_parquet_metadata"],
                      "test_scanned": result["test"]["rows_scanned"]}), flush=True)


if __name__ == "__main__":
    main()
