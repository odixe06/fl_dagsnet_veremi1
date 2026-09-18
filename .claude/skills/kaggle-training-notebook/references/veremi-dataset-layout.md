# VeReMi NextGen on Kaggle

Use this reference only for dataset `odixe0502/veremi-nextgen2026-centralized` in this repository.

## Verified identity and layout

Kaggle MCP reports dataset version 1 as `Ready`, with ref:

```text
odixe0502/veremi-nextgen2026-centralized
```

The dataset API exposes these relative paths:

```text
upload/
  README.md
  feature_schema.json
  label_mapping.json
  load_test.py
  manifest.json
  scaler.json
  statistics.json
  statistics.md
  train/part-*.parquet
  train/part-*.stats.json
  test/part-*.parquet
  test/part-*.stats.json
```

For a CLI-pushed notebook using this ref in `dataset_sources`, the runtime mount was measured as:

```text
/kaggle/input/datasets/odixe0502/veremi-nextgen2026-centralized/upload/
```

Therefore the current exact paths are:

```text
/kaggle/input/datasets/odixe0502/veremi-nextgen2026-centralized/upload/train/
/kaggle/input/datasets/odixe0502/veremi-nextgen2026-centralized/upload/test/
/kaggle/input/datasets/odixe0502/veremi-nextgen2026-centralized/upload/feature_schema.json
/kaggle/input/datasets/odixe0502/veremi-nextgen2026-centralized/upload/label_mapping.json
/kaggle/input/datasets/odixe0502/veremi-nextgen2026-centralized/upload/scaler.json
```

This differs from the shorter `/kaggle/input/<dataset-slug>/` path commonly shown for datasets attached through the UI. Do not rewrite the measured path into the UI form.

## Resolve safely

Prefer the exact measured root, but verify its sentinels and retain a diagnostic fallback in case Kaggle changes the mount layout:

```python
import os
from pathlib import Path

EXPECTED = Path(
    "/kaggle/input/datasets/odixe0502/"
    "veremi-nextgen2026-centralized/upload"
)

def is_veremi_root(path: Path) -> bool:
    return (
        (path / "feature_schema.json").is_file()
        and (path / "label_mapping.json").is_file()
        and (path / "scaler.json").is_file()
        and any((path / "train").glob("*.parquet"))
        and any((path / "test").glob("*.parquet"))
    )

def resolve_veremi_root() -> Path:
    if is_veremi_root(EXPECTED):
        return EXPECTED

    print("Expected mount missing; observed /kaggle/input tree:")
    for root, dirs, files in os.walk("/kaggle/input"):
        rel = Path(root).relative_to("/kaggle/input")
        if len(rel.parts) > 5:
            dirs[:] = []
            continue
        print("  " * len(rel.parts) + Path(root).name + "/" +
              (f" [{len(files)} files]" if files else ""))

    candidates = sorted(
        path.parent for path in Path("/kaggle/input").rglob("feature_schema.json")
        if is_veremi_root(path.parent)
    )
    assert len(candidates) == 1, (
        f"Expected exactly one VeReMi root, found {len(candidates)}: {candidates}"
    )
    return candidates[0]

DATA = resolve_veremi_root()
```

Requiring both split directories and all three sidecars prevents an unrelated `train/` directory from winning. Print observed state before raising; “dataset not attached” is a conclusion, not a useful diagnostic.

## Verified sidecar contracts

Download small sidecars locally before changing their parser:

```bash
conda run -n nckh kaggle datasets download \
  odixe0502/veremi-nextgen2026-centralized \
  -f upload/label_mapping.json -p <temp-dir> --unzip
```

Repeat for `upload/feature_schema.json` and `upload/scaler.json`.

Measured invariants:

- `label_mapping.json` contains `label_to_class`, `classes`, and `class_to_label`, each describing the same 16-class axis. Read one and assert the other two agree.
- `feature_schema.json` uses `groups`, not `feature_groups`. It declares 66 features: `raw=15`, `position=4`, `rate=5`, `session=16`, `geometry=20`, `profile=6`; 13 features are listed as zero-imputed.
- `scaler.json` contains 66 feature entries and `n_train_rows=43_045_415`; its standardization and imputation metadata must be reported rather than recomputed from test.

Treat a failed redundancy assertion as corrupted metadata and stop. Falling back to generated class names or silently omitting feature groups would invalidate per-class or per-group reporting.

## Materialise and audit in one decode pass

`/kaggle/temp` is wiped between sessions, so decoding the parquet once for audit and again for cached arrays repeats the dominant CPU work on every resume. Read row counts from parquet footers first, allocate the cache, then for each decoded float32 block:

1. accumulate label counts, NaN/Inf counts, ranges, and other audit statistics;
2. write the same block into the fp16 or target-format cache;
3. discard it before loading the next block.

Compute audit statistics before the fp16 cast, and assert final written row counts against the footers. This changes I/O scheduling, not the dataset or reported statistics.
