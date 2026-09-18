"""Shared loader for the local copy of the dataset (identical to the Kaggle one).

train/ is already standardised — never touch it. test/ is raw and needs scaler.json.
A column is a model feature iff its name starts with f_ (66 of them).
"""
import json, numpy as np, pyarrow.parquet as pq
from pathlib import Path

ROOT = Path("/home/odixe/nckh/dataset")
RUN  = Path("/home/odixe/nckh/edl_cmso/papers/build2-paper-order/run1-wd1e4-diverged-r25/edl_cmso_v2_r50_b4096")
PKG  = RUN            # holds proj/
SCHEMA   = json.loads((ROOT / "feature_schema.json").read_text())
FEATURES = [c for c in SCHEMA["feature_columns"] if c.startswith("f_")]
assert len(FEATURES) == 66, len(FEATURES)
CFG  = json.loads((RUN / "config.json").read_text())
SEL  = np.array(CFG["sel_ch"])
PATCH_LEN = CFG["patch_len"]


def load_train(n_rows, seed=0, max_rows=1_500_000):
    """A sample spread over all 15 shards, read ROW GROUP BY ROW GROUP.

    This machine has 8 GB of RAM. An earlier version of this function called
    pq.read_table(f) on each shard, which materialises the whole 2.7M-row shard before
    sampling from it — that OOM-killed the box (exit 137). Each shard is 41 row groups of
    131,072 rows / 30 MB, so read only as many groups as the sample needs and never hold
    more than one at a time.

    Reading only the first shard is also not a sample of this dataset: rows are grouped by
    receiver within (attack_type, scenario), so shard 0 carries 5 of the 16 classes. Take
    from every shard.
    """
    if n_rows > max_rows:
        raise ValueError(f"n_rows={n_rows:,} over the {max_rows:,} cap for an 8 GB box; "
                         f"raise max_rows deliberately if you really mean it")
    files = sorted((ROOT / "train").glob("*.parquet"))
    per = -(-n_rows // len(files))                       # ceil
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for f in files:
        pf = pq.ParquetFile(f)
        ngroups = pf.metadata.num_row_groups
        # Rows inside a row group are contiguous, and the file is grouped by receiver
        # within (attack_type, scenario) — so one row group holds only a few classes.
        # Spread the shard's quota over many groups instead of filling from one.
        ngrp_use = min(ngroups, 16)
        order = rng.permutation(ngroups)[:ngrp_use]
        want, per_grp = per, -(-per // ngrp_use)
        for gi in order:
            if want <= 0:
                break
            t = pf.read_row_group(int(gi), columns=FEATURES + ["label"])
            take = min(want, per_grp, t.num_rows)
            if take < t.num_rows:
                sel = np.sort(rng.choice(t.num_rows, size=take, replace=False))
                t = t.take(sel)
            xs.append(np.column_stack([t[c].to_numpy(zero_copy_only=False)
                                       for c in FEATURES]).astype(np.float32))
            ys.append(t["label"].to_numpy().astype(np.int64))
            want -= take
            del t
    X = np.concatenate(xs); Y = np.concatenate(ys)
    del xs, ys
    r = rng.permutation(len(X))[:n_rows]
    return np.ascontiguousarray(X[r]), np.ascontiguousarray(Y[r])


def group_of_selected():
    """For each of the 133 selected fused channels: is it a raw Haar coefficient?"""
    return SEL < PATCH_LEN            # True = raw wavelet, False = ViT/GAT (LayerNorm'd)


def build(dev, load_init=True):
    import sys, torch
    sys.path.insert(0, str(PKG))
    import proj.model as M
    m = M.build_model(CFG, 66, CFG["sel_ch"]).to(dev)
    if load_init:
        M.load_extractor_state(m, torch.load(RUN / "extractor_init.pt",
                                             map_location="cpu"))
    return m, M
