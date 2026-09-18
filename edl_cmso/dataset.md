# VeReMi NextGen - 16-class centralized dataset

Misbehaviour detection on V2X CAM logs, derived from the VeReMi NextGen source
release (Zenodo record `19665762`, DOI `10.5281/zenodo.19665762`). One row is one
received CAM message, seen from the receiving vehicle's side.

| split | files | rows | size | feature state |
|---|---:|---:|---:|---|
| `train/` | 15 | 43,045,415 | 7.3 GiB | imputed and standardised - use as is |
| `test/` | 4 | 10,761,343 | 1.7 GiB | imputed, **not standardised** - see below |

Neither split contains NaN or Inf.

Task: 16-class classification, `label` in `0..15` (`0` = benign). Binary is
`label > 0`. Class names are in `label_mapping.json`; per-class counts and all
integrity checks are in `statistics.md`.

## Two preprocessing steps, and why they sit where they do

**Imputation came first, before the split.** Every non-finite feature cell was
filled with `0.0` while the features were still one continuous stream, ahead of
the 80/20 cut. A constant fill needs no statistic, so its position in the order
cannot leak anything from test into train - and it means neither split ships
with NaN.

What was imputed: the 13 session features listed under `zero_imputed_features`
in `feature_schema.json` describe a change since the previous message, so they
have no value on the first message of a session - 23.9% of test rows. They are
`0` there. `f_first_in_session` is `1` on exactly those rows, so the imputation
stays visible to a model instead of being hidden. No other column was ever
non-finite, and there is no `Inf` anywhere.

**Standardisation came second, on train only.** That step *does* need statistics,
so `mean` and `std_used` in `scaler.json` were fitted on the 43,045,415 train
rows alone, with population variance (`ddof=0`), and applied to `train/` only.
`test/` is on the original scale. Standardise it yourself before scoring:

```python
import json, numpy as np, pyarrow.parquet as pq

scaler   = json.load(open("scaler.json"))["features"]
features = json.load(open("feature_schema.json"))["feature_columns"]
table    = pq.read_table("test/part-00000.parquet")

X = np.empty((table.num_rows, len(features)), dtype=np.float32)
for i, name in enumerate(features):
    stats = scaler[name]
    v = np.asarray(table[name].combine_chunks(), dtype=np.float64)
    X[:, i] = (v - stats["mean"]) / stats["std_used"]
y = np.asarray(table["label"].combine_chunks(), dtype=np.int8)
```

`load_test.py` in this directory does exactly that and streams batches:

```bash
python load_test.py                 # scan test, report class counts, assert 0 non-finite
python load_test.py --split train   # train is already scaled and is not transformed again
```

```python
from load_test import iter_batches, to_xy
for table in iter_batches("test"):
    X, y = to_xy(table)
```

**Never fit a scaler, imputer, or any other statistic on `test/`.** Do not apply
the transform to `train/` - those shards are already standardised.

## Sequence models

Group by `(attack_type, scenario, receiver_id, sender_alias)`, then sort by
`rcv_time_ns`. **Do not group by `receiver_id` alone**: vehicle IDs are reused
across simulation runs, and one receiver talks to many senders.

Every receiver's rows sit in a single file, in receive-time order, so a
sequential read never splits a session. Two things to plan for:

- **88% of test flows are a single message - but that is almost entirely one
  class.** `trafficCongestionSybil` changes pseudonym on every single message:
  2,393,335 test flows for exactly 2,393,335 rows, maximum flow length 1. Exclude
  it and the other 15 classes average 24 messages per flow (38 in train), with a
  per-class median between 7 and 62 and only 3.9% of flows one message long.
- **The all-zero session block is very close to a label for Sybil.** 100.00% of
  Sybil rows sit in a flow whose every row has all 13 `zero_imputed_features` at
  0 - that is, every row is a first-in-session row. The next highest class is
  `feignedBraking` at 0.20%, and `dosAttack` is 0.00%. If your Sybil F1 is
  near-perfect, say where it comes from, and report an ablation with the
  `session` feature group removed.
- Single-message flows are 22% of test rows. Filtering short flows out is not
  data cleaning here - it deletes most of a class.
- 161,265 test flows began before the train/test cut, so their first test row is
  a mid-session row, with real session values rather than the imputed zeros.

## Schema

87 columns. **A column is a model feature if and only if its name starts with
`f_`.** The rest are identifiers, labels, or raw context kept for analysis and
plotting - feeding them to a model leaks information.

- 66 `f_*` features (float32), grouped in `feature_schema.json` as `raw` 15,
  `position` 4, `geometry` 20, `session` 16, `rate` 5, `profile` 6.
- 1 label: `label` (int8).
- 2 partition columns materialised as real columns: `attack_type`, `scenario`.
- 9 identifiers: `receiver_id`, `sender_id`, `sender_alias`, `message_id`,
  `rcv_time_ns`, `send_time_ns`, `orig_split`, `source_run`, `t_rel_s`.
- 9 raw context columns: positions, headings, driver profiles,
  `snd_dist_road_edge`.

There is no `attacker` column. It was dropped on purpose: with this class design
it is fully determined by the label, so keeping it would leak the binary answer.

## How the splits were made

Split by **simulation time**, with a separate cut point per (class x scenario) -
64 cut points: `rcv_time_ns < t*` goes to train, `>=` goes to test. The train
share stays within `[0.799945, 0.800083]` across all 64 pairs. Nothing is split
randomly, so no message can leak from a session's future into its past.

Three things worth knowing about the class construction:

1. **Benign is taken once.** The 16 runs of each scenario share one underlying
   simulation, so benign rows repeat 16 times in the source. The benign class
   here is the `attacker = 0` rows of `trafficCongestionSybil`, which match the
   `normal` run value for value across all 26 payload columns in all 4 scenarios
   and additionally carry `distance_to_road_edge`.
2. **3,338,358 rows were dropped.** The source sets `attacker = 1` only when a
   deviation crosses a threshold, so some manipulated messages are unflagged.
   They were removed from both classes rather than poisoning benign or forcing
   the model to learn a near-indistinguishable difference.
3. **Attack classes keep only their attacker rows**, but features were computed
   over the complete message stream of each run before that filter, so session
   context is never broken by the filter.

`manifest.json` records the shuffle seed, the sequence key, and which source
shards went into each output file.

## Reporting results

State these alongside any number you publish: the split is by time, not by
vehicle; the benign class comes from an attack-free stream, which makes the
`rate` feature group unusually strong; 3.3M ambiguous rows were dropped; Sybil
activity is partly visible through `f_first_in_session`; `timeDelayAttack` is the
hardest class; class imbalance is 41:1; and 88% of test flows are one message
long - almost all of them Sybil - so say how you handled them.

## License

This bundle is derived work. The licence is the one attached to the source
release - `LICENSE.txt` on Zenodo record `19665762`. Cite the VeReMi NextGen
dataset, DOI `10.5281/zenodo.19665762`.
