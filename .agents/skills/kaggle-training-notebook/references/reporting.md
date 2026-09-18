# Write the training report

Read this reference only after outputs have been pulled, or when the user asks for an explicitly labelled interim report.

## Artifact roles

| File | Purpose |
|---|---|
| `notebook/<name>.executed.ipynb` | Evidence: code, formulas, logs, and rendered output together |
| `rebuild.md` | Ledger: deviations, decisions, and every run attempt |
| `report.md` | Synthesis: what was rebuilt, what was measured, and what it means |

Write `report.md` from verified local artifacts, not from memory or a transient Kaggle log. Link to the notebook and ledger instead of duplicating them.

## Required report content

1. Paper title, authors, DOI; dataset ref and version; kernel slug/version; run name and dates; actual hardware; completed/planned rounds; wall time and quota spent.
2. Included and omitted paper stages, with short reasons sourced from `rebuild.md`.
3. Dataset size, 16-class taxonomy, imbalance, temporal split, and the caveats in `dataset.md`. Keep the Sybil/session leakage, ambiguous-row removal, benign construction, and hard `timeDelayAttack` class beside the headline results.
4. Effective settings from the pulled `config.json`, plus `meta.json` and the selected-feature artifact. Do not infer settings from the editable source notebook.
5. Every completed round and all 10 metrics, using the exact names and order of `METRIC_KEYS` in [`metrics.md`](metrics.md), from deduplicated, round-sorted `metrics/history.csv`. Keep all columns even when values coincide. State `accuracy = precision_micro = recall_micro = recall_weighted = f1_micro` next to the table. This applies to both per-build reports and the root aggregate `report.md`; show every round for each included run, with invalid rounds explicitly labelled. A chart, linked CSV, selected rounds, or a three-metric summary does not fulfill this requirement.
6. Convergence, final confusion, and per-class figures generated from pulled artifacts.
7. Final-round per-class analysis, emphasizing support and the classes identified as hard or leakage-prone.
8. Paper results in their own section and this rebuild's results in another. For this project they are not numerically comparable: binary versus 16-class, different datasets, splits, and metric conventions.
9. What the evidence supports and does not support: overfitting, distribution shift, leakage, missing ablations, incomplete rounds, or other limitations.
10. Reproduction pointers: kernel ref, dataset ref, `machine_shape`, push command, checkpoint location, executed notebook, and artifact paths.

## Reporting rules

- Headline the final completed round. A best test round is an after-the-fact observation, not a selected checkpoint; label it explicitly if shown.
- Never put this paper's published numbers and the VeReMi rebuild numbers in the same table.
- Trace metrics to `metrics/history.csv`, class results to `reports/round_XXX.txt` or the matching confusion matrix, settings to `config.json`, and taxonomy/preprocessing to `meta.json` plus dataset sidecars.
- Report disappointing curves and failures plainly. Do not select the best test round to make the result look stronger.
- Use relative links and image paths so the report renders from the repository checkout.
- If fewer than the planned rounds exist, label the report interim or incomplete; do not imply the run finished.

## Generate figures

When a report already has a generator, inspect and update that generator as well as its
rendered output so regeneration preserves the 10-metric contract. In this repository,
`make_tables.py --apply` owns the marked table section of root `report.md`, and
`make_figures.py` owns its figures; build 1 has its own `make_report.py` and `make_figures.py`.
Run them in `nckh` only when modifying those reports. Preserve their section markers and
relative paths. Keep paper references `§4.8/§4.9/§4.10` distinct from report references
`mục 2.1`, etc.; bulk renumbering must not alter paper citations.

Run the bundled deterministic script in `nckh`:

```bash
conda run -n nckh python \
  .agents/skills/kaggle-training-notebook/scripts/make_report_figures.py \
  --run-dir papers/<slug>/runs/<run_name>
```

By default it writes to `papers/<slug>/figures/` and creates `summary.json`. Inspect the images and summary before citing them. The script orders per-class bars by support rather than by score and distinguishes the final round from the post-hoc peak.

Embed images relative to `report.md`, for example:

```markdown
![Test metrics per round](figures/convergence.png)
```

Give each figure a caption that states what the reader should notice.

## Completion gate

Before handing off, verify that every cited round exists, each number matches its source file, figure paths resolve, the final/peak distinction is accurate, and the dataset caveats appear in the results narrative rather than only in a footnote.

Also verify the rendered Markdown contains all 10 metric names and values for every reported
round, not merely that the CSV has them. Compare JSON and CSV values before display rounding,
check duplicates agree before deduplication, and confirm round coverage against the run's
completed/planned count. Validate ranges, full-test support, and collapse identities as
specified in [`metrics.md`](metrics.md). Missing evidence must remain explicitly `N/A` with a
reason and an incomplete-results label; never fill a missing metric with zero. Historical
checkpoint/prediction deletions recorded in `CONTEXT.md` are intentional: recover metrics from
retained raw confusion counts when possible, and do not call those deletions pull failures.
