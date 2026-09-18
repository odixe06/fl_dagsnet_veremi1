# The paper dossier

Every rebuild gets a folder in the project. It is written **before** the notebook, and it is the thing that survives after the Kaggle session is gone.

```
papers/<slug>/
  paper.md          the original work, described faithfully
  rebuild.md        what this build changed, and why
  notebook/         the .ipynb actually pushed
  runs/<run_name>/  outputs pulled back from Kaggle
    checkpoints/  metrics/  preds/  confusion/  reports/  logs/
```

`<slug>` is `firstauthor-year-keyword`, lowercase and hyphenated — `zhou-2023-edl-ids`. `run_name` matches `CFG.run_name` so a folder maps to exactly one training run.

Two documents, because they answer two different questions. `paper.md` answers *what did the authors do* — and stays fixed once written. `rebuild.md` answers *what did we do differently* — and grows as the work does. Collapsing them into one file loses the boundary between the source of truth and the delta, which is precisely the boundary a reviewer asks about.

## `paper.md`

Faithful to the source. No opinions about our dataset here.

```markdown
# <Title>

**Citation** · Authors, Venue Year, DOI/arXiv
**Link** · <url>       **Code** · <official repo or "none released">

## Problem
What task, what inputs, what outputs. 2-4 sentences.

## Method
Architecture layer by layer, in the paper's own notation. Every equation that
defines the method, in LaTeX, with the paper's equation numbers:

$$\mathcal{L} = ...   \tag{3}$$

## Training setup as published
| Item | Paper |
|---|---|
| Dataset | |
| Classes | |
| Split | |
| Preprocessing | |
| Optimizer / LR / schedule | |
| Batch size | |
| Epochs (or rounds) | |
| Loss | |
| Hardware | |
| Seeds / repeats | |

## Reported results
The paper's own table, verbatim — metric names exactly as printed.

| Metric | Reported |
|---|---|

## Gaps in the paper
Anything unstated that a reimplementation must decide: init scheme, whether
normalization statistics come from train only, tie-breaking, unstated
hyperparameters. This section is what makes the deviations honest.
```

Pull these from the actual PDF or the authors' code. When a value is not stated, write `unstated` — never a plausible-looking number, because that number will later be read as the paper's.

## Equations that render

The dossier is read through a Markdown viewer with KaTeX, which is stricter than
a LaTeX compiler: on a construct it does not accept it drops the whole block and
prints `ParseError: KaTeX parse error` in its place. The file still looks fine in
an editor, so the breakage is only visible to the reader.

**One `\tag` per `$$…$$` block.** A second one fails the block with
`Multiple \tag`. Two equations printed on one line in the PDF still need two
blocks here — the numbers stay intact:

```markdown
<!-- breaks: Multiple \tag -->
$$F_l = H_l(\dots) \tag{38} \qquad F_{\text{DenseNet}} = F_L \tag{39}$$

<!-- renders -->
$$F_l = H_l(\dots) \tag{38}$$

$$F_{\text{DenseNet}} = F_L \tag{39}$$
```

The same applies to `\qquad`-joined equations, `cases`/`aligned` bodies, and any
block where two paper equation numbers were merged to save vertical space. Never
collapse two numbers into one tag (`\tag{38,39}`) to keep them on one line — the
tag has to match the number a reviewer looks up.

Three more that break silently in the same way: `\label`, `\eqref` and
`\nonumber` are unsupported (the `\tag` *is* the number); a `$` inside `\text{}`
reopens math mode; `$$` must open and close on the same line or wrap the body
across whole lines, never interleaved with prose.

Check the dossier after writing or editing it — this is fast and catches all of
the above:

```bash
grep -n '\\tag{.*\\tag{' papers/<slug>/*.md            # >1 \tag in one block
grep -o '\$\$' papers/<slug>/paper.md | wc -l          # must be even
grep -n '\\label\|\\eqref\|\\nonumber' papers/<slug>/*.md
```

## `rebuild.md`

The delta, and everything downstream of it.

```markdown
# Rebuild — <slug>

**Paper** · [paper.md](paper.md)   **Run** · `<run_name>`
**Notebook** · [notebook/<name>.ipynb](notebook/)   **Kernel** · <user>/<slug>
**Status** · planned | running | complete | abandoned

## Our dataset
Name, source, rows in train / test, class count, per-class counts, imbalance
ratio, split policy (and why: time-based, flow-grouped, given). Link the audit
output from Step 3.

## Deviations from the paper
Every difference, with a reason. This is the document's core.

| # | Paper | This build | Why |
|---|---|---|---|
| 1 | 5 classes (NSL-KDD) | 15 classes (our capture) | our label taxonomy |
| 2 | batch 128, 1 GPU | 256/GPU x 2 T4 = 512 | 2x T4, same LR scaled |
| 3 | `unstated` init | Kaiming normal | paper does not state it |

Mark each as **forced** (our data leaves no choice), **hardware** (2x T4),
or **judgement** (we chose; the paper allowed either). Judgement rows are the
ones a reviewer will ask about, so give them a full sentence.

## Round definition
Which of the three shapes this task uses, and what one round means here.

## Config
The `CFG` block as pushed, verbatim.

## Results
Filled after each pull. Always report this rebuild in its own table, with all 10 exact
metric names from [`metrics.md`](metrics.md) for each reported round. Keep equal-valued
metrics; report missing evidence explicitly rather than reducing the schema.

| Round | Metric | Value |
|---|---|---|

When task, data, class semantics, and metric definitions are commensurable, an additional comparison table may be useful. Otherwise keep the paper's published table separate and state why no numerical delta is valid.

## Run history
| Attempt | Date | Rounds done | Outcome | Fix applied |
|---|---|---|---|---|
| 1 | 2026-08-31 | 3/10 | OOM at round 4 | batch 512 -> 256 |

## Open questions
Anything still unresolved, addressed to the user.
```

The **Run history** table is what makes a resumed run legible weeks later: it records that rounds 0–2 came from attempt 1 and rounds 3–9 from attempt 2, which is otherwise invisible once `history.csv` is merged.

## When to write

- `paper.md` — at Step 2, before any code. If the paper is not available, say so and ask for it; do not reconstruct a method from memory.
- `rebuild.md` — skeleton at Step 2, dataset section at Step 3, config at Step 4, results and run history at every pull.
- Update `rebuild.md` on **every** babysit fix. A hyperparameter changed to get past a crash is a deviation like any other, and it is the one most likely to be forgotten.
