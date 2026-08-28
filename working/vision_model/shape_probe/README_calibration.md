# Calibration and linear probe — *why* the clinical probe fails

Companion to [`README.md`](README.md), which covers building the shape images and
running the generative probe. This document covers the two follow-up scripts:

```
score_logprobs.py   GPU. Forced-choice label log-probs, and pooled hidden states.
calibrate.py        CPU. Re-thresholds those log-probs; probes those hidden states.
```

Nothing in `build_shapes.py`, `run_shape_probe.py` or `medgemma_pilot/` is
modified. `score_logprobs.py` imports `run_shape_probe.build_messages`, so the
prompt, the definitions block and the few-shot turns are byte-identical to the
generative run by construction.

## The observation these scripts exist to explain

The `clinical` ladder on MRI backgrounds, difficulties 0.65 and 1.0, 202 images
per class:

| run | acc | irregular | lobulated | round_oval | times it predicted `irregular` |
| :--- | ---: | ---: | ---: | ---: | ---: |
| zero-shot | 0.436 | 0.054 | 0.574 | 0.673 | 22 / 606 |
| k=1 | 0.490 | 0.153 | 0.470 | 0.847 | 43 / 606 |
| k=5 | 0.548 | **0.020** | 0.901 | 0.723 | **10 / 606** |

Three things follow, and none of them are visible in the accuracy column.

**The k=5 gain is a prior shift, not learning.** It predicted `lobulated` 428
times out of 606 — becoming a two-class classifier that dumps everything
non-round into `lobulated`. Because the test set is balanced, that *buys*
accuracy. The best-scoring configuration is the least useful one. Plain accuracy
cannot see this; per-class recall and the **prediction marginals** can, which is
why `calibrate.py` prints both on every line.

**The model has a 1-D roundness axis, not a 3-class concept.** At k=5,
`P(pred = round_oval | true)` is 0.025 / 0.084 / 0.72 across
irregular / lobulated / round_oval — a clean monotone ordering. `irregular` and
`lobulated` both sit at the not-round end and are collapsed. Essentially all
error mass is on that one boundary.

**But `irregular` precision is well above chance:** 11/22 = 0.50 zero-shot,
31/43 = **0.72** at k=1 (small denominators, wide CIs, but 0.72 against a 0.33
chance rate). When the model commits to `irregular` it is usually right — it just
almost never commits. That is not "cannot perceive irregularity"; that is a
usable signal behind a badly placed decision threshold.

Hard labels cannot take that further. Both scripts below exist to.

## `score_logprobs.py --mode score` — read the answer as a distribution

`run_shape_probe.py --mode infer` generates text and parses one winner out of it,
which records 0.34/0.33/0.33 identically to 0.98/0.01/0.01. Instead, fix the
answer prefix `{"prediction": "` and score each label string continuing from
there. Each image yields three log-probs. No autoregressive decoding, so this is
*faster* per image than `--mode infer`, and `PARSE_FAILED` rows cannot occur.

```bash
# GPU, medgemma_pilot environment. Sharding flags match run_shape_probe.py.
python score_logprobs.py --mode score --model-id /models/medgemma-1.5-4b-it \
    --metadata /results/shape_probe/clinical/shape_metadata.csv \
    --out /results/shape_probe/clinical/logprobs_0shot.csv
```

Output is one row per image with a `logp_<label>` column per class, plus the
metadata columns needed to slice by difficulty, background and `case_id`.

Implementation notes worth knowing:

- The prefix and the label are tokenised **jointly**, and the label's tokens are
  located by longest shared prefix with the tokenised prefix alone. Tokenising
  them separately and concatenating is wrong — the prefix ends in `"` and every
  label starts with a letter, so a SentencePiece-style tokenizer merges across
  the boundary and you would score ids the model would never emit.
- Prompts are left-padded (mandatory for this model), so every row's prompt ends
  at the same index and the continuation always starts there. Continuations
  differ in length and are right-padded — harmless under causal attention.
- `--batch-size` counts *images*; the batch is expanded by the number of labels
  internally, so peak memory is `batch_size × n_labels` rows.

> **`score` is not the same inference path as `--mode infer`.** MedGemma 1.5
> thinks before answering, and forced-choice scoring straight after the
> generation prompt skips that. Pass
> `--thinking-from /results/.../probe_results.csv` to replay the `thinking`
> column an earlier generative run already wrote, so the label is scored in the
> context the real run had. Run it both ways: the gap is what the chain of
> thought is worth on this task.

## `calibrate.py --logprobs` — is it a threshold problem?

```bash
# CPU, ROOT uv environment (needs scikit-learn, which medgemma_pilot
# deliberately does not depend on)
uv run python working/vision_model/shape_probe/calibrate.py \
    --logprobs /results/shape_probe/clinical/logprobs_0shot.csv --geometry
```

Prints the raw argmax result, then re-thresholds **the same log-probs** three
ways, in increasing strength:

| method | params | uses labels? | what it does |
| :--- | ---: | :--- | :--- |
| prior correction | 0 | no | subtracts `log mean_x p_y(x)` — the model's content-independent label prior (Zhao et al. 2021) |
| marginal matching | 0 | no | fits a per-class bias so the *predicted* marginal is uniform, which the balanced build justifies |
| matrix scaling | K²+K = 12 | yes | multinomial logistic regression on the K log-probs; out-of-fold, `GroupKFold` on `case_id` |

The first two use no labels at all — they are properties of the output
distribution, not of the answers — so they are legitimate to apply to the full
test set. A per-class bias also absorbs the fact that `round_oval`, `lobulated`
and `irregular` tokenise to different numbers of tokens (a length bias on
sequence log-prob), so no separate length normalisation is needed.

Every block reports accuracy, **balanced accuracy**, per-class recall *and*
precision, the prediction marginal, and the confusion matrix. Read balanced
accuracy and the marginal, not accuracy — that is precisely how the k=5 collapse
disguised itself as progress.

### Read the pairwise AUC first

It is printed **before** any calibration, because it is the one number a bad
threshold cannot spoil. It asks only whether the score *ranks* true irregulars
above true lobulateds:

| `lobulated vs irregular` AUC | reading |
| :--- | :--- |
| ≈ 0.5 | no signal. Re-thresholding cannot help, and neither can a bias-only fine-tune. The problem is upstream — see the input-side check below. |
| ≈ 0.65 | weak signal. Calibration recovers some of it; expect a ceiling. |
| ≈ 0.8 with 0.02 recall | the entire deficit is the threshold. Calibration should move it a lot, and LoRA should pay off cheaply. |

`--logprobs` also prints balanced accuracy by difficulty, raw vs calibrated, so
you can see whether calibration helps uniformly or only at d=1.0.

## `calibrate.py --embeddings` — is the signal in the representation?

Calibration can only exploit information the representation already carries.
`--mode embed` dumps pooled hidden states so an external linear classifier can
measure that directly, independent of the model's ability to *say* the answer.

```bash
python score_logprobs.py --mode embed --model-id /models/medgemma-1.5-4b-it \
    --metadata /results/shape_probe/clinical/shape_metadata.csv \
    --layers -1,mid \
    --out /results/shape_probe/clinical/embeddings.npz     # + .meta.csv beside it

uv run python working/vision_model/shape_probe/calibrate.py \
    --embeddings /results/shape_probe/clinical/embeddings.npz
```

Two poolings per requested layer, separating two different failures:

| pooling | what it says |
| :--- | :--- |
| `image_mean` | mean over the query image's token positions — did the vision pathway **encode** it |
| `last` | the final prompt position — did it **reach** the point the LM head decodes from |

`image_mean` high and `last` low means the information is present but not routed
to the decision; that is the most favourable possible setting for LoRA.

Probes are out-of-fold with `GroupKFold` on `case_id`, so the several difficulty
levels built from one source crop never straddle the split. Without that
grouping the numbers are inflated by leakage and you will not notice.

Every class **pair** is probed separately as well as all three together, because
a 3-class number can look mediocre for the wrong reason — an easy `round_oval`
propping up a collapsed pair. **The `lobulated vs irregular` binary is the number
that matters.**

| binary probe | reading | what to do |
| :--- | :--- | :--- |
| ≥ 0.80 | features present, readout broken | calibrate; LoRA should pay off cheaply and with little data |
| 0.55–0.65 | partially present | LoRA helps but caps out; fix inputs too |
| ≈ 0.50 | absent at this resolution | **do not fine-tune yet** — crop tighter / upscale first |

`--mode embed` is zero-shot only, and refuses `--num-few-shot`: few-shot puts
several images in context, so pooling "the image tokens" would mix exemplars into
the query vector. The probe measures the representation, not the prompt, so
zero-shot is the right condition anyway.

## `--geometry` — is the generator's distinction even learnable?

A sanity check, not a result: if the generator's own parameters do not separate
the classes, no model result about them means anything.

It **drops any parameter absent for some class** before fitting. This matters —
`shape_params` records `lobe_k` only for lobulated and `jag_amp` only for
irregular, so keeping them and filling gaps with 0 lets a classifier score 1.000
off which columns are merely *present*, without reading a single value. The check
would be vacuous.

If nothing survives the drop, the script says so rather than printing a
meaningless 1.000. The honest version then needs the descriptors that *are*
defined for all three classes, recomputed from `r(θ)` — `dom_k`, `kurt`,
peak-to-peak, the table in [`README.md`](README.md). Worth doing: if the masks
separate at ~0.95 but `image_mean` embeddings separate at ~0.55, the
discriminating contour detail is being destroyed between the 128×128 crop and the
vision tokenizer, and the fix is input-side rather than weight-side.

## On the cluster

```bash
sbatch jobs/run_score_logprobs.sh      # edit the paths / toggles at the top
```

One job does everything: both GPU passes sharded across the allocated GPUs, then
both `calibrate.py` blocks on CPU. Toggle the passes independently —

```bash
RUN_SCORE=1      # forced-choice log-probs, no thinking block
RUN_SCORE_COT=1  # same, scored after replaying $ZEROSHOT's thinking column
RUN_EMBED=1      # pooled hidden states for the linear probe
```

`RUN_SCORE_COT` auto-disables itself with a note if the generative results it
needs aren't there yet, so you can submit this before
`jobs/run_shape_probe.sh` has ever run.

Batch sizes are deliberately below the generative job's 64, for two different
reasons:

- **score** — the batch is expanded by the number of labels internally (3 rows
  per image on a default clinical build), so peak memory is
  `BATCH_SCORE × n_labels` rows.
- **embed** — `output_hidden_states` materialises *every* layer's activations for
  the whole sequence at once, which on the 27B is far more memory than the
  forward pass itself. This is the binding constraint in the job; raise
  `BATCH_EMBED` only after reading the `img/s` line.

Same GPU/sharding caveat as the other launchers: keep `--nodes=1` and set
`--gpus-per-node` equal to `NUM_SHARDS`. With `--nodes=2` the second node sits
idle while `CUDA_VISIBLE_DEVICES=1` selects a device that doesn't exist, and that
case does *not* crash — `device_map="auto"` quietly places the model on CPU and
the shard appears to hang.

> **Not resume-safe**, unlike `run_shape_probe.py`: `score_logprobs.py`
> truncates its output rather than appending, so re-running a pass redoes it in
> full. Deliberate — these passes are ~1 forward per image so a restart is
> cheap, and appending would risk silently mixing two configurations (with and
> without a thinking prefix) into one CSV.

## Order to run things, and what each outcome means

1. **`--mode score`, zero-shot, plus `--logprobs`.** Cheapest and most
   informative. Gives the AUC verdict and the calibrated baseline that any
   fine-tune must beat — otherwise you will credit LoRA for a gain that was
   really just bias correction.
2. **`--mode score --thinking-from ...`.** Confirms whether the chain of thought
   is carrying weight, i.e. whether (1) understates the generative path.
3. **`--mode embed`, plus `--embeddings`.** The `lobulated vs irregular` binary
   decides whether the next move is LoRA or the input pipeline.
4. **`--geometry`.** Cheap; run it alongside (1).

Only after those does a LoRA run answer a well-posed question. Frame that run as
a **data-efficiency curve** — balanced accuracy vs. N, holding out difficulty
level and MRI background — because the transferable asset for the eventual
radiologist-labelled phase is the recipe and the N-vs-accuracy curve, not the
weights: human `irregular` will not be this generator's `irregular`.
