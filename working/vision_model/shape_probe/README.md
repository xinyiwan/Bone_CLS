# Shape probe — what can MedGemma actually see?

A pseudo-segmentation experiment. In the real pilot we feed MedGemma the lesion
crop with the radiologist's **red contour** burned in (`run_medgemma.py
--use-contour`) and *assume* the model uses it. This probe tests that directly:
keep everything the same — same crops, same centre, same red colour and 2 px
thickness, same decoding path — but replace the true contour with a synthetic
one, and ask only *what shape is drawn*.

There are **two ladders**, selected at build time with `--shape-set`.

## `icons` — perception (chance 25%)

`circle / square / triangle / star`. Answers *is the overlay visible at all*.

Every class here is a polygon with a distinct **vertex count** (3, 4, ∞,
10-with-spikes), so it is solvable by corner-counting — a categorical cue real
tumour margins do not have. Near-perfect accuracy therefore means the overlay is
legible; it does **not** mean margin shape is legible. That is what the second
ladder is for.

## `clinical` — discrimination (chance 20%)

The five values of the `shape` feature in
`../medgemma_pilot/feature_prompts.yaml`: `round_oval / lobulated / geographic /
irregular / exophytic`. Answers *can it tell 5 smooth lobes from 20 jagged
spikes when both are "bumpy"*.

All five come out of **one radial equation** (`shapes.py`), differing only in
parameters:

```
r(θ) = R · [ 1 + a·sin(kθ+φ)      lobulated   k = 4-7 smooth bulges
               − d·dent(θ)        geographic  one broad concave bite
               + b·noise(θ)       irregular   7 random harmonics, k = 7-22
               + c·bump(θ)        exophytic   one flat-topped stalk (mushroom)
               + ε·surface(θ) ]   all classes tiny shared texture
```

so corner-counting cannot separate them — the model has to judge the *character*
of the boundary. Every family is normalised to the same inscribing radius, and
every family carries the same faint surface texture and slight ellipticity, so
neither size nor smoothness-of-rasterisation is a shortcut cue.

### Why this is the useful experiment

`a`, `d`, `b`, `c` are **continuous**, and `--difficulty` scales all of them at
once. Build a sweep and you get a psychometric curve instead of a single number:

```
Accuracy by difficulty (deformation amplitude; lower = subtler):
difficulty   0.35   0.60   1.00
lobulated   0.167  0.833  1.000     <- threshold sits between d=0.35 and d=0.6
```

which supports statements like *"MedGemma separates lobulated from round only
once bulges exceed ~15% of R"*. Measure the deformation amplitude actually
present in the annotated lesions and you learn whether the real task is even
above the model's resolution.

It is also an **upper bound** on the real `shape` feature: same question, same
vocabulary, same prompt path, but perfect labels and no anatomy. Real-MRI
accuracy can only be lower, and the gap isolates "real images + inter-rater
label noise" from "the model cannot do this shape task".

> The `clinical` prompt in `run_shape_probe.py` deliberately mirrors the
> `label_definitions` in `feature_prompts.yaml`. If you retune those, retune the
> prompt to match — otherwise the probe stops bounding the real run.

**Before spending GPU time, look at `preview.py`'s contact sheet and check you
would label the tiles correctly yourself.** If a human cannot separate the
classes at `d=0.35`, a model failing there is not evidence about the model.

## Why it's a separate directory

Nothing in `preprocess/` or `medgemma_pilot/` is touched. This module only
*consumes* the preprocess `metadata.csv` (it re-uses the existing crops rather
than re-reading NIfTIs) and *imports* `medgemma_pilot/run_medgemma.py` for model
loading, generation, thinking-block stripping, and answer parsing — so the
probe's inference path is identical to the real run by construction, not by
copy-paste.

```
shape_probe/
  shapes.py           # icon rasterizers + the clinical radial generator
                      #   (red outline, thickness 2 — matches preprocess/overlay.py)
  build_shapes.py     # metadata.csv -> shape images + shape_metadata.csv
  preview.py          # contact sheet QC before burning GPU time
  run_shape_probe.py  # infer + eval  (imports medgemma_pilot/run_medgemma.py)
  review_server.py    # browser UI: accuracy + click through the wrong ones
  jobs/run_shape_probe.sh   # SLURM launcher (inference only)
```

`run_shape_probe.py` takes the same throughput flags as `run_medgemma.py`
(`--backend`, `--batch-size`, `--num-shards`, `--shard-index`) and uses its
`make_generate`, so the one-process-per-GPU pattern is identical. There is **no
aggregate step**: the probe scores per image, not per lesion, so shard CSVs are
just concatenated — `--mode eval --results a.shard0.csv a.shard1.csv ...`.

## How the shape is placed

The preprocess crops are already lesion-bbox + margin, so **the lesion centre is
the crop centre** — the shape goes at the image centre. Its radius is derived
from the real lesion extent, recovered from the `crop_bbox` and `margin_used`
columns:

```
lesion_frac = (crop_extent - 2 * margin) / crop_extent
radius      = 0.5 * min(H * frac_rows, W * frac_cols) * --shape-scale
```

so the shape covers about the area the true contour would. That keeps the probe
at the real task's difficulty instead of making it a trivially large shape.
Rotation is randomised per image; all shapes — icons and clinical families
alike — are inscribed in the same circle, so they can't be told apart by size.

*Caveat:* `crop_bbox` is the **requested** box. Under the pipeline default
`--pad-mode clip`, lesions touching the volume border yield a smaller actual
crop, so those rows are centred a few pixels off and slightly over-sized. Use
`--pad-mode pad` in preprocess if you want it exact.

## Conditions

`--background` gives a small ladder that separates two different failures:

| value   | what it tests |
|---------|---------------|
| `mri`   | **the real question** — shape drawn over the actual lesion crop |
| `blank` | can the model see shapes at all, with no anatomy competing? |
| `noise` | is it the anatomy or just texture that hides the overlay? |

If `blank` is high and `mri` is at chance, the overlay is being drowned out by
the image. If even `blank` is at chance, the model can't do the shape task at
all and the probe says nothing about contours — try `--shape-scale 1.5` or
`--filled` first.

Other knobs worth a run: `--filled` (solid shape — an upper bound on
salience), `--shape-scale`, `--all-shapes` (paired design: every shape in the
set on each background, removes background as a confound).

## Run

```bash
# 1. Build the pseudo-segmentation images (no GPU)
#    perception ladder:
python build_shapes.py \
    --metadata /results/preprocess/overlay_128/metadata.csv \
    --out-root /results/shape_probe/mri --background mri

#    discrimination ladder + amplitude sweep. One random (balanced) class per
#    source row per level = 3 images per row, matching how the icons build
#    behaves. Add --all-shapes for the paired design instead (5x more images:
#    every class on every background).
python build_shapes.py \
    --metadata /results/preprocess/overlay_128/metadata.csv \
    --out-root /results/shape_probe/clinical --background mri \
    --shape-set clinical --difficulty 1.0,0.6,0.35

# 2. QC — look at this before running inference
python preview.py --metadata /results/shape_probe/mri/shape_metadata.csv \
    --out /results/shape_probe/mri/preview.png --n 24

# 3. Infer (resume-safe: re-run the same command to continue)
python run_shape_probe.py --mode infer \
    --model-id /models/medgemma-1.5-4b-it \
    --metadata /results/shape_probe/mri/shape_metadata.csv \
    --out /results/shape_probe/mri/probe_results.csv

# 4. Score
python run_shape_probe.py --mode eval --results /results/shape_probe/mri/probe_results.csv
```

Start with `--limit 40` on steps 1 and 3 to get a read in a few minutes.

## Few-shot

`--num-few-shot N` prepends N labeled examples **per class** as completed
user → assistant turns before the query image. `1` with the clinical set means
five example turns, one per margin class, interleaved.

```bash
# zero-shot and few-shot on the SAME image set (exemplars from a separate build,
# so nothing is held out of --metadata and the two runs are directly comparable)
python run_shape_probe.py --mode infer --num-few-shot 1 \
    --few-shot-metadata /results/shape_probe/clinical_easy/shape_metadata.csv \
    --metadata /results/shape_probe/clinical/shape_metadata.csv \
    --out /results/shape_probe/clinical/probe_results_1shot.csv

# or draw exemplars from the build itself -- those images are then held out of
# scoring (a model must not be graded on an image it was just shown the answer to)
python run_shape_probe.py --mode infer --num-few-shot 1 \
    --metadata /results/shape_probe/clinical/shape_metadata.csv \
    --out /results/shape_probe/clinical/probe_results_1shot.csv
```

Exemplars are picked from the **easiest difficulty present** — the point of an
example is to show the prototype, so on a sweep you want a pronounced lobulated
margin teaching the class, then to ask about `d=0.35`. Selection is seeded
(`--seed`) and happens before sharding, so every shard shows the model the same
examples and the shard CSVs remain concatenable.

This matters far more for `clinical` than for `icons`. Nobody needs an example
to know what a triangle is; "lobulated vs irregular" is a wording judgement, and
a worked example pins it down in a way the prose definitions cannot. The
`num_few_shot` column is stamped on every row, so `--mode eval` and the review
server both break accuracy down by it — concatenate a 0-shot and a 1-shot
results CSV to read the contrast off one table.

> Prefer `--few-shot-metadata` when you plan to compare against a zero-shot run.
> Drawing exemplars from the scored build removes 5 images from the denominator,
> so the two runs are no longer scoring quite the same set.

## On the cluster

```bash
sbatch run_shape_probe.sh          # edit CONDITIONS / NUM_SHARDS / BATCH_SIZE at the top
```

Inference only — run steps 1 and 2 (build + preview) yourself first, then point
`SHAPE_META` at the `shape_metadata.csv` they produced. It shards across GPUs
and scores. Re-submitting the same script resumes from the existing shard CSVs
rather than re-inferring.

For a second condition (`blank`, `noise`, `mri_big`, …), build it and re-submit
with `SHAPE_META`/`OUTDIR` pointed at that directory; to compare conditions in
one table, pass every results CSV to a single `--mode eval` — it breaks accuracy
down by `background` automatically.

This is a separate launcher from the real experiment's, on purpose — the probe
takes no feature config and has no aggregate step. Only the middle (shard,
batch, infer) is shared, and that is shared through the identical CLI flags
rather than by merging the two scripts.

Controls are just the same three commands with a different `--background` and
`--out-root`; to score them together, concatenate the results CSVs — `eval`
breaks accuracy down by `background`, `modality` and `plane` automatically.

## Review UI

```bash
python review_server.py --results /results/shape_probe/mri/probe_results.csv --port 8547
# several files are concatenated -- shards, or one per condition:
python review_server.py --results /results/shape_probe/*/probe_results*.csv --port 8547
```

Flat by design, unlike `medgemma_pilot/review_server.py`. That one groups by
subject → feature because the real experiment's unit of truth is the lesion and
its per-image predictions have to be majority-voted. Here **every image carries
its own ground truth**, so there is nothing to aggregate — one summary page
(overall accuracy vs chance, breakdowns by shape / background / modality /
plane, confusion matrix, prediction distribution) and a filterable gallery.

The gallery filters are the point: `only wrong` next to the actual images is how
you find out *why* it failed — outline too thin at 128×128, shape clipped, star
read as a blob — rather than just that it did.

## Reading the output

`eval` prints accuracy vs chance, per-shape recall, the **prediction
distribution** and a confusion matrix. Watch the distribution: a model that
answers "circle" for everything lands at ~25% *and* collapses onto one label —
that is a null result, not partial ability. Genuine partial perception looks
like a spread of predictions with a visible diagonal.

Dependencies are the ones `medgemma_pilot` already needs (torch, transformers,
pandas, pillow) plus `opencv-python`, which `preprocess` already uses.
