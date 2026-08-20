# Shape probe — can MedGemma actually see the overlay?

A pseudo-segmentation sanity experiment. In the real pilot we feed MedGemma the
lesion crop with the radiologist's **red contour** burned in
(`run_medgemma.py --use-contour`) and *assume* the model uses it. This probe
tests that assumption directly: keep everything the same — same crops, same
centre, same red colour and 2 px thickness, same decoding path — but replace the
true contour with a **circle / square / triangle / star**, and ask only *which
shape is drawn*.

**Chance is 25%.** If accuracy sits near chance on the `mri` condition, the
model is not reading the overlay, and any "contour helps" result in the main
experiment needs re-interpreting.

## Why it's a separate directory

Nothing in `preprocess/` or `medgemma_pilot/` is touched. This module only
*consumes* the preprocess `metadata.csv` (it re-uses the existing crops rather
than re-reading NIfTIs) and *imports* `medgemma_pilot/run_medgemma.py` for model
loading, generation, thinking-block stripping, and answer parsing — so the
probe's inference path is identical to the real run by construction, not by
copy-paste.

```
shape_probe/
  shapes.py           # shape rasterizers (red outline, thickness 2 — matches preprocess/overlay.py)
  build_shapes.py     # metadata.csv -> shape images + shape_metadata.csv
  preview.py          # contact sheet QC before burning GPU time
  run_shape_probe.py  # infer + eval  (imports medgemma_pilot/run_medgemma.py)
  run_shape_probe.sh  # SLURM launcher: build -> shard across GPUs -> eval
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
Rotation is randomised per image; all four shapes are inscribed in the same
circle, so they can't be told apart by size.

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
salience), `--shape-scale`, `--all-shapes` (paired design: all four shapes on
each background, removes background as a confound).

## Run

```bash
# 1. Build the pseudo-segmentation images (no GPU)
python build_shapes.py \
    --metadata /results/preprocess/overlay_128/metadata.csv \
    --out-root /results/shape_probe/mri --background mri

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

## Reading the output

`eval` prints accuracy vs chance, per-shape recall, the **prediction
distribution** and a confusion matrix. Watch the distribution: a model that
answers "circle" for everything lands at ~25% *and* collapses onto one label —
that is a null result, not partial ability. Genuine partial perception looks
like a spread of predictions with a visible diagonal.

Dependencies are the ones `medgemma_pilot` already needs (torch, transformers,
pandas, pillow) plus `opencv-python`, which `preprocess` already uses.
