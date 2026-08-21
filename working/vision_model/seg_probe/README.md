# seg_probe — can an automatic segmenter replace the radiologist contour?

The medgemma pilot feeds the VLM a crop with the radiologist's contour burned in.
Replacing that with a foundation model would make the pipeline **deployable** (no
expert 3D mask at inference time) and **reproducible** (no inter-rater variance).

The risk is specific and easy to miss: every SAM descendant has a strong
**smoothness prior**. It can match the lesion's extent — high Dice — while
erasing the lobulation and spiculation that the `shape` and margin features are
supposed to measure. You would ship a pipeline that scores well on segmentation
and quietly extracts the segmenter's prior instead of the lesion's morphology.

So this scores two things per lesion:

| | metrics | question |
|---|---|---|
| overlap | `dice`, `iou`, `hd95` | is it the same **region**? |
| roughness | `circularity`, `solidity`, `lobe_amp`, `jag_amp` | is it the same **kind of boundary**? |

and reports the roughness **delta** (auto − radiologist). That delta is the
number that decides whether this direction helps or guts Step 3.

## The two things people get wrong

**Jitter the prompt box.** SAM is promptable; the obvious prompt is the tight
bbox of the radiologist mask. But that box encodes the lesion's extent to the
pixel — SAM snaps to it, and you measure "can it fill in a box derived from the
answer". `jitter_box` shifts the centre by ±10% and rescales 0.9–1.2×, roughly
what a real detector would give you. If Dice holds up under jitter the segmenter
is finding the lesion; if it collapses, it was tracing your prompt. Sweeping
`--shift-frac` tells you how accurate an upstream detector needs to be.

`--no-jitter` exists for diagnosis only. Never report a number from it.

**Prompting is not automation.** Every model here needs a box. If that box comes
from the radiologist mask, a human is still in the loop — jitter makes the number
honest, not the pipeline autonomous. For that you need a detector or nnU-Net in
front generating the box. Say so explicitly rather than letting "foundation
model" imply "no human".

## Roughness metrics

- **circularity** `4πA/P²` — 1.0 for a circle, falls as the boundary convolves.
  Rasterisation-sensitive, so only compare masks of the same size.
- **solidity** `A / A(convex hull)` — makes no star-shaped assumption; the one
  that responds to concave bites (the `geographic` class).
- **lobe_amp / jag_amp** — resample the contour as `r(θ)` about the centroid,
  normalise to mean radius 1, FFT, and take RMS amplitude over `k=3–7`
  (lobulation) and `k=8–22` (spiculation).

That last one is deliberately the **same parameterisation** as
`shape_probe/shapes.py`, so a real lesion's `lobe_amp` is directly comparable to
the synthetic `--difficulty` axis. It's what turns "the model got 60% on shape"
into *"real lesions sit at deformation amplitude ~0.12 and the model's threshold
is ~0.15"*.

It is calibrated: generating masks from `clinical_polygon` at known amplitude and
measuring them back gives

| injected | measured `lobe_amp` |
|---|---|
| 0.262 | 0.259 |
| 0.157 | 0.156 |
| 0.091 | 0.091 |

with a noise floor of ~0.015 on a smooth ellipse. `r(θ)` assumes the contour is
star-shaped about its centroid — true for most lesions, not for strongly
exophytic or crescent ones, where the outermost crossing per angle is taken and
concavity is under-reported. `solidity` is reported alongside because it makes no
such assumption.

## Backends

Two need no weights and no GPU, and **both matter**:

- **`box_fill`** — fills the prompt box. The **Dice floor**. On a compact lesion
  in a tight crop this scores ~0.73 on synthetic data. Any real segmenter must
  clear it by a wide margin; if yours gets 0.75, that was never evidence.
- **`smooth_oracle`** — the ground-truth mask, morphologically smoothed. Dice
  ~0.99 **by construction** while the boundary detail is destroyed. This is the
  null model for the whole directory: **if the report doesn't flag
  smooth_oracle, it won't flag SAM either.** Run it first.

On the synthetic set they behave as designed:

```
                n   dice    hd95  circ_delta  jag_delta
box_fill       20  0.728  20.177       0.113      0.001
smooth_oracle  20  0.988   1.214       0.080     -0.021
```

`smooth_oracle` at Dice 0.988 has lost 46% of its `jag_amp`. That is the failure
mode this directory exists to catch, and Dice is blind to it.

Weights required: **`medsam`** (`wanglab/medsam-vit-base`), **`sam`**
(`facebook/sam-vit-base`), **`sam2`** (`facebook/sam2-hiera-large`).

> Bone-tumour MRI is **out of distribution** for all of them — the medical
> fine-tunes are heavily CT / pathology / endoscopy, MRI a minority, bone lesions
> rarer still. Run vanilla `sam` next to `medsam`: on OOD anatomy the medical
> fine-tune does not reliably win, and knowing that early saves time.
>
> And do not skip the boring baseline: you already have radiologist 3D masks. If
> that's a few hundred cases, **nnU-Net will probably beat every zero-shot model
> here**, and it's fully automatic. Foundation models win when you have no
> labels. You have labels.

## Run

```bash
# 0. preprocess must write the mask crops (new --save-mask flag)
python ../preprocess/run.py ... --save-mask

# 1. calibrate the metrics -- no GPU, do this first
python run_seg_probe.py --mode segment --backend box_fill \
    --metadata /results/preprocess/overlay_128/metadata.csv --out /results/seg_probe/box_fill/r.csv
python run_seg_probe.py --mode segment --backend smooth_oracle \
    --metadata /results/preprocess/overlay_128/metadata.csv --out /results/seg_probe/smooth/r.csv

# 2. real segmenters
python run_seg_probe.py --mode segment --backend medsam \
    --metadata /results/preprocess/overlay_128/metadata.csv --out /results/seg_probe/medsam/r.csv

# 3. one table for all of them
python run_seg_probe.py --mode eval --results /results/seg_probe/*/r.csv

# 4. look at the worst cases -- green = radiologist, red = auto, yellow = prompt box
python run_seg_probe.py --mode preview --results /results/seg_probe/medsam/r.csv \
    --out /results/seg_probe/medsam/worst.png --n 16
```

`--mode preview` sorts by **worst** Dice, not random. The mean tells you nothing
you can act on.

## How to read the result

| Dice | roughness delta | conclusion |
|---|---|---|
| high | ~0 | substitution is safe — use it, and gain reproducibility |
| high | large positive `circ`, negative `jag` | **the trap**: right region, wrong margin. Auto-seg is fine for cropping and for texture features, but margin features must not be read off it |
| low | — | not usable; try nnU-Net before concluding the approach fails |

The middle row is the likely outcome, and it isn't a dead end — it argues for
using auto-segmentation to **localize** (a box or a dot marker) while letting the
VLM find the boundary itself in the image. See the marker-strength ladder
discussion: for `shape` and margin features, overlaying an exact contour and then
asking about margin shape is close to circular anyway.
