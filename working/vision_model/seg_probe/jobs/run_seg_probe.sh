#!/bin/bash
# Segmentation probe: can an automatic segmenter replace the radiologist contour
# without destroying the margin features the pipeline extracts?
# See ../README.md. Reports Dice AND boundary roughness -- the second one is the
# point, because a smoothness-prior failure looks fine in Dice alone.
#
# ORDER MATTERS. This runs three backends in one job:
#   box_fill       CPU, no weights -- the Dice FLOOR (fills the prompt box).
#                  A real segmenter must clear it by a wide margin.
#   smooth_oracle  CPU, no weights -- GT mask, morphologically smoothed. Dice
#                  ~0.99 by construction while boundary detail is destroyed.
#                  This is the METRIC CHECK: if the report does not flag
#                  smooth_oracle as smoothed, it will not flag MedSAM either,
#                  and the whole run is uninterpretable.
#   medsam         GPU. The actual question.
# The two CPU baselines cost seconds. Never read the medsam number without them.
#
# Set RUN_SAM=1 to add vanilla facebook/sam-vit-base. Worth it: bone-tumour MRI
# is out of distribution for BOTH, and the medical fine-tune does not reliably
# win on OOD anatomy. Knowing that early saves a lot of time.
#
# ONE GPU IS ENOUGH. MedSAM is ViT-B (~risk-free memory-wise at 256px crops) and
# there is no sharding in run_seg_probe.py -- it is a per-image loop over a few
# hundred/thousand small crops, not a 27B decode. If it ever gets slow, shard by
# splitting the metadata CSV, not by asking for more GPUs here.
#
#SBATCH --job-name=seg_probe
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/prjs1779/BONE-AI/logs/out/slurm-%x-%j.out
#SBATCH --error=/projects/prjs1779/BONE-AI/logs/err/slurm-%x-%j.err

set -euo pipefail

export UV_CACHE_DIR=/projects/prjs1779/BONE-AI/.uv-cache
export HF_HOME=/scratch-shared/$USER/hf-cache
# Compute nodes have no internet. Fail loudly on a cache miss instead of hanging
# on a connection attempt that cannot succeed -- see the MODEL preflight below.
export HF_HUB_OFFLINE=1

# Absolute, because sbatch COPIES this script to a node-local spool dir before
# running it -- inside the job ${BASH_SOURCE[0]} is /var/spool/slurmd/..., not
# the repo, so deriving the path from the script's own location lands outside
# the uv project ("warning: --no-sync has no effect when used outside of a
# project", then ModuleNotFoundError: No module named 'torch').
REPO=/gpfs/work2/0/prjs1779/BONE-AI/Bone_CLS

# Output of preprocess run.py --save-mask (the metadata.csv must have a populated
# mask_path column; the probe scores in the crop's own frame and refuses to
# re-derive it).
META=${META:-/projects/prjs1779/BONE-AI/output/preprocess/shape_256_m/metadata.csv}
OUTDIR=${OUTDIR:-/scratch-shared/$USER/BONE-AI/results/seg_probe/shape_256_m}

# Local checkpoint dirs. See "PRE-DOWNLOAD" below -- these must already exist.
MEDSAM=${MEDSAM:-/scratch-shared/$USER/models/medsam-vit-base}
SAM=${SAM:-/scratch-shared/$USER/models/sam-vit-base}
RUN_SAM=${RUN_SAM:-0}

# Which CPU baselines to (re)run. Set CPU_BASELINES="" if they are already in
# $OUTDIR from an earlier job -- the eval globs the directory, so previously
# written results are still included in the table. They cost seconds, so the only
# reason to skip is that you have them.
CPU_BASELINES=${CPU_BASELINES:-"box_fill smooth_oracle"}

# Prompt-box jitter. The tight GT box encodes the lesion extent to the pixel, so
# prompting with it measures "can SAM fill in a box derived from the answer".
# 0.10 / 0.9-1.2x is roughly a good detector. Sweep SHIFT upward to find how
# accurate an upstream detector would have to be.
SHIFT=${SHIFT:-0.10}
SCALE_LO=${SCALE_LO:-0.90}
SCALE_HI=${SCALE_HI:-1.20}

# smooth_oracle's kernel, in pixels, and therefore how hard it smooths RELATIVE
# to the crop. 9 suits 256px crops; at the 128px default it would be ~2x more
# aggressive. This is a knob on the metric CHECK, not on any result: if the
# report does not print "SMOOTHING DETECTED" for smooth_oracle, the check failed
# -- raise it until it does, then trust the medsam verdict. Leaving it too low
# means the report cannot detect smoothing at all and medsam gets a free pass.
SMOOTH_PX=${SMOOTH_PX:-9}
# Set LIMIT=40 for a smoke test that finishes in a minute.
LIMIT_ARG=${LIMIT:+--limit $LIMIT}

cd "$REPO/working/vision_model/seg_probe"
mkdir -p "$OUTDIR"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "  META    = $META"
echo "  OUTDIR  = $OUTDIR"
echo "  MEDSAM  = $MEDSAM"
echo "  jitter  = shift $SHIFT, scale $SCALE_LO-$SCALE_HI"

# ------------------------------------------------------------- preflight ----
# Each of these is a failure that otherwise surfaces late, as something that
# looks like a bad result rather than a broken setup.
[[ -f "$REPO/pyproject.toml" ]] || {
    echo "FATAL: no pyproject.toml under $REPO -- uv would run outside the project" >&2
    exit 1
}
[[ -f "$META" ]] || { echo "FATAL: no metadata at $META" >&2; exit 1; }

# mask_path is added by preprocess --save-mask. Without it every row is skipped
# and you get an empty results CSV, which scores as "no rows" rather than as an
# error. Check the column exists AND that some row actually has a value.
uv run --no-sync python - "$META" <<'PY' || exit 1
import csv, sys
with open(sys.argv[1], newline="") as fh:
    r = csv.DictReader(fh)
    if "mask_path" not in (r.fieldnames or []):
        sys.exit("FATAL: metadata has no mask_path column -- re-run preprocess with --save-mask")
    if not any((row.get("mask_path") or "").strip() for row in r):
        sys.exit("FATAL: mask_path column is present but empty on every row")
print("metadata OK: mask_path present and populated")
PY

uv run --no-sync python -c "import torch, transformers, cv2" || {
    echo "FATAL: the venv is not importable (mid-install, or wrong project root)" >&2
    exit 1
}

# PRE-DOWNLOAD. Compute nodes have no internet, so the weights must be on disk
# before the job starts. On a LOGIN node, once:
#   HF_HOME=/scratch-shared/$USER/hf-cache uv run python -c "
#     from transformers import SamModel, SamProcessor
#     for repo, dst in [('wanglab/medsam-vit-base', '$MEDSAM'),
#                       ('facebook/sam-vit-base',   '$SAM')]:
#         SamModel.from_pretrained(repo).save_pretrained(dst)
#         SamProcessor.from_pretrained(repo).save_pretrained(dst)"
[[ -d "$MEDSAM" ]] || {
    echo "FATAL: no MedSAM checkpoint at $MEDSAM." >&2
    echo "       Compute nodes are offline -- download it on a login node first" >&2
    echo "       (see the PRE-DOWNLOAD comment in this script)." >&2
    exit 1
}

nvidia-smi -L || { echo "FATAL: no GPU visible" >&2; exit 1; }

# ------------------------------------------------------------------ run ----
# CPU baselines first: they are seconds, and they are what makes the medsam
# number readable. Running them after would invite reading medsam alone.
for BE in $CPU_BASELINES; do
    echo "--- $BE (CPU baseline)"
    uv run --no-sync python run_seg_probe.py --mode segment --backend "$BE" \
        --metadata "$META" --out "$OUTDIR/$BE/seg_results.csv" \
        --shift-frac "$SHIFT" --scale-lo "$SCALE_LO" --scale-hi "$SCALE_HI" \
        --smooth-px "$SMOOTH_PX" $LIMIT_ARG
done

echo "--- medsam (GPU)"
uv run --no-sync python run_seg_probe.py --mode segment --backend medsam \
    --model-id "$MEDSAM" \
    --metadata "$META" --out "$OUTDIR/medsam/seg_results.csv" \
    --shift-frac "$SHIFT" --scale-lo "$SCALE_LO" --scale-hi "$SCALE_HI" \
    $LIMIT_ARG

if (( RUN_SAM )); then
    [[ -d "$SAM" ]] || { echo "FATAL: RUN_SAM=1 but no checkpoint at $SAM" >&2; exit 1; }
    echo "--- sam (GPU)"
    uv run --no-sync python run_seg_probe.py --mode segment --backend sam \
        --model-id "$SAM" \
        --metadata "$META" --out "$OUTDIR/sam/seg_results.csv" \
        --shift-frac "$SHIFT" --scale-lo "$SCALE_LO" --scale-hi "$SCALE_HI" \
        $LIMIT_ARG
fi

# --------------------------------------------------------------- report ----
# One table over every backend that produced results, so the floor, the metric
# check and the real model are read side by side.
shopt -s nullglob
RESULTS=("$OUTDIR"/*/seg_results.csv)
(( ${#RESULTS[@]} )) || { echo "FATAL: no results CSVs under $OUTDIR" >&2; exit 1; }

uv run --no-sync python run_seg_probe.py --mode eval --results "${RESULTS[@]}" \
    | tee "$OUTDIR/report.txt"

# Worst-Dice contact sheet: green = radiologist, red = automatic, yellow =
# jittered prompt box. The mean tells you nothing you can act on.
uv run --no-sync python run_seg_probe.py --mode preview \
    --results "$OUTDIR/medsam/seg_results.csv" \
    --out "$OUTDIR/medsam/worst.png" --n 16

echo "[$(date -Is)] done -- report: $OUTDIR/report.txt"
