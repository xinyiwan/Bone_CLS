#!/bin/bash
# MedSAM only, on CPU (genoa). Use this when the gpu_a100 queue is long: MedSAM
# is ViT-B over a few hundred small crops, so a Genoa node that starts NOW beats
# an A100 that starts in an hour -- and costs far fewer SBU. For a repeated
# --shift-frac sweep, prefer jobs/run_seg_probe.sh on GPU instead.
#
# Nothing about the result differs. backends.py picks the device automatically
# (cuda if available, else cpu); only wall-clock changes.
#
# ASSUMES box_fill and smooth_oracle have already been run into $OUTDIR. They are
# the Dice floor and the metric check -- the medsam number is not interpretable
# without them. The eval at the end globs whatever is present and will say so if
# they are missing.
#
#SBATCH --job-name=medsam_cpu
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=02:00:00
#SBATCH --output=/projects/prjs1779/BONE-AI/logs/out/slurm-%x-%j.out
#SBATCH --error=/projects/prjs1779/BONE-AI/logs/err/slurm-%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------- paths ----
PROJECT_ROOT=${PROJECT_ROOT:-/projects/prjs1779/BONE-AI}
# Absolute, because sbatch copies this script to a node-local spool dir; relative
# paths would resolve against wherever you happened to type sbatch.
REPO=${REPO:-/gpfs/work2/0/prjs1779/BONE-AI/Bone_CLS}

META=${META:-${PROJECT_ROOT}/output/preprocess/shape_256_m/metadata.csv}
OUTDIR=${OUTDIR:-/scratch-shared/$USER/BONE-AI/results/seg_probe/shape_256_m}
MEDSAM=${MEDSAM:-/scratch-shared/$USER/models/medsam-vit-base}

# Must match whatever box_fill / smooth_oracle were run with, or the eval is
# averaging across different prompt difficulties (it will warn if they differ).
SHIFT=${SHIFT:-0.10}
SCALE_LO=${SCALE_LO:-0.90}
SCALE_HI=${SCALE_HI:-1.20}
# LIMIT=40 for a smoke test. The per-row log prints img/s and an ETA, so a short
# run tells you exactly what walltime the full set needs.
LIMIT_ARG=${LIMIT:+--limit $LIMIT}

# ---------------------------------------------------------- environment ----
export UV_CACHE_DIR=${UV_CACHE_DIR:-${PROJECT_ROOT}/.uv-cache}
export HF_HOME=${HF_HOME:-/scratch-shared/$USER/hf-cache}
# Compute nodes have no internet -- fail on a cache miss rather than hanging on a
# connection that cannot succeed. The MEDSAM preflight below is the real guard.
export HF_HUB_OFFLINE=1
# torch defaults to one thread per core only in some builds; set it explicitly so
# the 192 cores are actually used. This is the difference between ~1 img/s and
# something usable -- the ViT-B encoder at 1024x1024 is all matmul.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export PYTHONUNBUFFERED=1

cd "$REPO/working/vision_model/seg_probe"
mkdir -p "$OUTDIR"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "  META    = $META"
echo "  OUTDIR  = $OUTDIR"
echo "  MEDSAM  = $MEDSAM"
echo "  threads = $OMP_NUM_THREADS"
echo "  jitter  = shift $SHIFT, scale $SCALE_LO-$SCALE_HI"

# ------------------------------------------------------------- preflight ----
[[ -f "$REPO/pyproject.toml" ]] || {
    echo "FATAL: no pyproject.toml under $REPO -- uv would run outside the project" >&2
    exit 1
}
[[ -f "$META" ]] || { echo "FATAL: no metadata at $META" >&2; exit 1; }

# An absent/empty mask_path makes every row skip, which scores as "no rows"
# rather than as an error -- a clean-looking empty report. Catch it in seconds.
uv run --no-sync python - "$META" <<'PY' || exit 1
import csv, sys
with open(sys.argv[1], newline="") as fh:
    r = csv.DictReader(fh)
    if "mask_path" not in (r.fieldnames or []):
        sys.exit("FATAL: metadata has no mask_path column -- re-run preprocess with --save-mask")
    rows = list(r)
n = sum(1 for row in rows if (row.get("mask_path") or "").strip())
if not n:
    sys.exit("FATAL: mask_path column is present but empty on every row")
print(f"metadata OK: {n}/{len(rows)} row(s) have a mask_path")
PY

uv run --no-sync python -c "import torch, transformers, cv2" || {
    echo "FATAL: the venv is not importable (mid-install, or wrong project root)" >&2
    exit 1
}

# PRE-DOWNLOAD, on a LOGIN node, once (compute nodes are offline):
#   HF_HOME=/scratch-shared/$USER/hf-cache uv run python -c "
#     from transformers import SamModel, SamProcessor
#     SamModel.from_pretrained('wanglab/medsam-vit-base').save_pretrained('$MEDSAM')
#     SamProcessor.from_pretrained('wanglab/medsam-vit-base').save_pretrained('$MEDSAM')"
[[ -d "$MEDSAM" ]] || {
    echo "FATAL: no MedSAM checkpoint at $MEDSAM." >&2
    echo "       Compute nodes are offline -- download it on a login node first" >&2
    echo "       (see the PRE-DOWNLOAD comment in this script)." >&2
    exit 1
}

# ------------------------------------------------------------------ run ----
uv run --no-sync python run_seg_probe.py --mode segment --backend medsam \
    --model-id "$MEDSAM" \
    --metadata "$META" \
    --out "$OUTDIR/medsam/seg_results.csv" \
    --shift-frac "$SHIFT" --scale-lo "$SCALE_LO" --scale-hi "$SCALE_HI" \
    $LIMIT_ARG

# --------------------------------------------------------------- report ----
shopt -s nullglob
RESULTS=("$OUTDIR"/*/seg_results.csv)
(( ${#RESULTS[@]} )) || { echo "FATAL: no results CSVs under $OUTDIR" >&2; exit 1; }
if (( ${#RESULTS[@]} < 3 )); then
    echo "NOTE: only ${#RESULTS[@]} backend(s) present. box_fill (the Dice floor) and" >&2
    echo "      smooth_oracle (the roughness-metric check) belong in the same table --" >&2
    echo "      run them too, they are seconds on CPU." >&2
fi

uv run --no-sync python run_seg_probe.py --mode eval --results "${RESULTS[@]}" \
    | tee "$OUTDIR/report.txt"

# green = radiologist, red = automatic, yellow = jittered prompt box
uv run --no-sync python run_seg_probe.py --mode preview \
    --results "$OUTDIR/medsam/seg_results.csv" \
    --out "$OUTDIR/medsam/worst.png" --n 16

echo "[$(date -Is)] done -- report: $OUTDIR/report.txt"
