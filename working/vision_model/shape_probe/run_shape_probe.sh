#!/bin/bash
# Shape probe (pseudo-segmentation sanity check): can MedGemma see the red
# overlay at all? Chance = 25%. See README.md for what the result means.
#
# INFERENCE ONLY -- assumes build_shapes.py has already written $SHAPE_META and
# you have eyeballed the preview. There is no aggregate step: the probe scores
# per image, not per lesion, so shard CSVs are just concatenated at eval.
#
#SBATCH --job-name=shape_probe_mg
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/projects/prjs1779/BONE-AI/logs/out/slurm-%x-%j.out
#SBATCH --error=/projects/prjs1779/BONE-AI/logs/err/slurm-%x-%j.err

set -euo pipefail

export UV_CACHE_DIR=/projects/prjs1779/BONE-AI/.uv-cache
export HF_HOME=/scratch-shared/$USER/hf-cache

MODEL=/scratch-shared/$USER/models/medgemma-1.5-4b-it
# Output of build_shapes.py --out-root <dir>  (NOT the preprocess metadata.csv).
SHAPE_META=/scratch-shared/$USER/BONE-AI/shape_probe/mri/shape_metadata.csv
OUTDIR=/scratch-shared/$USER/BONE-AI/shape_probe/mri
OUT=$OUTDIR/probe_results.csv
NUM_SHARDS=1
BATCH_SIZE=16

# Resolve the repo dir from this script's location, so the job does not depend
# on the submitting shell's cwd.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTDIR"

# Each shard writes $OUT with a .shard<i> suffix (run_shape_probe.py adds it when
# --num-shards > 1) -- they append concurrently, so they must not share a file.
# Re-running the same command resumes from whatever is already in those files.
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i uv run --no-sync python run_shape_probe.py --mode infer \
        --model-id "$MODEL" \
        --metadata "$SHAPE_META" \
        --batch-size $BATCH_SIZE \
        --num-shards $NUM_SHARDS --shard-index "$i" \
        --out "$OUT" &
done
wait   # any shard failing makes `wait` return non-zero, and set -e aborts before scoring

SHARDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    if (( NUM_SHARDS > 1 )); then SHARDS+=("${OUT%.csv}.shard${i}.csv"); else SHARDS+=("$OUT"); fi
done

uv run --no-sync python run_shape_probe.py --mode eval --results "${SHARDS[@]}"
