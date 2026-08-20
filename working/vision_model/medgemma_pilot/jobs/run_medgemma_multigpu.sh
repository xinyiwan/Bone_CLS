#!/bin/bash
# Data-parallel MedGemma inference: one process per GPU, each handling every
# Nth metadata row, then a single aggregate over the shard CSVs.
#
# Every row is independent, so this is a clean ~4x on top of batching. Note we
# run 4 SEPARATE single-GPU processes rather than one process with
# device_map="auto" across 4 GPUs -- a 4B model fits in one GPU, and splitting
# it across four would only add inter-GPU traffic and make each token slower.
#
#SBATCH --job-name=run_medgemma_mg
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
METADATA=/projects/prjs1779/BONE-AI/output/preprocess/metadata.csv
OUTDIR=/scratch-shared/$USER/BONE-AI
OUT=$OUTDIR/results.csv
NUM_SHARDS=1
BATCH_SIZE=16

mkdir -p "$OUTDIR"
cd Bone_CLS/working/vision_model/medgemma_pilot

# Each shard writes $OUT with a .shard<i> suffix (run_medgemma.py adds it when
# --num-shards > 1) -- they append concurrently, so they must not share a file.
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i uv run --no-sync python run_medgemma.py --mode infer \
        --model-id "$MODEL" \
        --metadata "$METADATA" \
        --config feature_prompts.yaml \
        --batch-size $BATCH_SIZE \
        --num-shards $NUM_SHARDS --shard-index "$i" \
        --out "$OUT" &
done
wait   # any shard failing makes `wait` return non-zero, and set -e aborts before aggregating

SHARDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    SHARDS+=("${OUT%.csv}.shard${i}.csv")
done

uv run --no-sync python run_medgemma.py --mode aggregate \
    --inference-results "${SHARDS[@]}" \
    --out "$OUTDIR/results_sanity.csv"
