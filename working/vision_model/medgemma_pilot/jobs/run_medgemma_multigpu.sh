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

# Absolute, because sbatch COPIES this script to a node-local spool dir before
# running it -- inside the job ${BASH_SOURCE[0]} is /var/spool/slurmd/..., not
# the repo, so deriving the path from the script's own location lands outside
# the uv project ("warning: --no-sync has no effect when used outside of a
# project", then ModuleNotFoundError: No module named 'torch').
REPO=/gpfs/work2/0/prjs1779/BONE-AI/Bone_CLS

MODEL=/scratch-shared/$USER/models/medgemma-1.5-4b-it
METADATA=/projects/prjs1779/BONE-AI/output/preprocess/metadata.csv
OUTDIR=/scratch-shared/$USER/BONE-AI
OUT=$OUTDIR/results.csv
NUM_SHARDS=1
BATCH_SIZE=16

mkdir -p "$OUTDIR"
cd "$REPO/working/vision_model/medgemma_pilot"

# Preflight: two seconds here beats discovering a broken environment after SLURM
# has handed us the GPUs. An interactive `uv add` that overlaps a job start
# rewrites .venv underneath it, and the importer sees a half-unpacked package
# ("No module named 'torch._utils_internal'"). Do not `uv add` / `uv sync` while
# jobs are queued or running.
[[ -f "$REPO/pyproject.toml" ]] || {
    echo "FATAL: no pyproject.toml under $REPO -- uv would run outside the project" >&2
    exit 1
}
uv run --no-sync python -c "import torch, transformers" || {
    echo "FATAL: the venv is not importable (mid-install, or wrong project root)" >&2
    exit 1
}

# Each shard writes $OUT with a .shard<i> suffix (run_medgemma.py adds it when
# --num-shards > 1) -- they append concurrently, so they must not share a file.
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i uv run --no-sync python run_medgemma.py --mode infer \
        --model-id "$MODEL" \
        --metadata "$METADATA" \
        --config feature_prompts.yaml \
        --batch-size $BATCH_SIZE \
        --num-shards $NUM_SHARDS --shard-index "$i" \
        --out "$OUT" &
    pids+=($!)
done

# Wait on each pid INDIVIDUALLY. A bare `wait` (no arguments) always returns 0
# in bash -- it discards the children's exit statuses -- so `set -e` never fires
# and a dead shard is silently skipped. That is how a job with a Python
# traceback in its log still gets reported COMPLETED, and how a majority vote
# taken over partial shards gets aggregated as if it were the full run.
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
(( rc == 0 )) || { echo "FATAL: a shard failed -- refusing to aggregate partial results" >&2; exit 1; }

SHARDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    # run_medgemma.py only adds the .shard<i> suffix when --num-shards > 1, so
    # the single-shard case must use $OUT unchanged -- otherwise aggregate is
    # handed a path that was never written.
    if (( NUM_SHARDS > 1 )); then SHARDS+=("${OUT%.csv}.shard${i}.csv"); else SHARDS+=("$OUT"); fi
done

uv run --no-sync python run_medgemma.py --mode aggregate \
    --inference-results "${SHARDS[@]}" \
    --out "$OUTDIR/results_sanity.csv"
