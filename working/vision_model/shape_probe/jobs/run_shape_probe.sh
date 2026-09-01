#!/bin/bash
# Shape probe (pseudo-segmentation sanity check). Two ladders, chosen when you
# BUILD the images with build_shapes.py --shape-set:
#   icons     can MedGemma see the red overlay at all?          chance 25%
#   clinical  can it discriminate the 5 margin classes?         chance 20%
# run_shape_probe.py reads which one from the metadata, so nothing here changes
# between them -- just point SHAPE_META/OUTDIR at the right build.
# See ../README.md for what the result means.
#
# INFERENCE ONLY -- assumes build_shapes.py has already written $SHAPE_META and
# you have eyeballed the preview. There is no aggregate step: the probe scores
# per image, not per lesion, so shard CSVs are just concatenated at eval.
#
# GPUs: keep --nodes=1 and set --gpus-per-node = NUM_SHARDS below. sbatch runs
# this script on the FIRST allocated node only, so --nodes=2 does not give it a
# second GPU -- the extra node just sits idle while CUDA_VISIBLE_DEVICES=1
# selects a device that does not exist. That case does NOT crash: CUDA reports
# no devices, device_map="auto" quietly places the model on CPU, and the shard
# appears to hang instead of failing. Multi-node would need srun per node.
#
#SBATCH --job-name=shape_probe_mg
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
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
# project", then ModuleNotFoundError: No module named 'torch'). $SLURM_SUBMIT_DIR
# would work but silently depends on where you happened to type sbatch.
REPO=/gpfs/work2/0/prjs1779/BONE-AI/Bone_CLS
BACKGROUND_VAR=mri
LEVEL=s


MODEL=/scratch-shared/$USER/models/medgemma-27b-it
# Output of build_shapes.py --out-root <dir>  (NOT the preprocess metadata.csv).
SHAPE_META=/scratch-shared/$USER/BONE-AI/pseudo_shape/$BACKGROUND_VAR/shape_256_cli_${LEVEL}/shape_metadata.csv
OUTDIR=/scratch-shared/$USER/BONE-AI/results/pseudo_shape/$BACKGROUND_VAR
OUT=$OUTDIR/probe_results_cli_${LEVEL}_27b.csv
NUM_SHARDS=2
BATCH_SIZE=64

# Token budget for the whole generation, THINKING BLOCK INCLUDED. The script
# default is 512, which was not enough: MedGemma 1.5 reasons at length on the
# clinical set, ran past the cap mid-thought, and never emitted its JSON answer.
# Before the parser was fixed those rows silently resolved to the FIRST option in
# the vocabulary -- `round_oval` -- so truncation showed up as inflated
# round_oval recall rather than as an error. Set it explicitly here so the budget
# is a visible part of the run config, and check the "generations cut off
# mid-thought" line that --mode eval now prints: it should be 0.
#
# Costs throughput: static batching returns only when the SLOWEST member of a
# batch stops, so a bigger cap lets one long thought hold up its whole batch.
MAX_NEW_TOKENS=1536

cd "$REPO/working/vision_model/shape_probe"

mkdir -p "$OUTDIR"

# Preflight. Two seconds here beats discovering a broken environment after SLURM
# has handed us the GPUs, and it separates the two failures that look identical
# in the log:
#   - no pyproject.toml found -> uv ran a bare interpreter (wrong $REPO / cwd)
#   - half-unpacked package    -> an interactive `uv add` overlapped this job
#     start and rewrote .venv underneath it ("No module named 'torch._utils_internal'")
# Do not `uv add` / `uv sync` while jobs are queued or running.
[[ -f "$REPO/pyproject.toml" ]] || {
    echo "FATAL: no pyproject.toml under $REPO -- uv would run outside the project" >&2
    exit 1
}
# Every shard pins itself to CUDA_VISIBLE_DEVICES=$i, so there must be at least
# NUM_SHARDS real devices ON THIS NODE. Fewer means the surplus shards land on a
# nonexistent device and fall back to CPU -- no error, just a shard that never
# finishes. Fail here instead.
N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if (( N_GPU < NUM_SHARDS )); then
    echo "FATAL: NUM_SHARDS=$NUM_SHARDS but this node has $N_GPU GPU(s)." >&2
    echo "       Set #SBATCH --gpus-per-node=$NUM_SHARDS with --nodes=1 (sbatch only" >&2
    echo "       runs this script on the first node, so --nodes>1 adds no GPUs here)." >&2
    exit 1
fi
echo "$N_GPU GPU(s) on this node, running $NUM_SHARDS shard(s)"

uv run --no-sync python -c "import torch, transformers" || {
    echo "FATAL: the venv is not importable (mid-install, or wrong project root)" >&2
    exit 1
}

# Each shard writes $OUT with a .shard<i> suffix (run_shape_probe.py adds it when
# --num-shards > 1) -- they append concurrently, so they must not share a file.
# Re-running the same command resumes from whatever is already in those files.
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i uv run --no-sync python run_shape_probe.py --mode infer \
        --model-id "$MODEL" \
        --metadata "$SHAPE_META" \
        --batch-size $BATCH_SIZE \
        --max-new-tokens $MAX_NEW_TOKENS \
        --num-shards $NUM_SHARDS --shard-index "$i" \
        --out "$OUT" &
    pids+=($!)
done

# Wait on each pid INDIVIDUALLY. A bare `wait` (no arguments) always returns 0
# in bash -- it discards the children's exit statuses -- so `set -e` never fires
# and a dead shard is silently skipped. That is how a job with a Python
# traceback in its log still gets reported COMPLETED, and how a partial-data
# accuracy gets scored as if it were the real thing.
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
(( rc == 0 )) || { echo "FATAL: a shard failed -- refusing to score partial results" >&2; exit 1; }

SHARDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    # run_shape_probe.py only adds the .shard<i> suffix when --num-shards > 1,
    # so the single-shard case must use $OUT unchanged.
    if (( NUM_SHARDS > 1 )); then SHARDS+=("${OUT%.csv}.shard${i}.csv"); else SHARDS+=("$OUT"); fi
done

uv run --no-sync python run_shape_probe.py --mode eval --results "${SHARDS[@]}"
