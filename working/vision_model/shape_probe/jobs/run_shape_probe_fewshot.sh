#!/bin/bash
# Shape probe, FEW-SHOT arm. Same inference as run_shape_probe.sh, but each
# prompt is preceded by NUM_FEW_SHOT labeled example turns PER CLASS.
#
# This is aimed at the `clinical` set. Nobody needs an example to know what a
# triangle is; "lobulated vs irregular" is a wording judgement, and one worked
# example per class pins it down in a way the prose definitions cannot. Run it
# against the zero-shot results from run_shape_probe.sh -- the final eval below
# concatenates both and breaks accuracy down by num_few_shot.
#
# GPUs: keep --nodes=1 and set --gpus-per-node = NUM_SHARDS below. sbatch runs
# this script on the FIRST allocated node only, so --nodes=2 does not give it a
# second GPU -- the extra node just sits idle while CUDA_VISIBLE_DEVICES=1
# selects a device that does not exist. That case does NOT crash: CUDA reports
# no devices, device_map="auto" quietly places the model on CPU, and the shard
# appears to hang instead of failing. Multi-node would need srun per node.
#
#SBATCH --job-name=shape_probe_fs
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --time=02:00:00
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
BACKGROUND_VAR=mri
LEVEL=s

MODEL=/scratch-shared/$USER/models/medgemma-27b-it
# Output of build_shapes.py --out-root <dir>  (NOT the preprocess metadata.csv).
SHAPE_META=/scratch-shared/$USER/BONE-AI/pseudo_shape/$BACKGROUND_VAR/shape_256_cli_${LEVEL}/shape_metadata.csv
OUTDIR=/scratch-shared/$USER/BONE-AI/results/pseudo_shape/$BACKGROUND_VAR

# N examples PER CLASS. 1 with the clinical set = 5 example turns, one per
# margin class. 2 = 10 turns; watch the prompt length and the memory note below.
NUM_FEW_SHOT=1

# Where the exemplars come from.
#   set  -> a DIFFERENT build (recommended): nothing is held out of $SHAPE_META,
#           so this run scores exactly the same images as the zero-shot run and
#           the two accuracies are directly comparable.
#   ""   -> draw exemplars from $SHAPE_META itself; those images are then held
#           out of scoring, so the denominator shrinks by NUM_FEW_SHOT*classes
#           and the comparison against zero-shot is no longer like-for-like.
# Point this at the EASIEST build you have -- an example should show the
# prototype, not an ambiguous case.
FEWSHOT_META=/scratch-shared/$USER/BONE-AI/pseudo_shape/$BACKGROUND_VAR/shape_256_cli_e/shape_metadata.csv

OUT=$OUTDIR/probe_results_cli_${LEVEL}_27b_fs${NUM_FEW_SHOT}.csv
# The zero-shot run to compare against; leave as-is to reuse run_shape_probe.sh's
# output. Skipped silently if it isn't there yet.
ZEROSHOT=$OUTDIR/probe_results_cli_${LEVEL}_27b.csv

NUM_SHARDS=2
# LOWER than the zero-shot job on purpose. Few-shot sends NUM_FEW_SHOT*classes+1
# images per prompt (6 at 1-shot clinical, not 1), so the same --batch-size is
# ~6x the pixels and KV cache. make_hf_generate halves the batch on OOM rather
# than dying, but every halving throws away the work already done on the failed
# attempt -- cheaper to start low and raise it after reading the img/s line.
BATCH_SIZE=8

cd "$REPO/working/vision_model/shape_probe"

mkdir -p "$OUTDIR"

# Preflight: environment, GPUs, and the two metadata files.
[[ -f "$REPO/pyproject.toml" ]] || {
    echo "FATAL: no pyproject.toml under $REPO -- uv would run outside the project" >&2
    exit 1
}
[[ -f "$SHAPE_META" ]] || { echo "FATAL: no shape metadata at $SHAPE_META" >&2; exit 1; }
if [[ -n "$FEWSHOT_META" && ! -f "$FEWSHOT_META" ]]; then
    echo "FATAL: FEWSHOT_META set but missing: $FEWSHOT_META" >&2
    echo "       Build it first, or set FEWSHOT_META=\"\" to draw exemplars from" >&2
    echo "       \$SHAPE_META (those images are then held out of scoring)." >&2
    exit 1
fi

N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if (( N_GPU < NUM_SHARDS )); then
    echo "FATAL: NUM_SHARDS=$NUM_SHARDS but this node has $N_GPU GPU(s)." >&2
    echo "       Set #SBATCH --gpus-per-node=$NUM_SHARDS with --nodes=1." >&2
    exit 1
fi
echo "$N_GPU GPU(s) on this node, running $NUM_SHARDS shard(s), ${NUM_FEW_SHOT}-shot per class"

uv run --no-sync python -c "import torch, transformers" || {
    echo "FATAL: the venv is not importable (mid-install, or wrong project root)" >&2
    exit 1
}

# --few-shot-metadata is only passed when FEWSHOT_META is non-empty; an empty
# string would be parsed as a path and fail.
FS_ARGS=(--num-few-shot "$NUM_FEW_SHOT")
[[ -n "$FEWSHOT_META" ]] && FS_ARGS+=(--few-shot-metadata "$FEWSHOT_META")

# Each shard writes $OUT with a .shard<i> suffix (run_shape_probe.py adds it when
# --num-shards > 1) -- they append concurrently, so they must not share a file.
# Exemplar selection is seeded and happens before sharding, so every shard shows
# the model the SAME examples and the shard CSVs stay concatenable.
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i uv run --no-sync python run_shape_probe.py --mode infer \
        --model-id "$MODEL" \
        --metadata "$SHAPE_META" \
        --batch-size $BATCH_SIZE \
        --num-shards $NUM_SHARDS --shard-index "$i" \
        "${FS_ARGS[@]}" \
        --out "$OUT" &
    pids+=($!)
done

# Wait on each pid INDIVIDUALLY. A bare `wait` (no arguments) always returns 0
# in bash -- it discards the children's exit statuses -- so `set -e` never fires
# and a dead shard is silently skipped.
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
(( rc == 0 )) || { echo "FATAL: a shard failed -- refusing to score partial results" >&2; exit 1; }

SHARDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    if (( NUM_SHARDS > 1 )); then SHARDS+=("${OUT%.csv}.shard${i}.csv"); else SHARDS+=("$OUT"); fi
done

echo "===== ${NUM_FEW_SHOT}-shot ====="
uv run --no-sync python run_shape_probe.py --mode eval --results "${SHARDS[@]}"

# The comparison this job exists for. eval breaks accuracy down by num_few_shot,
# so concatenating the zero-shot shards gives the contrast in one table.
ZS=()
for f in "$ZEROSHOT" "${ZEROSHOT%.csv}".shard*.csv; do [[ -f "$f" ]] && ZS+=("$f"); done
if (( ${#ZS[@]} )); then
    echo "===== zero-shot vs ${NUM_FEW_SHOT}-shot ====="
    uv run --no-sync python run_shape_probe.py --mode eval --results "${SHARDS[@]}" "${ZS[@]}"
else
    echo "no zero-shot results at $ZEROSHOT -- run jobs/run_shape_probe.sh for the contrast"
fi
