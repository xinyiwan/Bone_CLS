#!/bin/bash
# Shape probe, DIAGNOSTIC arm: why does the clinical set fail?
#
# The generative runs (run_shape_probe.sh / _fewshot.sh) record one parsed label
# per image, which cannot tell "no idea" from "nearly said the right thing". This
# job answers that with two GPU passes over the SAME images, then scores both on
# CPU. See ../README_calibration.md for what each number means.
#
#   score  forced-choice log-probs of each label -> is it a THRESHOLD problem?
#   embed  pooled hidden states                  -> is the signal in the
#                                                   REPRESENTATION at all?
#
# Both are a single forward pass per image with no autoregressive decoding, so
# each is FASTER per image than --mode infer. The whole job is cheaper than the
# zero-shot generative run it explains.
#
# Run this BEFORE any LoRA experiment. It produces the calibrated baseline a
# fine-tune has to beat -- otherwise a gain that was really just bias correction
# gets credited to the fine-tune.
#
# GPUs: keep --nodes=1 and set --gpus-per-node = NUM_SHARDS below. sbatch runs
# this script on the FIRST allocated node only, so --nodes=2 does not give it a
# second GPU -- the extra node just sits idle while CUDA_VISIBLE_DEVICES=1
# selects a device that does not exist. That case does NOT crash: CUDA reports
# no devices, device_map="auto" quietly places the model on CPU, and the shard
# appears to hang instead of failing. Multi-node would need srun per node.
#
#SBATCH --job-name=shape_probe_lp
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --time=01:30:00
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

LOGP=$OUTDIR/logprobs_cli_${LEVEL}_27b.csv
LOGP_COT=$OUTDIR/logprobs_cli_${LEVEL}_27b_cot.csv
EMB=$OUTDIR/embeddings_cli_${LEVEL}_27b.npz

# The generative zero-shot run, used ONLY as the source of `thinking` text for
# the RUN_SCORE_COT pass. Same name run_shape_probe.sh writes.
ZEROSHOT=$OUTDIR/probe_results_cli_${LEVEL}_27b.csv

# --- which passes to run -------------------------------------------------
# Ordered by value per GPU-minute. RUN_SCORE alone already gives the pairwise
# AUC verdict and the calibrated baseline; the other two refine it.
RUN_SCORE=1      # forced-choice log-probs, no thinking block
RUN_SCORE_COT=1  # same, but scored after replaying $ZEROSHOT's thinking column
RUN_EMBED=1      # pooled hidden states for the linear probe

NUM_SHARDS=2

# LOWER than the generative job's 64, for two different reasons per mode:
#   score  the batch is expanded by the number of labels internally (3 rows per
#          image on a default clinical build), so peak memory is
#          BATCH_SCORE * n_labels rows.
#   embed  output_hidden_states materialises EVERY layer's activations for the
#          whole sequence at once -- on the 27B that is n_layers * seq * 5376
#          floats per row, far more than the forward pass itself. This is the
#          binding constraint in the job; raise it only after reading img/s.
BATCH_SCORE=16
BATCH_EMBED=4

# Hidden-state layers to pool. 'mid' is resolved against the loaded model's
# depth, not hardcoded, so this is correct for both the 4B and the 27B.
LAYERS=last,mid

cd "$REPO/working/vision_model/shape_probe"

mkdir -p "$OUTDIR"

# --- preflight ----------------------------------------------------------
# Two seconds here beats discovering a broken environment after SLURM has handed
# us the GPUs.
[[ -f "$REPO/pyproject.toml" ]] || {
    echo "FATAL: no pyproject.toml under $REPO -- uv would run outside the project" >&2
    exit 1
}
[[ -f "$SHAPE_META" ]] || { echo "FATAL: no shape metadata at $SHAPE_META" >&2; exit 1; }

N_GPU=$(nvidia-smi -L 2>/dev/null | wc -l)
if (( N_GPU < NUM_SHARDS )); then
    echo "FATAL: NUM_SHARDS=$NUM_SHARDS but this node has $N_GPU GPU(s)." >&2
    echo "       Set #SBATCH --gpus-per-node=$NUM_SHARDS with --nodes=1." >&2
    exit 1
fi
echo "$N_GPU GPU(s) on this node, running $NUM_SHARDS shard(s)"

uv run --no-sync python -c "import torch, transformers" || {
    echo "FATAL: the venv is not importable (mid-install, or wrong project root)" >&2
    exit 1
}
# calibrate.py is CPU-only but needs scikit-learn. Checked NOW rather than after
# the GPU passes, so a missing dep costs seconds instead of the whole allocation.
uv run --no-sync python -c "import sklearn" || {
    echo "FATAL: scikit-learn missing -- calibrate.py cannot score the GPU output." >&2
    echo "       It is in the root pyproject; run 'uv sync' in $REPO." >&2
    exit 1
}

if (( RUN_SCORE_COT )) && [[ ! -f "$ZEROSHOT" && ! -f "${ZEROSHOT%.csv}.shard0.csv" ]]; then
    echo "NOTE: no generative results at $ZEROSHOT -- skipping the thinking-replay pass."
    echo "      Run jobs/run_shape_probe.sh first if you want it."
    RUN_SCORE_COT=0
fi

# --- helpers ------------------------------------------------------------
# Shard file names. score_logprobs.py only adds the .shard<i> suffix when
# --num-shards > 1, so the single-shard case must use the base name unchanged.
# Fills the global array SHARD_FILES, rather than echoing a string: a path list
# expanded unquoted would word-split on any space in $OUTDIR, and quoting it
# would collapse the whole list into one argument.
shard_files() {   # $1 = base path, $2 = extension (csv|npz)
    local base=$1 ext=$2 i
    SHARD_FILES=()
    if (( NUM_SHARDS > 1 )); then
        for i in $(seq 0 $((NUM_SHARDS - 1))); do
            SHARD_FILES+=("${base%.$ext}.shard${i}.${ext}")
        done
    else
        SHARD_FILES=("$base")
    fi
}

# Fan a mode out across GPUs and wait. Each pid is waited on INDIVIDUALLY: a bare
# `wait` always returns 0 in bash -- it discards the children's exit statuses --
# so `set -e` never fires and a dead shard is silently skipped. That is how a job
# with a Python traceback in its log still gets reported COMPLETED.
fan_out() {       # $1 = label for the log, rest = args after --shard-index
    local label=$1; shift
    local pids=() i rc=0
    echo "===== GPU pass: $label ====="
    for i in $(seq 0 $((NUM_SHARDS - 1))); do
        CUDA_VISIBLE_DEVICES=$i uv run --no-sync python score_logprobs.py \
            --model-id "$MODEL" \
            --metadata "$SHAPE_META" \
            --num-shards $NUM_SHARDS --shard-index "$i" \
            "$@" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    (( rc == 0 )) || { echo "FATAL: a $label shard failed -- refusing to score partial results" >&2; exit 1; }
}

# --- GPU passes ---------------------------------------------------------
# NOTE: unlike run_shape_probe.py, score_logprobs.py is NOT resume-safe -- it
# truncates its output rather than appending. Re-running a pass redoes it in
# full. That is deliberate: these passes are ~1 forward per image, so a restart
# is cheap, and appending would risk silently mixing two configurations (with and
# without a thinking prefix, say) into one CSV.
if (( RUN_SCORE )); then
    fan_out "score (no thinking)" --mode score --batch-size $BATCH_SCORE --out "$LOGP"
fi

if (( RUN_SCORE_COT )); then
    # Replays the thinking text the generative run already produced, so the label
    # is scored in the context --mode infer actually had. Pass every zero-shot
    # shard: score_logprobs.py builds one image_path -> thinking map from them.
    ZS=()
    for f in "$ZEROSHOT" "${ZEROSHOT%.csv}".shard*.csv; do [[ -f "$f" ]] && ZS+=("$f"); done
    # --thinking-from takes ONE csv, so concatenate the shards first if needed.
    THINK_SRC=$ZEROSHOT
    if (( ${#ZS[@]} > 1 )); then
        THINK_SRC=$OUTDIR/.thinking_src_cli_${LEVEL}_27b.csv
        uv run --no-sync python -c "
import sys, pandas as pd
pd.concat([pd.read_csv(p) for p in sys.argv[2:]], ignore_index=True).to_csv(sys.argv[1], index=False)
" "$THINK_SRC" "${ZS[@]}"
    elif (( ${#ZS[@]} == 1 )); then
        THINK_SRC=${ZS[0]}
    fi
    fan_out "score (after replayed thinking)" --mode score --batch-size $BATCH_SCORE \
        --thinking-from "$THINK_SRC" --out "$LOGP_COT"
fi

if (( RUN_EMBED )); then
    # Zero-shot only, and the script enforces it: few-shot puts several images in
    # context, so pooling "the image tokens" would mix exemplars into the query
    # vector. The probe measures the representation, not the prompt.
    # --layers=... and NOT --layers ...: the value starts with '-' (as in
    # "-1,mid"), and argparse treats a leading dash as an option name unless the
    # token parses as a negative number, which "-1,mid" does not. The = form is
    # the only one that survives. 'last' is accepted as a dash-free alias.
    fan_out "embed (hidden states)" --mode embed --batch-size $BATCH_EMBED \
        "--layers=$LAYERS" --out "$EMB"
fi

# --- CPU scoring --------------------------------------------------------
# Read the pairwise AUC FIRST in each block below. It is printed before any
# calibration because it is the one number a bad threshold cannot spoil: it asks
# only whether the score RANKS true irregulars above true lobulateds.
#   AUC ~0.5  -> no signal; neither calibration nor a bias-only LoRA can help,
#                and the next move is the input pipeline (crop tighter, upscale).
#   AUC ~0.8 with ~0.02 recall -> the whole deficit is the threshold.
if (( RUN_SCORE )); then
    echo "===== calibration: forced choice, no thinking ====="
    shard_files "$LOGP" csv
    uv run --no-sync python calibrate.py --geometry \
        --logprobs "${SHARD_FILES[@]}"
fi

if (( RUN_SCORE_COT )); then
    echo "===== calibration: forced choice AFTER replayed thinking ====="
    echo "      (gap vs the block above = what the chain of thought is worth)"
    shard_files "$LOGP_COT" csv
    uv run --no-sync python calibrate.py \
        --logprobs "${SHARD_FILES[@]}"
fi

if (( RUN_EMBED )); then
    echo "===== linear probe on frozen hidden states ====="
    echo "      the lobulated-vs-irregular binary is the number that matters:"
    echo "      >=0.80 features present, readout broken -> LoRA should pay off"
    echo "      ~0.50  absent at this resolution        -> fix inputs, do NOT fine-tune yet"
    shard_files "$EMB" npz
    uv run --no-sync python calibrate.py \
        --embeddings "${SHARD_FILES[@]}"
fi
