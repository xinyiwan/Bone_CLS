#!/bin/bash
# Data-parallel MedGemma inference in FREE-TEXT mode: no label vocabulary is
# offered, the model describes the feature in prose, and the prose is read by a
# human afterwards.
#
# Sibling of run_medgemma_multigpu.sh, and deliberately NOT a flag on it: the
# two runs end differently. That script finishes with `--mode aggregate`, which
# majority-votes per-image labels into one label per (case, feature). There are
# no labels here, so aggregating would vote over empty strings and write a
# results_sanity.csv that looks valid and means nothing. This one ends with
# `--mode combine`, which merges the shards while KEEPING one row per image --
# the form review_server.py reads.
#
#SBATCH --job-name=run_medgemma_rank
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
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
# The pilot subset, not the full metadata: this arm is graded by hand, so the
# run size is bounded by how much prose you are willing to read (~40 cases).
METADATA=/projects/prjs1779/BONE-AI/output/preprocess/shape_256_stack/metadata_subset.csv
OUTDIR=/scratch-shared/$USER/BONE-AI/stack/rank
OUT=$OUTDIR/freetext_slice.csv
NUM_SHARDS=1

# Lower than the label run's 32. There the answer is one JSON line; here it is
# two prose paragraphs, so sequences are far longer and a static batch costs its
# SLOWEST member -- a big batch spends most of its time padding.
BATCH_SIZE=1
# Also raised: 1024 was sized for a thinking block plus a one-sentence `reason`.
# Prose overruns it, and a truncated answer is indistinguishable from a terse
# one when you are reading them by hand.
MAX_NEW_TOKENS=2048

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
#
# No --num-few-shot: run_medgemma.py rejects it in this mode. The exemplars
# answer with a label in JSON, which would teach back the terse forced-choice
# format that free text exists to avoid.
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i uv run --no-sync python run_medgemma.py --mode infer \
        --input-mode stack \
        --output-mode ranked \
        --model-id "$MODEL" \
        --metadata "$METADATA" \
        --config feature_prompts.yaml \
        --batch-size $BATCH_SIZE \
        --max-new-tokens $MAX_NEW_TOKENS \
        --num-shards $NUM_SHARDS --shard-index "$i" \
        --out "$OUT" &
    pids+=($!)
done

# Wait on each pid INDIVIDUALLY. A bare `wait` (no arguments) always returns 0
# in bash -- it discards the children's exit statuses -- so `set -e` never fires
# and a dead shard is silently skipped. That is how a job with a Python
# traceback in its log still gets reported COMPLETED, and how a review pass over
# partial shards gets read as if it were the full run.
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
(( rc == 0 )) || { echo "FATAL: a shard failed -- refusing to combine partial results" >&2; exit 1; }

SHARDS=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    # run_medgemma.py only adds the .shard<i> suffix when --num-shards > 1, so
    # the single-shard case must use $OUT unchanged -- otherwise combine is
    # handed a path that was never written.
    if (( NUM_SHARDS > 1 )); then SHARDS+=("${OUT%.csv}.shard${i}.csv"); else SHARDS+=("$OUT"); fi
done

COMBINED=$OUTDIR/freetext_slice_all.csv
uv run --no-sync python run_medgemma.py --mode combine \
    --inference-results "${SHARDS[@]}" \
    --out "$COMBINED"

# Truncation is the one failure this arm cannot spot by eye, because a cut-off
# answer reads like a short one. Count rows whose prose ends mid-sentence and
# say so in the log rather than leaving it to be noticed during the manual read.
uv run --no-sync python - "$COMBINED" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1], dtype=str).fillna("")
prose = df["reason"].str.strip()
empty = int((prose == "").sum())

# In RANKED mode the reply is required to end on the bare "ASSESSMENT: a > b > c"
# line, so no answer ends in a full stop -- the punctuation test would flag every
# row. There, an unparsed `ranking` is the real truncation signal.
ranked = "ranking" in df.columns and (df["ranking"].str.strip() != "").any()
if ranked:
    bad = int((df["ranking"].str.strip() == "").sum())
    label, hint = "no ranking parsed", "ranking line missing or malformed"
else:
    bad = int((prose != "").sum() - prose[prose != ""].str.endswith((".", "!", "?")).sum())
    label, hint = "not ending in terminal punctuation", "truncated or refused"
print(f"free-text rows: {len(df)} | empty: {empty} | {label}: {bad}")
if empty or bad:
    print(f"  ^ likely {hint} -- raise --max-new-tokens and re-run "
          "(inference is resume-safe, but delete the affected rows first)")
PY

echo "review with: python review_server.py --results $COMBINED --port 8000"
