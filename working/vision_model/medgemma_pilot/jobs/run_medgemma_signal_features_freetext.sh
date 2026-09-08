#!/bin/bash
# Free-text-with-RANKING inference for the five new per-sequence signal
# features (T1W_intensity, T1W_findings, T2W_intensity, T2W_pattern,
# T1W_post_contrast_enhancement): the model reasons in prose under LESION /
# OBSERVATIONS / REASONING, then ends on an ASSESSMENT ranking over that
# feature's label_options -- see run_medgemma_freetext.sh, which this is a
# feature-scoped sibling of.
#
# Why ranked, not label, for these: the labeled run (run_medgemma_signal_features.sh)
# came back with reasons that were too terse to tell a real read from a guess.
# Ranked mode keeps the forced vocabulary (so it is still scorable, unlike bare
# free_text) but forces the REASONING to be written out before the answer, and
# the LESION heading is a built-in grounding check -- it makes the model say
# whether it can even locate the lesion before describing it, which is the
# "does it actually see the feature" check. This is not a YAML setting; it's
# baked into prompts.free_text_format for every feature.
#
# Same reason this ends in `--mode combine` (one row per image) rather than
# `--mode aggregate` (majority vote): a ranking isn't a single label to vote
# over, and review_server.py needs the per-image rows anyway.
#
#SBATCH --job-name=run_medgemma_signal_rank
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

MODEL_NAME=27b
MODEL=/scratch-shared/$USER/models/medgemma-${MODEL_NAME}-it
# Regenerate this with preprocess/feature_config.yaml (which now includes the
# five signal-feature blocks) BEFORE running this job.
METADATA_FULL=/projects/prjs1779/BONE-AI/output/preprocess/metadata.csv
FEATURES=(T1W_intensity T1W_findings T2W_intensity T2W_pattern T1W_post_contrast_enhancement)

OUTDIR=/scratch-shared/$USER/BONE-AI/freetext/signal_features
METADATA=$OUTDIR/metadata_signal_features_${MODEL_NAME}.csv
OUT=$OUTDIR/freetext_slice_${MODEL_NAME}.csv
NUM_SHARDS=1

# Lower than the label run's 32. The reply is prose across four headings, not
# one JSON line, so sequences are far longer and a static batch costs its
# SLOWEST member -- a big batch spends most of its time padding.
BATCH_SIZE=24
# Raised from the label run's default: prose overruns 1024, and a truncated
# answer is indistinguishable from a terse one when read by eye.
MAX_NEW_TOKENS=2048

REPETITION_PENALTY=1.1
NO_REPEAT_NGRAM_SIZE=0

mkdir -p "$OUTDIR"
cd "$REPO/working/vision_model/medgemma_pilot"
uv sync

# Preflight: two seconds here beats discovering a broken environment after SLURM
# has handed us the GPU. An interactive `uv add` that overlaps a job start
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

# Filter the full metadata down to just the five signal features (same
# approach as run_medgemma_signal_features.sh) -- run_medgemma.py has no
# --features flag, it processes whatever feature_name rows are in --metadata.
uv run --no-sync python - "$METADATA_FULL" "$METADATA" "${FEATURES[@]}" <<'PY'
import sys
import pandas as pd

src, dst, *features = sys.argv[1:]
df = pd.read_csv(src, dtype=str)
keep = df[df["feature_name"].isin(features)]
if keep.empty:
    raise SystemExit(
        f"FATAL: no rows in {src} match feature_name in {features} -- "
        "did you regenerate metadata with the updated feature_config.yaml?"
    )
keep.to_csv(dst, index=False)
print(f"kept {len(keep)}/{len(df)} rows for features: {sorted(keep['feature_name'].unique())}")
PY

# Each shard writes $OUT with a .shard<i> suffix (run_medgemma.py adds it when
# --num-shards > 1) -- they append concurrently, so they must not share a file.
#
# No --num-few-shot: run_medgemma.py rejects it under output-mode free_text,
# and ranked few-shot exemplars would need hand-written LESION/OBSERVATIONS/
# REASONING prose per feature, which none of the five yaml blocks have yet.
pids=()
for i in $(seq 0 $((NUM_SHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$i uv run --no-sync python run_medgemma.py --mode infer \
        --output-mode ranked \
        --model-id "$MODEL" \
        --metadata "$METADATA" \
        --config feature_prompts.yaml \
        --batch-size $BATCH_SIZE \
        --max-new-tokens $MAX_NEW_TOKENS \
        --repetition-penalty $REPETITION_PENALTY \
        --no-repeat-ngram-size $NO_REPEAT_NGRAM_SIZE \
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

COMBINED=$OUTDIR/freetext_slice_all_signal_features_${MODEL_NAME}.csv
uv run --no-sync python run_medgemma.py --mode combine \
    --inference-results "${SHARDS[@]}" \
    --out "$COMBINED"

# Truncation is the one failure this arm cannot spot by eye, because a cut-off
# answer reads like a short one. In RANKED mode the reply is required to end
# on the bare "ASSESSMENT: a > b > c" line, so an unparsed `ranking` is the
# real truncation signal (every valid answer ends mid-word or on that line,
# never a full stop, so a punctuation check would flag every row).
uv run --no-sync python - "$COMBINED" <<'PY'
import sys, pandas as pd
df = pd.read_csv(sys.argv[1], dtype=str).fillna("")
bad = df["ranking"].str.strip() == "" if "ranking" in df.columns else df["reason"].str.strip() == ""
print(f"free-text/ranked rows: {len(df)} | no ranking parsed: {int(bad.sum())}")
if bad.any():
    print("  by feature:")
    print(df.loc[bad, "feature_name"].value_counts().to_string())
    print("  ^ likely ranking line missing or malformed -- raise --max-new-tokens and re-run "
          "(inference is resume-safe, but delete the affected rows first)")
PY

echo "review with: python review_server.py --results $COMBINED --port 8000"
