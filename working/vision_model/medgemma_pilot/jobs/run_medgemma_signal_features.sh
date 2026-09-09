#!/bin/bash
# Data-parallel MedGemma inference (LABELED mode) scoped to just the new
# per-sequence signal features added to feature_prompts.yaml /
# preprocess/feature_config.yaml: T1W_intensity, T1W_findings, T2W_intensity,
# T2W_pattern, T1W_post_contrast_enhancement.
#
# Sibling of run_medgemma_multigpu.sh: same infer -> aggregate flow, forced-
# choice labels (not free text), so it ends with `--mode aggregate` like that
# script rather than `--mode combine` like run_medgemma_freetext.sh.
#
# run_medgemma.py has no --features flag -- it processes whatever
# (case, feature_name) rows are in --metadata, intersected with the features
# defined in --config. So scoping to "just the new features" is done by
# filtering the metadata CSV down to those feature_name values BEFORE
# inference, rather than by passing a flag.
#
#SBATCH --job-name=run_medg_signal
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
# Regenerate this with preprocess/feature_config.yaml (now including the five
# signal-feature blocks) BEFORE running this job, so these feature_name rows
# actually exist in the metadata.
METADATA_FULL=/projects/prjs1779/BONE-AI/output/preprocess/shape_256_m/metadata_pilot40_all_fea.csv
FEATURES=(T1W_intensity T1W_fluid_level T2W_intensity T2W_pattern T1W_post_contrast_enhancement)

OUTDIR=/scratch-shared/$USER/BONE-AI/signal_features
METADATA=$OUTDIR/metadata_signal_features_${MODEL_NAME}.csv
OUT=$OUTDIR/results_${MODEL_NAME}.csv
NUM_SHARDS=1
BATCH_SIZE=32

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

# Filter the full metadata down to just the new signal features. Done here
# (not by re-running preprocess) so this job stays a pure inference step --
# preprocess's feature_config.yaml is the single source of truth for which
# crops exist; this script only picks which of those rows to score today.
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

# Also combine to one row per image (like run_medgemma_freetext.sh) -- this is
# the per-image CSV review_server.py wants, NOT results_sanity.csv above
# (that one is majority-voted per (case, feature) and has no image_path to
# render).
COMBINED=$OUTDIR/results_per_image.csv
uv run --no-sync python run_medgemma.py --mode combine \
    --inference-results "${SHARDS[@]}" \
    --out "$COMBINED"

# Sanity check: an empty/PARSE_FAILED parsed_label is how a truncated or
# malformed answer shows up in labeled mode (unlike free text, there is no
# prose to eyeball, so this is the only signal short of opening the CSV).
uv run --no-sync python - "$COMBINED" <<'PY'
import sys
import pandas as pd

df = pd.read_csv(sys.argv[1], dtype=str).fillna("")
bad = df["parsed_label"].isin(["", "PARSE_FAILED"])
print(f"labeled rows: {len(df)} | unparsed/failed: {int(bad.sum())}")
if bad.any():
    print("  by feature:")
    print(df.loc[bad, "feature_name"].value_counts().to_string())
    print("  ^ raise --max-new-tokens and re-run "
          "(inference is resume-safe, but delete the affected rows first)")
PY

echo "review with: python review_server.py --results $COMBINED --port 8000"
