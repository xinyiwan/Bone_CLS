#!/bin/bash
#SBATCH --job-name=preprocess_2dslicer
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=192
#SBATCH --time=01:00:00
#SBATCH --output=/projects/prjs1779/BONE-AI/logs/out/slurm-%x-%j.out
#SBATCH --error=/projects/prjs1779/BONE-AI/logs/err/slurm-%x-%j.err

set -euo pipefail

# ---------------------------------------------------------------- paths ----
PROJECT_ROOT=${PROJECT_ROOT:-/projects/prjs1779/BONE-AI}
REPO_ROOT=${REPO_ROOT:-${PROJECT_ROOT}/Bone_CLS}

DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/subdata}
OUT_ROOT=${OUT_ROOT:-${PROJECT_ROOT}/output/preprocess/shape_256_stack}
SEQUENCE_TABLE=${SEQUENCE_TABLE:-${DATA_ROOT}/case_metadata.csv}
CONFIG=${CONFIG:-${REPO_ROOT}/working/vision_model/preprocess/feature_config_stack.yaml}
LABELS_DIR=${PROJECT_ROOT}/output/label_out/jsons

# -------------------------------------------------------------- options ----
OUT_SIZE=${OUT_SIZE:-256}
# extra flags, override to e.g. EXTRA_ARGS="--overlay" for a real extraction run
EXTRA_ARGS=${EXTRA_ARGS:---overlay --save-mask}

# ---------------------------------------------------------- environment ----
export UV_CACHE_DIR=${UV_CACHE_DIR:-${PROJECT_ROOT}/.uv-cache}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}
export MKL_NUM_THREADS=${OMP_NUM_THREADS}
export PYTHONUNBUFFERED=1

cd "${REPO_ROOT}"
mkdir -p "${OUT_ROOT}"

echo "[$(date -Is)] host=$(hostname) job=${SLURM_JOB_ID:-local}"
echo "  DATA_ROOT      = ${DATA_ROOT}"
echo "  OUT_ROOT       = ${OUT_ROOT}"
echo "  SEQUENCE_TABLE = ${SEQUENCE_TABLE}"
echo "  CONFIG         = ${CONFIG}"
echo "  OUT_SIZE       = ${OUT_SIZE}"
echo "  EXTRA_ARGS     = ${EXTRA_ARGS}"
echo "  LABELS_DIR     = ${LABELS_DIR}"

# ------------------------------------------------------------------ run ----
uv run python3 working/vision_model/preprocess/run.py \
    --data-root "${DATA_ROOT}" \
    --out-root "${OUT_ROOT}" \
    --sequence-table "${SEQUENCE_TABLE}" \
    --config "${CONFIG}" \
    --out-size "${OUT_SIZE}" \
    --labels-dir  "${LABELS_DIR}" \
    ${EXTRA_ARGS}

echo "[$(date -Is)] done"
