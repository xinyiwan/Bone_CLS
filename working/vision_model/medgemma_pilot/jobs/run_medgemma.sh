#!/bin/bash
#SBATCH --job-name=run_medgemma
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
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

# --batch-size: images per GPU call. 1 = the old row-by-row behaviour, which
# leaves the A100 mostly idle (a 4B model decoding one sequence is memory-
# bandwidth bound). 16 is a safe start for the 4B in bf16 on an 80GB card; the
# script halves the batch automatically if it ever OOMs. Check the "img/s" line
# in the log and tune from there.
cd "$REPO/working/vision_model/medgemma_pilot"
uv run --no-sync python run_medgemma.py --mode infer \
    --model-id /scratch-shared/$USER/models/medgemma-1.5-4b-it \
    --metadata /projects/prjs1779/BONE-AI/output/preprocess/metadata.csv \
    --config feature_prompts.yaml \
    --batch-size 16 \
    --out /scratch-shared/$USER/BONE-AI/results.csv