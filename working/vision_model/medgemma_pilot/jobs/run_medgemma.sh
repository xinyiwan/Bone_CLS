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



export UV_CACHE_DIR=/projects/prjs1779/BONE-AI/.uv-cache
export HF_HOME=/scratch-shared/$USER/hf-cache

cd Bone_CLS/working/vision_model/medgemma_pilot
uv run --no-sync python run_medgemma.py --mode infer \
    --model-id /scratch-shared/$USER/models/medgemma-1.5-4b-it \
    --metadata /projects/prjs1779/BONE-AI/output/preprocess/metadata.csv \
    --config feature_prompts.yaml \
    --out /scratch-shared/$USER/BONE-AI/results.csv