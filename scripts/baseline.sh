#!/bin/bash
#SBATCH --job-name=unsup
#SBATCH --account=def-ssanner
#SBATCH --nodes=1
#SBATCH --gpus-per-node=v100l:4
#SBATCH --mem-per-gpu=24G
#SBATCH --time=03-00:00:00
#SBATCH --cpus-per-task=4
export OMP_NUM_THREADS=16

source /home/gfloto/env_store/diff_env/bin/activate
cd /home/gfloto/projects/def-ssanner/gfloto/projects/text_diff/scripts

module load gcc/9.3.0 arrow cuda/11

python -m torch.distributed.launch \
--nproc_per_node=4 \
--use_env run_train.py \
--name unsup \
--dataset_unsup wiki \
--folder_name /home/gfloto/scratch/diffusion_models \
--data_dir /home/gfloto/scratch/datasets/detox \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 400000 \
--save_interval 10000 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 128 \
--microbatch 32 \
--dataset detox \
--vocab bert \
--seq_len 64 \
--schedule_sampler lossaware \
--notes detox
#--resume_checkpoint /home/gfloto/scratch/diffusion_models/check_models/qqp_model000999.pt \
