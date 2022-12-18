#!/bin/bash
#SBATCH --job-name=dec_15_small
#SBATCH --account=def-ssanner
#SBATCH --gpus-per-node=v100l:1
#SBATCH --time=00-00:00:10
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4

source /home/gfloto/env_store/diff_env/bin/activate
cd /home/gfloto/projects/def-ssanner/gfloto/projects/diffuseq/scripts

module load gcc/9.3.0 arrow cuda/11

python -m torch.distributed.launch \
--nproc_per_node=1 \
--master_port=12233 \
--use_env run_train.py \
--data_dir /home/gfloto/projects/def-ssanner/gfloto/projects/diffuseq/datasets/detox \
--resume_checkpoint /home/gfloto/scratch/diffusion_models/check_models/qqp_model000999.pt \
--folder_name /home/gfloto/scratch/diffusion_models \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 100 \
--save_interval 10 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 2048 \
--microbatch 256 \
--dataset detox \
--vocab bert \
--seq_len 64 \
--schedule_sampler lossaware \
--notes detox
