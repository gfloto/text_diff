python -m torch.distributed.launch --nproc_per_node=1 --master_port=12233 --use_env run_train.py \
--diff_steps 2000 \
--lr 0.0001 \
--learning_steps 10000 \
--save_interval 5 \
--seed 102 \
--noise_schedule sqrt \
--hidden_dim 128 \
--bsz 2048 \
--microbatch 256 \
--dataset detox \
--data_dir datasets/detox \
--vocab bert \
--seq_len 64 \
--schedule_sampler lossaware \
--notes detox \
--folder_name  /home/griffin/scratch/diffusion_models

#--resume_checkpoint /home/griffin/scratch/check_models/qqp/model000999.pt \