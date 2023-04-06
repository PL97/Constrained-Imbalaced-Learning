#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=40g
#SBATCH --tmp=40g
#SBATCH --mail-type=NONE  
#SBATCH --mail-user=peng0347@umn.edu 

##SBATCH -p v100                                   
##SBATCH --gres=gpu:v100:1

#SBATCH --gres=gpu:a100:1
#SBATCH -p a100-4
##SBATCH -p apollo_agate

module load parallel

cd /home/jusun/peng0347/Constrained-Imbalaced-Learning
export PATH=/home/jusun/peng0347/miniconda3/envs/tf_opt/bin:$PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jusun/peng0347/miniconda3/envs/tf_opt/
CUDA_VISIBLE_DEVICES=0 python main.py --run_name constrained_IL\
                --dataset breast-cancer-wisc \
                --model MLP \
                --batch_size 8000 \
                --learning_rate 5e-4 \
                --learning_rate_s 5e-2\
                --method AL_OAP \
                --subprob_max_epoch 100 \
                --rounds 20 \
                --alpha 0.99 \
                --t 0.5 \
                --solver AdamW \
                --warm_start 10 \
                --rho 1 \
                --delta 1 \
                --sto 1 \
                --saved_fig_name 01_ws \
                --reg 0.1

## example: sbatch AL.slurm AL_FPOR breast-cancer-wisc-diag 20