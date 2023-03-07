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
##SBATCH -p a100-4
#SBATCH -p apollo_agate

module load parallel

cd /home/jusun/peng0347/Constrained-Imbalaced-Learning
export PATH=/home/jusun/peng0347/miniconda3/envs/tf_opt/bin:$PATH
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/jusun/peng0347/miniconda3/envs/tf_opt/
sh run.sh $1 $2 $3

## example: sbatch AL.slurm AL_FPOR breast-cancer-wisc-diag 20