#!/bin/bash

#SBATCH -p volta-gpu
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=1G
#SBATCH -t 00:04:00
#SBATCH --qos gpu_access
#SBATCH --gres=gpu:1
#SBATCH --output=results/output/slurm-%x-%j.out

PARAMETER_ID=$1
REALIZATION=$2
SEED=$3

module add python/3.9.6

python3 run_sim_slurm.py --parameter_id=$PARAMETER_ID --realization=$REALIZATION --seed=$SEED