#!/bin/bash

# slurm stuff goes here

PARAMETER_ID=$1
REALIZATION=$2

python3 rum_sim_slurm.py --parameter_id=$PARAMETER_ID --realization=$REALIZATION