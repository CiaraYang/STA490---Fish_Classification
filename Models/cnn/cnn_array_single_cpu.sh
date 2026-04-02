#!/bin/bash
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=0:180:00
#SBATCH --cpus-per-task=1
#SBATCH --array=0-9999%500

#SBATCH --output=Models/cnn/drac/logs/lg_%A_%a.out
#SBATCH --error=Models/cnn/drac/logs/lg_%A_%a.err

BATCH=$1

module load r/4.4

Rscript --vanilla cnn_hyperparameter_tuning_server.R \
  -m $(($SLURM_ARRAY_TASK_ID + 1 + $BATCH*10000))