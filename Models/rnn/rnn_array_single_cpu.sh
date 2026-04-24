#!/bin/bash
#SBATCH --mem-per-cpu=2500M      # increase as needed
#SBATCH --time=0:120:00
#SBATCH --cpus-per-task=1
#SBATCH --array=0-179%500

BATCH=$1

module load r/4.4

Rscript --vanilla rnn_hyperparameter_tuning_server.R -m $(($SLURM_ARRAY_TASK_ID + 1 + $BATCH*200  ))
