#!/bin/bash
#SBATCH --mem-per-cpu=2500M
#SBATCH --time=0:180:00
#SBATCH --cpus-per-task=1
#SBATCH --array=1-216%500

module load r/4.4

Rscript --vanilla dnn_hyperparameter_tuning_server.R -m $SLURM_ARRAY_TASK_ID
