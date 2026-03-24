#!/bin/bash
#SBATCH --mem-per-cpu=2500M      # increase as needed
#SBATCH --time=0:180:00
#SBATCH --cpus-per-task=1

module load r/4.4

MODEL=$1

Rscript --vanilla cnn_hyperparameter_tuning_server.R -m $MODEL

