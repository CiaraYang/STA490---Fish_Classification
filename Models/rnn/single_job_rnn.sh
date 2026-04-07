#!/bin/bash
#SBATCH --mem-per-cpu=2500M      # increase as needed
#SBATCH --time=0:120:00
#SBATCH --cpus-per-task=1

module load r/4.4

MODEL=$1

Rscript --vanilla rnn_hyperparameter_tuning_server.R -m $MODEL

