#!/usr/bin/env Rscript
library("optparse")

option_list = list(
  make_option(c("-b", "--batch"), type="integer", default="1", 
              help="100k batch ID for rerunning missing models [default= %default]", 
              metavar="integer")
)

opt_parser = OptionParser(option_list = option_list)
opt = parse_args(opt_parser)

library(dplyr)
library(tidyr)
library(readr)
library(tibble)

setwd("../../../../../")

nbatch = opt$batch

batch_fitted_models = as.integer(1:100000 + 100000 * (nbatch - 1))

t1 = Sys.time()

results_list <- list()

for (i in batch_fitted_models) {
  file_path <- paste0(
    "Models/dnn/drac/training/training_metrics_b",
    nbatch,
    "/training_output_",
    i,
    ".rds"
  )
  
  if (file.exists(file_path)) {
    aux_row <- readRDS(file_path)
    results_list[[length(results_list) + 1]] <- as.data.frame(t(aux_row))
  }
}

t2 = Sys.time()
print(t2 - t1)

if (length(results_list) == 0) {
  stop(paste0("No .rds files found for batch ", nbatch))
}

final_data <- bind_rows(results_list) %>%
  mutate(model_id = as.integer(model_id))

saveRDS(
  final_data,
  file = paste0("Models/dnn/drac/training/training_metrics_full/val_metrics_b", nbatch, ".rds")
)

print(paste0("Output from the directory training_metrics_b", nbatch, " processed!"))