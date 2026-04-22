#!/usr/bin/env Rscript
library("optparse")
option_list = list(
  # make_option(c("-d", "--debugging"), type="character", default="n",
  #             help="Debugging mode (only for local runs) [default= %default]", metavar="character"),
  # make_option(c("-w", "--class_weights"), type="character", default="y",
  #             help="Include class weights [default= %default]", metavar="character"),
  # make_option(c("-c", "--callbacks"), type="character", default="y",
  #             help="Include callabcks [default= %default]", metavar="character"),
  # make_option(c("-m", "--model"), type="integer", default="1", 
  #             help="model ID [default= %default]", metavar="integer"),
  make_option(c("-b", "--batch"), type="integer", default="1", 
              help="100k batch ID for rerunning missing models [default= %default]", metavar="integer")#,
  # make_option(c("-r", "--rerun"), type="character", default="n",
  #             help="Rerun missing models [default= %default]", metavar="character")
  
)
### Everything is set to incorporate step size tunning for sourceCpp if needed in the future 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

library(dplyr)
library(tidyr)
library(readr)

setwd("~/STA490---Fish_Classification")

nbatch <- opt$batch

models_per_batch <- 200
total_models <- 648

start_idx <- (nbatch - 1) * models_per_batch + 1
end_idx <- min(nbatch * models_per_batch, total_models)

if (start_idx > total_models) {
  stop("Batch index exceeds total number of models.")
}

batch_fitted_models <- as.integer(start_idx:end_idx)

input_dir <- file.path(
  "Models", "rnn", "drac", "training",
  "training_metrics_full",
  paste0("training_metrics_b", nbatch)
)

output_file <- file.path(
  "Models", "rnn", "drac", "training",
  "training_metrics_full",
  paste0("val_metrics_b", nbatch, ".rds")
)

cat("Working directory:\n")
print(getwd())

cat("Input directory exists:\n")
print(dir.exists(input_dir))

if (!dir.exists(input_dir)) {
  stop(paste("Input directory does not exist:", input_dir))
}

final_data <- tibble(
  val_loss = numeric(),
  best_epoch_loss = numeric(),
  val_auc = numeric(),
  model_id = integer()
)

t1 <- Sys.time()

for (i in batch_fitted_models) {
  infile <- file.path(input_dir, paste0("training_output_", i, ".rds"))
  
  if (file.exists(infile)) {
    aux_row <- readRDS(infile)
    
    aux_row <- tibble(
      val_loss = as.numeric(aux_row[1]),
      best_epoch_loss = as.numeric(aux_row[2]),
      val_auc = as.numeric(aux_row[3]),
      model_id = as.integer(aux_row[4])
    )
    
    final_data <- bind_rows(final_data, aux_row)
  }
}

t2 <- Sys.time()
print(t2 - t1)

cat("Number of rows collected:\n")
print(nrow(final_data))

saveRDS(final_data, file = output_file)

print(paste0("Output from training_metrics_b", nbatch, " processed!"))

