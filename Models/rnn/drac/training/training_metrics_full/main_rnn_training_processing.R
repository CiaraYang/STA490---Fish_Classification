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

setwd("../../../../../")

nbatch = opt$batch

{ 
  models_per_batch = 2000
  total_models = 14400
  
  start_idx = (nbatch - 1) * models_per_batch + 1
  end_idx = min(nbatch * models_per_batch, total_models)
  
  if(start_idx > total_models){
    stop("Batch index exceeds total number of models.")
  }
  
  batch_fitted_models = as.integer(start_idx:end_idx)
  mat_final_data = c(val_loss = 999, best_epoch_loss = -1, val_auc = -1, model_id = -1)
  
  t1 = Sys.time()
  
  for(i in batch_fitted_models){
    model_fitted = file.exists(paste0("Models/rnn/drac/training/training_metrics_full/training_metrics_b",nbatch,"/training_output_",i,".rds"))
    if(model_fitted){
      aux_row = readRDS(paste0("Models/rnn/drac/training/training_metrics_full/training_metrics_b",nbatch,"/training_output_",i,".rds"))
      mat_final_data = rbind(mat_final_data,aux_row)
      rm(aux_row)
    }
  }
  t2 = Sys.time()
  t2-t1
  
  colnames(mat_final_data) = c("val_loss","best_epoch_loss","val_auc","model_id")
  
  final_data = as_tibble(mat_final_data) %>% 
    mutate(model_id = as.integer(model_id)) %>% 
    filter(row_number() != 1)
  
}

saveRDS(final_data,file = paste0("Models/rnn/drac/training/training_metrics_full/val_metrics_b",nbatch,".rds"))

print(paste0("Output from the directory training_metrics_b",nbatch," processed !"))