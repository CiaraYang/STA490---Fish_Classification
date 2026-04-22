#!/usr/bin/env Rscript
library("optparse")
option_list = list(
  make_option(c("-d", "--debugging"), type="character", default="n",
              help="Debugging mode (only for local runs) [default= %default]", metavar="character"),
  # make_option(c("-w", "--class_weights"), type="character", default="y",
  #             help="Include class weights [default= %default]", metavar="character"),
  # make_option(c("-c", "--callbacks"), type="character", default="y",
  #             help="Include callabcks [default= %default]", metavar="character"),
  make_option(c("-m", "--model"), type="integer", default="1", 
              help="model ID [default= %default]", metavar="integer"),
  make_option(c("-b", "--batch"), type="integer", default="1", 
              help="100k batch ID for rerunning missing models [default= %default]", metavar="integer"),
  make_option(c("-r", "--rerun"), type="cypharacter", default="n",
              help="Rerun missing models [default= %default]", metavar="character")
  
)
### Everything is set to incorporate step size tunning for sourceCpp if needed in the future 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

model_id = opt$model
debug_mode = opt$debugging == "y"
rerun_model = opt$rerun == "y"

setwd("../../")

Sys.setenv(UV_OFFLINE=1)



# =========================================================
# 02_rnn_hyperparameter_tuning.R
# RNN hyperparameter tuning for current dataset
# =========================================================

library(dplyr)
library(tensorflow)
library(keras3)

if (debug_mode) {
  library(reticulate)
  use_python("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3", required = TRUE)
}

x_train = readRDS("Data/RNN_data/x_train.rds")
x_validate = readRDS("Data/RNN_data/x_validate.rds")
x_test = readRDS("Data/RNN_data/x_test.rds")

y_train = readRDS("Data/RNN_data/y_train.rds")
# y_validate = readRDS("Data/RNN_data/y_validate.rds")
# y_test = readRDS("Data/RNN_data/y_test.rds")
# 
dummy_y_train = readRDS("Data/RNN_data/dummy_y_train.rds")
dummy_y_val = readRDS("Data/RNN_data/dummy_y_val.rds")
# dummy_y_test = readRDS("Data/RNN_data/dummy_y_test.rds")


set.seed(15)

# -------------------------
# Check input shape
# -------------------------
print(dim(x_train))
print(dim(x_validate))

input_shape_use <- c(dim(x_train)[2], dim(x_train)[3]) # should be 5 and 91

# -------------------------
# Class weights
# -------------------------
class_counts <- table(y_train)
print(class_counts)

cw <- as.numeric(class_counts["Alewife"] / class_counts["Rainbow Smelt"])
class_weight_list <- list("0" = 1, "1" = cw)

print(class_weight_list)

# -------------------------
# Early stopping
# -------------------------
callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
    min_delta = 1e-2,
    patience = 25,
    restore_best_weights = TRUE
  )
)

# -------------------------
# Hyperparameter grid
# -------------------------
lstmunits <- c(32, 64, 128)
neuron1   <- c(32, 64, 128)
batchsize <- c(32, 64, 128)
lr        <- c(1e-4, 5e-4, 1e-3)
dropout1  <- c(0.0, 0.2)
regrate   <- c(1e-6, 1e-5)
use_batchnorm <- c(TRUE, FALSE)

grid.search.full <- expand.grid(
  lstmunits = lstmunits,
  neuron1 = neuron1,
  batchsize = batchsize,
  lr = lr,
  dropout1 = dropout1,
  regrate = regrate,
  use_batchnorm = use_batchnorm
)

saveRDS(grid.search.full, "Models/rnn/grid.search.full.rds")

## ---- Storage ----
val_loss <- rep(NA, 1)
best_epoch_loss <- rep(NA, 1)
val_auc <- rep(NA, 1)

print(paste0("Processing Model #", model_id))

t1 = Sys.time()

set_random_seed(15)

rnn <- keras_model_sequential() %>%
  layer_lstm(
    input_shape = input_shape_use,
    units = grid.search.full$lstmunits[model_id]
  ) %>%
  layer_activation_leaky_relu()

if (grid.search.full$use_batchnorm[model_id]) {
  rnn <- rnn %>% layer_batch_normalization()
}

if (grid.search.full$dropout1[model_id] > 0) {
  rnn <- rnn %>% layer_dropout(rate = grid.search.full$dropout1[model_id])
}

rnn <- rnn %>%
  layer_dense(
    units = grid.search.full$neuron1[model_id],
    activity_regularizer = regularizer_l2(l = grid.search.full$regrate[model_id])
  ) %>%
  layer_activation_leaky_relu() %>%
  layer_dense(units = 2, activation = "softmax")

rnn %>% compile(
  optimizer = optimizer_adam(learning_rate = grid.search.full$lr[model_id]),
  loss = loss_categorical_crossentropy(),
  metrics = c("accuracy", metric_auc(name = "auc"))
)

rnn_history <- rnn %>% fit(
  x_train, dummy_y_train,
  batch_size = grid.search.full$batchsize[model_id],
  epochs = 200,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = class_weight_list,
  callbacks = callbacks,
  verbose = 2
)

val_loss[1] <- min(rnn_history$metrics$val_loss)
best_epoch_loss[1] <- which.min(rnn_history$metrics$val_loss)
val_auc[1] <- max(rnn_history$metrics$val_auc)

t2 = Sys.time()
print(t2 - t1)

final_output = c(val_loss, best_epoch_loss, val_auc, model_id)

nbatch = (model_id - 1) %/% 200 + 1
saveRDS(
  final_output,
  file = paste0("Models/rnn/drac/training/training_metrics_b", nbatch, "/training_output_", model_id, ".rds")
)

print(paste0("training metrics for rnn with parameter configuration ", model_id, " ready!"))