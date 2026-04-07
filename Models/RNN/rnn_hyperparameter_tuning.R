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
  make_option(c("-r", "--rerun"), type="character", default="n",
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
regrate <- c(1e-6, 1e-5, 1e-4)
lstmunits <- c(256, 128, 64)
neuron1 <- c(256, 128, 64, 32, 16)
batchsize <- c(100, 500, 1000, 1500)

grid.search.full <- expand.grid(
  regrate = regrate,
  lstmunits = lstmunits,
  neuron1 = neuron1,
  batchsize = batchsize
)

# set.seed(15)
# x <- sample(1:nrow(grid.search.full), 20, replace = FALSE)
# grid.search.subset <- grid.search.full[x, ]

# val_loss <- vector(length = nrow(grid.search.subset))
# best_epoch <- vector(length = nrow(grid.search.subset))
# val_auc <- vector(length = nrow(grid.search.subset))

## ---- Storage ----
val_loss <- rep(NA, 1)
best_epoch_loss <- rep(NA, 1)
val_auc <- rep(NA, 1)

print(paste0("Processing Model #", model_id))

t1 = Sys.time()

# choose which models to run this time
# Do this for 1:10 then 11:20 because it takes too long on its own
# model_indices <- 11:20 # Change to 11:20 later

set_random_seed(15)

rnn <- keras_model_sequential() %>%
  layer_lstm(
    input_shape = input_shape_use,
    units = grid.search.full$lstmunits[model_id]
  ) %>%
  layer_activation_leaky_relu() %>%
  layer_batch_normalization() %>%
  layer_dense(
    units = grid.search.full$neuron1[model_id],
    activity_regularizer = regularizer_l2(l = grid.search.full$regrate[model_id])
  ) %>%
  layer_activation_leaky_relu() %>%
  layer_dense(units = 2, activation = "softmax")

rnn %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-4),
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

nbatch = (model_id - 1) %/% 100000 + 1
saveRDS(
  final_output,
  file = paste0("Models/rnn/drac/training/training_metrics_b", nbatch, "/training_output_", model_id, ".rds")
)

print(paste0("training metrics for cnn with parameter configuration ", model_id, " ready!"))

