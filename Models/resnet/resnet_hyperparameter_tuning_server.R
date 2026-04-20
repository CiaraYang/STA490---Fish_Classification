#!/usr/bin/env Rscript

library(optparse)

option_list <- list(
  make_option(c("-d", "--debugging"), type = "character", default = "n",
              help = "Debugging mode for local runs [default = %default]", metavar = "character"),
  make_option(c("-m", "--model"), type = "integer", default = 1,
              help = "Model ID (row index of full hyperparameter grid) [default = %default]", metavar = "integer"),
  make_option(c("-b", "--batch"), type = "integer", default = 1,
              help = "100k batch ID for reruns [default = %default]", metavar = "integer"),
  make_option(c("-r", "--rerun"), type = "character", default = "n",
              help = "Rerun missing models [default = %default]", metavar = "character")
)

### Everything is set to incorporate step size tunning for sourceCpp if needed in the future 
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

model_id <- opt$model
debug_mode <- opt$debugging == "y"
rerun_model <- opt$rerun == "y"

setwd("../../")
Sys.setenv(UV_OFFLINE = 1)

# =========================================================
# 02_tune_resnet.R
# Hyperparameter tuning for 1D ResNet
# =========================================================

library(dplyr)
library(tensorflow)
library(keras3)

if (debug_mode) {
  library(reticulate)
  use_python("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3", required = TRUE)
}

set.seed(15)
options(stringsAsFactors = FALSE)

## ---- Load prepared data ----
x_train <- readRDS("Data/x_train.rds")
x_validate <- readRDS("Data/x_validate.rds")
x_test <- readRDS("Data/x_test.rds")

y_train <- readRDS("Data/y_train.rds")
y_validate <- readRDS("Data/y_validate.rds")
y_test <- readRDS("Data/y_test.rds")

dummy_y_train <- readRDS("Data/dummy_y_train.rds")
dummy_y_val <- readRDS("Data/dummy_y_val.rds")
dummy_y_test <- readRDS("Data/dummy_y_test.rds")

cat("x_train shape:", dim(x_train), "\n")
cat("x_validate shape:", dim(x_validate), "\n")
cat("x_test shape:", dim(x_test), "\n")
print(table(y_train))
print(table(y_validate))
print(table(y_test))

## ---- Class weights ----
train_tab <- table(y_train)
cw_vals <- as.numeric(sum(train_tab) / (2 * train_tab))
class_weights <- setNames(as.list(cw_vals), names(train_tab))

## ---- Early stopping ----
callbacks <- list(
  callback_early_stopping(
    monitor = "val_auc",
    mode = "max",
    min_delta = 1e-3,
    patience = 30,
    restore_best_weights = TRUE
  )
)

## ---- Residual block ----
residual_block_1d <- function(x, filters, kernel_size, dropout_rate = 0) {
  shortcut <- x
  
  out <- x %>%
    layer_conv_1d(
      filters = filters,
      kernel_size = kernel_size,
      padding = "same",
      activation = "relu"
    ) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_conv_1d(
      filters = filters,
      kernel_size = kernel_size,
      padding = "same",
      activation = NULL
    ) %>%
    layer_batch_normalization()
  
  input_channels <- x$shape[[3]]
  if (!is.null(input_channels) && input_channels != filters) {
    shortcut <- shortcut %>%
      layer_conv_1d(
        filters = filters,
        kernel_size = 1,
        padding = "same",
        activation = NULL
      ) %>%
      layer_batch_normalization()
  }
  
  out <- layer_add(list(out, shortcut)) %>%
    layer_activation("relu")
  
  out
}

## ---- Model builder ----
build_resnet_model <- function(input_length,
                               filters1,
                               filters2,
                               kernel_size,
                               dropout_rate,
                               dense_units,
                               learning_rate) {
  
  inputs <- layer_input(shape = c(input_length, 1))
  
  x <- inputs %>%
    layer_conv_1d(
      filters = filters1,
      kernel_size = kernel_size,
      padding = "same",
      activation = "relu"
    ) %>%
    layer_batch_normalization()
  
  x <- residual_block_1d(x, filters = filters1, kernel_size = kernel_size, dropout_rate = dropout_rate)
  x <- residual_block_1d(x, filters = filters1, kernel_size = kernel_size, dropout_rate = dropout_rate)
  
  x <- x %>%
    layer_max_pooling_1d(pool_size = 2)
  
  x <- residual_block_1d(x, filters = filters2, kernel_size = kernel_size, dropout_rate = dropout_rate)
  x <- residual_block_1d(x, filters = filters2, kernel_size = kernel_size, dropout_rate = dropout_rate)
  
  x <- x %>%
    layer_global_average_pooling_1d() %>%
    layer_dense(units = dense_units, activation = "relu") %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 2, activation = "softmax")
  
  model <- keras_model(inputs = inputs, outputs = x)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = learning_rate),
    loss = "categorical_crossentropy",
    metrics = list(
      metric_auc(name = "auc"),
      metric_categorical_accuracy(name = "accuracy")
    )
  )
  
  model
}

## ---- Your full hyperparameter grid ----
filters1 <- c(16, 32, 64)
filters2 <- c(32, 64, 128)
kernel_size <- c(3, 5, 9)
dropout_rate <- c(0, 0.05, 0.1, 0.2)
dense_units <- c(32, 64, 128)
batch_size <- c(32, 64, 128)
learning_rate <- c(1e-3, 5e-4, 3e-4, 1e-4, 5e-5)

grid.search.full <- expand.grid(
  filters1 = filters1,
  filters2 = filters2,
  kernel_size = kernel_size,
  dropout_rate = dropout_rate,
  dense_units = dense_units,
  batch_size = batch_size,
  learning_rate = learning_rate
)

n_models <- nrow(grid.search.full)
if (model_id < 1 || model_id > n_models) {
  stop(sprintf("model_id %d is out of range. Valid range is 1 to %d.", model_id, n_models))
}

print(n_models)

## Optional: save once so you can reuse in processing scripts
dir.create("Models/resnet", recursive = TRUE, showWarnings = FALSE)
saveRDS(grid.search.full, "Models/resnet/grid.search.full.rds")

## ---- Storage ----
val_loss <- NA_real_
best_epoch_loss <- NA_integer_
best_epoch_auc <- NA_integer_
val_auc <- NA_real_
val_accuracy <- NA_real_
val_loss_at_best_auc <- NA_real_
val_accuracy_at_best_auc <- NA_real_

input_length <- dim(x_train)[2]

## ---- Run one model ----
cat("Processing ResNet Model #", model_id, "out of", n_models, "\n")

set_random_seed(15)

resnet_model <- build_resnet_model(
  input_length = input_length,
  filters1 = grid.search.full$filters1[model_id],
  filters2 = grid.search.full$filters2[model_id],
  kernel_size = grid.search.full$kernel_size[model_id],
  dropout_rate = grid.search.full$dropout_rate[model_id],
  dense_units = grid.search.full$dense_units[model_id],
  learning_rate = grid.search.full$learning_rate[model_id]
)

history <- resnet_model %>% fit(
  x = x_train,
  y = dummy_y_train,
  batch_size = grid.search.full$batch_size[model_id],
  epochs = 150,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = class_weights,
  callbacks = callbacks,
  verbose = 2
)

auc_hist <- history$metrics$val_auc
loss_hist <- history$metrics$val_loss
acc_hist <- history$metrics$val_accuracy

val_loss <- min(loss_hist)
best_epoch_loss <- which.min(loss_hist)

best_auc <- max(auc_hist)
tol <- 0.001
candidate_epochs <- which(auc_hist >= best_auc - tol)
best_epoch_auc <- max(candidate_epochs)

val_auc <- auc_hist[best_epoch_auc]
val_loss_at_best_auc <- loss_hist[best_epoch_auc]
val_accuracy_at_best_auc <- acc_hist[best_epoch_auc]

final_output <- c(
  val_loss = val_loss,
  best_epoch_loss = best_epoch_loss,
  best_epoch_auc = best_epoch_auc,
  val_auc = val_auc,
  val_loss_at_best_auc = val_loss_at_best_auc,
  val_accuracy_at_best_auc = val_accuracy_at_best_auc,
  model_id = model_id
)

nbatch <- (model_id - 1) %/% 100000 + 1
outdir <- paste0("Models/resnet/drac/training/training_metrics_b", nbatch)
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

saveRDS(final_output,
        file = file.path(outdir, paste0("training_output_", model_id, ".rds")))

cat("Training metrics for ResNet parameter configuration", model_id, "ready!\n")