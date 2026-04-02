#!/usr/bin/env Rscript
library("optparse")

option_list = list(
  make_option(c("-d", "--debugging"), type="character", default="n",
              help="Debugging mode (only for local runs) [default= %default]", metavar="character"),
  make_option(c("-m", "--model"), type="integer", default="1", 
              help="model ID [default= %default]", metavar="integer"),
  make_option(c("-b", "--batch"), type="integer", default="1", 
              help="100k batch ID for rerunning missing models [default= %default]", metavar="integer"),
  make_option(c("-r", "--rerun"), type="character", default="n",
              help="Rerun missing models [default= %default]", metavar="character")
)

opt_parser = OptionParser(option_list=option_list)
opt = parse_args(opt_parser)

model_id = opt$model
debug_mode = opt$debugging == "y"
rerun_model = opt$rerun == "y"

setwd("../../")

Sys.setenv(UV_OFFLINE=1)

if(rerun_model){
  if(opt$batch == 1){
    missing_models = readRDS("Models/cnn/drac/training/missing_models_1_100000.rds")
  } else if(opt$batch == 2){
    missing_models = readRDS("Models/cnn/drac/training/missing_models_100001_200000.rds")
  } else if(opt$batch == 3){
    missing_models = readRDS("Models/cnn/drac/training/missing_models_200001_300000.rds")
  } else if(opt$batch == 4){
    missing_models = readRDS("Models/cnn/drac/training/missing_models_300001_400000.rds")
  } else if(opt$batch == 5){
    missing_models = readRDS("Models/cnn/drac/training/missing_models_400001_500000.rds")
  } else if(opt$batch == 6){
    missing_models = readRDS("Models/cnn/drac/training/missing_models_500001_total.rds")
  }
  model_id = missing_models[model_id]
}

library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)
library(rsample)
library(keras3)

if(debug_mode){
  library(reticulate)
  use_python("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3", required = TRUE)
}

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

## ---- Check data ----
cat("x_train shape:", dim(x_train), "\n")
cat("x_validate shape:", dim(x_validate), "\n")
cat("x_test shape:", dim(x_test), "\n")
print(table(y_train))
print(table(y_validate))
print(table(y_test))

## ---- Class weight calculation ----
train_tab <- table(y_train)
class_weights_vec <- as.numeric(sum(train_tab) / (length(train_tab) * train_tab))
class_weights <- as.list(class_weights_vec)
names(class_weights) <- as.character(0:(length(class_weights) - 1))

## ---- Early stopping ----
callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
    min_delta = 1e-3,
    patience = 30,
    restore_best_weights = TRUE
  )
)

## ---- Parameter grid ----
filters1 <- c(8, 16, 32)
filters2 <- c(8, 16, 32)
filters3 <- c(8, 16, 32)
filters4 <- c(8, 16, 32)
filters5 <- c(8, 16, 32)

kernel_size <- c(3, 5, 7)
batch_size <- c(32, 64, 128)

droprate1 <- c(0, 0.1, 0.2)
droprate2 <- c(0, 0.1, 0.2)
droprate3 <- c(0, 0.1, 0.2)
droprate4 <- c(0, 0.1, 0.2)
droprate5 <- c(0, 0.1, 0.2)

grid.search.full <- expand.grid(
  filters1 = filters1,
  filters2 = filters2,
  filters3 = filters3,
  filters4 = filters4,
  filters5 = filters5,
  kernel_size = kernel_size,
  batch_size = batch_size,
  droprate1 = droprate1,
  droprate2 = droprate2,
  droprate3 = droprate3,
  droprate4 = droprate4,
  droprate5 = droprate5
)

## ---- Storage ----
val_loss <- rep(NA, 1)
best_epoch_loss <- rep(NA, 1)
val_auc <- rep(NA, 1)

print(paste0("Processing Model #", model_id))

t1 = Sys.time()
input_length <- dim(x_train)[2]

## ---- Model builder ----
build_cnn_model <- function(filters1, filters2, filters3, filters4, filters5,
                            kernel_size,
                            droprate1, droprate2, droprate3, droprate4, droprate5,
                            input_length) {
  
  model <- keras_model_sequential() %>%
    layer_conv_1d(
      input_shape = c(input_length, 1),
      filters = filters1,
      kernel_size = kernel_size,
      activation = "relu",
      padding = "same",
      strides = 1
    ) %>%
    layer_dropout(rate = droprate1) %>%
    layer_batch_normalization() %>%
    
    layer_conv_1d(
      filters = filters2,
      kernel_size = kernel_size,
      activation = "relu",
      padding = "same",
      strides = 1
    ) %>%
    layer_dropout(rate = droprate2) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    
    layer_conv_1d(
      filters = filters3,
      kernel_size = kernel_size,
      activation = "relu",
      padding = "same",
      strides = 1
    ) %>%
    layer_dropout(rate = droprate3) %>%
    layer_batch_normalization() %>%
    
    layer_conv_1d(
      filters = filters4,
      kernel_size = kernel_size,
      activation = "relu",
      padding = "same",
      strides = 1
    ) %>%
    layer_dropout(rate = droprate4) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_batch_normalization() %>%
    
    layer_conv_1d(
      filters = filters5,
      kernel_size = kernel_size,
      activation = "relu",
      padding = "same",
      strides = 1
    ) %>%
    layer_dropout(rate = droprate5) %>%
    layer_batch_normalization() %>%
    
    layer_flatten() %>%
    layer_dense(units = 2, activation = "softmax")
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 1e-4),
    loss = "categorical_crossentropy",
    metrics = list(metric_auc(name = "auc"))
  )
  
  return(model)
}

## ---- Run search ----
set_random_seed(15)

cnn <- build_cnn_model(
  filters1 = grid.search.full$filters1[model_id],
  filters2 = grid.search.full$filters2[model_id],
  filters3 = grid.search.full$filters3[model_id],
  filters4 = grid.search.full$filters4[model_id],
  filters5 = grid.search.full$filters5[model_id],
  kernel_size = grid.search.full$kernel_size[model_id],
  droprate1 = grid.search.full$droprate1[model_id],
  droprate2 = grid.search.full$droprate2[model_id],
  droprate3 = grid.search.full$droprate3[model_id],
  droprate4 = grid.search.full$droprate4[model_id],
  droprate5 = grid.search.full$droprate5[model_id],
  input_length = input_length
)

cnn_history <- cnn %>% fit(
  x_train, dummy_y_train,
  batch_size = grid.search.full$batch_size[model_id],
  epochs = 200,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = class_weights,
  callbacks = callbacks,
  verbose = 2
)

val_loss[1] <- min(cnn_history$metrics$val_loss)
best_epoch_loss[1] <- which.min(cnn_history$metrics$val_loss)
val_auc[1] <- max(cnn_history$metrics$val_auc)

t2 = Sys.time()
print(t2 - t1)

final_output = c(val_loss, best_epoch_loss, val_auc, model_id)

nbatch = (model_id - 1) %/% 100000 + 1
saveRDS(
  final_output,
  file = paste0("Models/cnn/drac/training/training_metrics_b", nbatch, "/training_output_", model_id, ".rds")
)

print(paste0("training metrics for cnn with parameter configuration ", model_id, " ready!"))