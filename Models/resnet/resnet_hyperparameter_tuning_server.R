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
include_cw = opt$class_weights == "y"
include_callbacks = opt$callbacks == "y"
debug_mode = opt$debugging == "y"
rerun_model = opt$rerun == "y"

setwd("../../")

Sys.setenv(UV_OFFLINE=1)

# =========================================================
# 02_tune_resnet.R
# Hyperparameter tuning for 1D ResNet
# =========================================================

library(dplyr)
library(tensorflow)
library(keras3)
# library(caret)

##
if(debug_mode){
  library(reticulate)
  use_python("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3", required = TRUE) # change here
}

set.seed(15)
options(stringsAsFactors = FALSE)

## ---- Load prepared data from updated data_process.R ----
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
# train_tab <- table(y_train)
# cw <- as.numeric(train_tab[1] / train_tab[2])
# class_weights <- list("0" = 1, "1" = cw)

# train_tab <- table(y_train)
# cw_vals <- as.numeric(sum(train_tab) / (2 * train_tab))
# class_weights <- setNames(as.list(cw_vals), c("0", "1"))

train_tab <- table(y_train)
cw_vals <- as.numeric(sum(train_tab) / (2 * train_tab))
class_weights <- setNames(as.list(cw_vals), names(train_tab))

## ---- Early stopping ----
# callbacks <- list(
#   callback_early_stopping(
#     monitor = "val_loss",
#     min_delta = 1e-3,
#     patience = 30,
#     restore_best_weights = TRUE
#   )
# )

callbacks <- list(
  callback_early_stopping(
    monitor = "val_auc",
    mode = "max",
    min_delta = 1e-3,
    patience = 12,
    restore_best_weights = TRUE
  )
)

# for using legacy optimizers which work better with newer Macs
optimizers <- keras3::keras$optimizers # change here

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
  
  # oftmax makes all class probabilities add to 1, and categorical 
  # crossentropy measures how well the model puts probability mass on the correct class.
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


## ---- Hyperparameter grid ----
# make the model search a bit simpler and more targeted.
filters1 <- c(8, 16, 32)
filters2 <- c(16, 32, 64)
kernel_size <- c(3, 5, 9)
dropout_rate <- c(0.1, 0.2, 0.3)
dense_units <- c(16, 32, 64)
batch_size <- c(64, 128)
learning_rate <- c(1e-3, 3e-4, 1e-4)

grid.search.full <- expand.grid(
  filters1 = filters1,
  filters2 = filters2,
  kernel_size = kernel_size,
  dropout_rate = dropout_rate,
  dense_units = dense_units,
  batch_size = batch_size,
  learning_rate = learning_rate
)

# set.seed(15)
# n_subset <- 30
# grid.search.subset <- grid.search.full[
#   sample(1:nrow(grid.search.full), n_subset, replace = FALSE),
# ]

# Create vectors to store validation loss and best epoch in
## ---- Storage ----
val_loss <- rep(NA_real_, 1)
best_epoch_loss <- rep(NA_integer_, 1)
best_epoch_auc <- rep(NA_integer_, 1)
val_auc <- rep(NA_real_, 1)
val_accuracy <- rep(NA_real_, 1)
val_loss_at_best_auc <- rep(NA_real_, 1)
val_accuracy_at_best_auc <- rep(NA_real_, 1)

input_length <- dim(x_train)[2]

## ---- Run tuning ----
cat("Processing ResNet Model #", model_id, "\n")

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
  epochs = 100,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = class_weights,
  callbacks = callbacks,
  verbose = 2
)

auc_hist <- history$metrics$val_auc
loss_hist <- history$metrics$val_loss
acc_hist <- history$metrics$val_accuracy

val_loss[1] <- min(loss_hist)
best_epoch_loss[1]<-which(loss_hist==min(loss_hist))
best_epoch_auc[1] <- which.max(auc_hist)
val_auc[1] <- auc_hist[best_epoch_auc[1]]
val_loss_at_best_auc[1] <- loss_hist[best_epoch_auc[1]]
val_accuracy_at_best_auc[1] <- acc_hist[best_epoch_auc[1]]

final_output = c(val_loss,best_epoch_loss,best_epoch_auc,val_auc,val_loss_at_best_auc,val_accuracy_at_best_auc,model_id)

nbatch = (model_id-1)%/%100000 + 1
saveRDS(final_output,file = paste0("Models/resnet/drac/training/training_metrics_b",nbatch,"/training_output_",model_id,".rds"))


print(paste0("training metrics for resnet with parameter configuration ",model_id," ready!"))
