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
## Changes model_id if rerun is set to true ##

library(keras3)
library(tensorflow)

##
if(debug_mode){
  library(reticulate)
  use_python("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3", required = TRUE) # change here
}

set_random_seed(15)

# load processed data from Data/
x_train <- readRDS("Data/x_train.rds")
x_validate <- readRDS("Data/x_validate.rds")
x_test <- readRDS("Data/x_test.rds")

y_train <- readRDS("Data/y_train.rds")
y_validate <- readRDS("Data/y_validate.rds")
y_test <- readRDS("Data/y_test.rds")

# convert CNN arrays to DNN matrices
x_train <- matrix(x_train[, , 1], nrow = dim(x_train)[1], ncol = dim(x_train)[2])
x_validate <- matrix(x_validate[, , 1], nrow = dim(x_validate)[1], ncol = dim(x_validate)[2])
x_test <- matrix(x_test[, , 1], nrow = dim(x_test)[1], ncol = dim(x_test)[2])

# input dimension
input_dim <- ncol(x_train)

# class weights
class_counts <- table(y_train)

class_weight <- list(
  "0" = as.numeric(sum(class_counts) / (2 * class_counts["0"])),
  "1" = as.numeric(sum(class_counts) / (2 * class_counts["1"]))
)

# build DNN model
build_dnn_model <- function(input_dim,
                            hidden1 = 128,
                            hidden2 = 64,
                            dropout = 0.3,
                            l2_lambda = 0.001,
                            lr = 0.001) {
  
  model <- keras_model_sequential(input_shape = c(input_dim)) |>
    layer_dense(
      units = hidden1,
      kernel_regularizer = regularizer_l2(l2_lambda)
    ) |>
    layer_batch_normalization() |>
    layer_activation("relu") |>
    layer_dropout(rate = dropout) |>
    layer_dense(
      units = hidden2,
      kernel_regularizer = regularizer_l2(l2_lambda)
    ) |>
    layer_batch_normalization() |>
    layer_activation("relu") |>
    layer_dropout(rate = dropout) |>
    layer_dense(units = 1, activation = "sigmoid")
  
  model |>
    compile(
      optimizer = optimizer_adam(learning_rate = lr),
      loss = "binary_crossentropy",
      metrics = c("accuracy", metric_auc(name = "auc"))
    )
}

# hyperparameter grid
grid_full <- expand.grid(
  hidden1 = c(32, 64, 128),
  hidden2 = c(16, 32, 64),
  dropout = c(0.3, 0.4, 0.5),
  l2 = c(0.001, 0.005),
  lr = c(0.001, 0.0005),
  batch_size = c(32, 64),
  stringsAsFactors = FALSE
)

# set.seed(15)
# n_try <- 20
# grid_sub <- grid_full[sample(nrow(grid_full), n_try, replace = FALSE), ]

# callbacks
make_callbacks <- function() {
  list(
    callback_early_stopping(
      monitor = "val_auc",
      mode = "max",
      min_delta = 0.002,
      patience = 30,
      restore_best_weights = TRUE
    ),
    callback_reduce_lr_on_plateau(
      monitor = "val_auc",
      mode = "max",
      factor = 0.5,
      patience = 5,
      min_lr = 1e-5
    )
  )
}


# Create vectors to store validation loss and best epoch in
## ---- Storage ----
val_loss<-rep(NA,1)
best_epoch_loss<-rep(NA,1)
val_auc<-rep(NA,1)
best_val_acc<-rep(NA,1)
best_val_auc<-rep(NA,1)

print(paste0("Processing Model #", model_id))

model <- build_dnn_model(
  input_dim = input_dim,
  hidden1 = grid_full$hidden1[model_id],
  hidden2 = grid_full$hidden2[model_id],
  dropout = grid_full$dropout[model_id],
  l2_lambda = grid_full$l2[model_id],
  lr = grid_full$lr[model_id]
)

history <- model |>
  fit(
    x = x_train,
    y = y_train,
    validation_data = list(x_validate, y_validate),
    epochs = 100,
    batch_size = grid_full$batch_size[model_id],
    callbacks = make_callbacks(),
    class_weight = class_weight,
    verbose = 2
  )

val_loss_vec <- history$metrics$val_loss
val_acc_vec <- history$metrics$val_accuracy
val_auc_vec <- history$metrics$val_auc

val_loss[1]<-min(history$metrics$val_loss)
best_epoch_loss[1]<-which(val_loss_vec==min(val_loss_vec))
val_auc[1] <- val_auc_vec[best_epoch_loss[1]]
best_val_acc[1] <- val_acc_vec[best_epoch_loss[1]]
best_val_auc[1] <- val_auc_vec[best_epoch_loss[1]]

final_output = c(val_loss,best_epoch_loss,val_auc,best_val_acc,best_val_auc,model_id)

nbatch = (model_id-1)%/%100000 + 1
saveRDS(final_output,file = paste0("Models/dnn/drac/training/training_metrics_b",nbatch,"/training_output_",model_id,".rds"))


print(paste0("training metrics for dnn with parameter configuration ",model_id," ready!"))

