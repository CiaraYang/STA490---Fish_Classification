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
  # make_option(c("-b", "--batch"), type="integer", default="1", 
  #             help="100k batch ID for rerunning missing models [default= %default]", metavar="integer"),
  # make_option(c("-r", "--rerun"), type="character", default="n",
  #             help="Rerun missing models [default= %default]", metavar="character")
  
)
### Everything is set to incorporate step size tunning for sourceCpp if needed in the future 
opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);

model_id = opt$model

setwd("../../")

Sys.setenv(UV_OFFLINE=1)

# =========================================================
# 03_train_final_cnn.R
# Train final 1D CNN and evaluate on test set
# =========================================================

# ## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)
library(rsample)
library(keras3)

##
if(debug_mode){
  library(reticulate)
  use_python("/Library/Frameworks/Python.framework/Versions/3.12/bin/python3", required = TRUE) # change here
}


## ---- Load prepared data ----
x_train = readRDS("Data/x_train.rds")
x_validate = readRDS("Data/x_validate.rds")
x_test = readRDS("Data/x_test.rds")

dummy_y_train = readRDS("Data/dummy_y_train.rds")
dummy_y_val = readRDS("Data/dummy_y_val.rds")
# saveRDS(dummy_y_val,"Data/dummy_y_val.rds")
y_train = readRDS("Data/y_train.rds")
y_validate = readRDS("Data/y_validate.rds")
y_test = readRDS("Data/y_test.rds")

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

# for using legacy optimizers which work better with newer Macs
optimizers <- keras3::keras$optimizers # change here

# Validation metric fitted models
final_data = readRDS(paste0("1d_structured_networks/cnn/val_metrics.rds"))

## ---- Parameter grid ----
grid.search.full = readRDS(file = "1d_structured_networks/cnn/grid.search.full.rds")
grid.search.full = as_tibble(grid.search.full) %>% mutate(model_id = row_number())

# # Create vectors to store validation loss and best epoch in
# ## ---- Storage ----
# val_loss<-rep(NA,1)
# best_epoch_loss<-rep(NA,1)
# val_auc<-rep(NA,1)

print(paste0("Processing Model #", model_id))

t1 = Sys.time()
input_length <- dim(x_train)[2]

## ---- Model builder ----
build_cnn_model <- function(filters1, filters2, filters3, filters4, filters5,
                            kernel_size,
                            droprate1, droprate2, droprate3, droprate4, droprate5,
                            input_length) {
  
  model <- keras_model_sequential(input_shape = c(input_length, 1)) %>%
    layer_conv_1d(
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


# val_loss[1] <- min(cnn_history$metrics$val_loss)
# best_epoch_loss[1] <- which.min(cnn_history$metrics$val_loss)
# val_auc[1] <- max(cnn_history$metrics$val_auc)


t2 = Sys.time()
t2 - t1

# print("--------")

## ---- Evaluate on test set ----

### Adapt this ###
eval <- cnn %>% evaluate(x_test, dummy_y_test, verbose = 0, return_dict = TRUE)
print(eval)

## ---- Predict on test set ----
pred_probs <- cnn %>% predict(x_test)

prob_smelt <- pred_probs[, 2]
pred_class_idx <- apply(pred_probs, 1, which.max)

species_pred <- factor(
  ifelse(pred_class_idx == 1, "Alewife", "Rainbow Smelt"),
  levels = c("Alewife", "Rainbow Smelt")
)

true_labels <- factor(
  ifelse(dummy_y_test[, 1] == 1, "Alewife", "Rainbow Smelt"),
  levels = c("Alewife", "Rainbow Smelt")
)

## ---- Confusion matrix ----
cm <- confusionMatrix(
  data = species_pred,
  reference = true_labels,
  positive = "Rainbow Smelt"
)

print(cm)

## ---- Additional metrics ----
roc_obj <- roc(
  response = true_labels,
  predictor = prob_smelt,
  levels = c("Alewife", "Rainbow Smelt"),
  direction = "<"
)

test_auc <- as.numeric(auc(roc_obj))

metrics_tbl <- tibble(
  loss = as.numeric(eval[["loss"]]),
  auc_from_keras = as.numeric(eval[["auc"]]),
  accuracy_from_keras = as.numeric(eval[["accuracy"]]),
  auc_from_pROC = test_auc,
  accuracy = as.numeric(cm$overall["Accuracy"]),
  sensitivity = as.numeric(cm$byClass["Sensitivity"]),
  specificity = as.numeric(cm$byClass["Specificity"]),
  balanced_accuracy = as.numeric(cm$byClass["Balanced Accuracy"]),
  precision = as.numeric(cm$byClass["Precision"]),
  recall = as.numeric(cm$byClass["Recall"]),
  f1 = as.numeric(cm$byClass["F1"])
)

print(metrics_tbl)

## ---- Save outputs ----
write.csv(metrics_tbl, "Data/cnn_test_metrics.csv", row.names = FALSE)
save(cnn, cnn_history, cm, metrics_tbl, pred_probs,
     file = "Data/final_cnn_model_results.RData")



# val_loss[1]<-min(cnn_history$metrics$val_loss)
# best_epoch_loss[1]<-which(cnn_history$metrics$val_loss==min(cnn_history$metrics$val_loss))
# val_auc[1] <- cnn_history$metrics$val_auc[best_epoch_loss[1]]
# 
# final_output = c(val_loss,best_epoch_loss,val_auc,model_id)
# 
# nbatch = (model_id-1)%/%100000 + 1
# saveRDS(final_output,file = paste0("Models/cnn/drac/training/test/cnn/training_metrics_b",nbatch,"/training_output_",model_id,".rds"))
# 
# 
# print(paste0("training metrics for cnn with parameter configuration ",model_id," ready!"))
# 
