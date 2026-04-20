#!/usr/bin/env Rscript

# server version for ResNet model testing

library(dplyr)
library(tensorflow)
library(caret)
library(keras3)
library(pROC)
library(tibble)
library(readr)

setwd("../../../../")

# -----------------------------
# Output folder
# -----------------------------
test_dir <- "Models/resnet/drac/testing"
if (!dir.exists(test_dir)) dir.create(test_dir, recursive = TRUE)

# -----------------------------
# Load data
# -----------------------------
cat("After setwd:", getwd(), "\n")
cat("x_train exists?:", file.exists("Data/x_train.rds"), "\n")

x_train <- readRDS("Data/x_train.rds")
x_validate <- readRDS("Data/x_validate.rds")
x_test <- readRDS("Data/x_test.rds")

y_train <- readRDS("Data/y_train.rds")
y_validate <- readRDS("Data/y_validate.rds")
y_test <- readRDS("Data/y_test.rds")

dummy_y_train <- readRDS("Data/dummy_y_train.rds")
dummy_y_val <- readRDS("Data/dummy_y_val.rds")
dummy_y_test <- readRDS("Data/dummy_y_test.rds")

# -----------------------------
# Load top 20 configs
# Change this path if your top20 file is stored elsewhere
# -----------------------------
top20_models <- readRDS(
  "Models/resnet/drac/training/top20_resnet_configs.rds"
)

# -----------------------------
# Class weights
# -----------------------------
train_tab <- table(y_train)
class_weights_vec <- as.numeric(sum(train_tab) / (length(train_tab) * train_tab))
class_weights <- as.list(class_weights_vec)
names(class_weights) <- as.character(0:(length(class_weights) - 1))

# -----------------------------
# Early stopping
# Keep this aligned with your ResNet tuning logic
# -----------------------------
callbacks <- list(
  callback_early_stopping(
    monitor = "val_auc",
    mode = "max",
    min_delta = 1e-3,
    patience = 30,
    restore_best_weights = TRUE
  )
)

# -----------------------------
# Residual block
# -----------------------------
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

# -----------------------------
# Build ResNet
# -----------------------------
build_resnet_model <- function(param, input_length) {
  inputs <- layer_input(shape = c(input_length, 1))
  
  x <- inputs %>%
    layer_conv_1d(
      filters = param$filters1,
      kernel_size = param$kernel_size,
      padding = "same",
      activation = "relu"
    ) %>%
    layer_batch_normalization()
  
  x <- residual_block_1d(
    x,
    filters = param$filters1,
    kernel_size = param$kernel_size,
    dropout_rate = param$dropout_rate
  )
  x <- residual_block_1d(
    x,
    filters = param$filters1,
    kernel_size = param$kernel_size,
    dropout_rate = param$dropout_rate
  )
  
  x <- x %>%
    layer_max_pooling_1d(pool_size = 2)
  
  x <- residual_block_1d(
    x,
    filters = param$filters2,
    kernel_size = param$kernel_size,
    dropout_rate = param$dropout_rate
  )
  x <- residual_block_1d(
    x,
    filters = param$filters2,
    kernel_size = param$kernel_size,
    dropout_rate = param$dropout_rate
  )
  
  x <- x %>%
    layer_global_average_pooling_1d() %>%
    layer_dense(units = param$dense_units, activation = "relu") %>%
    layer_dropout(rate = param$dropout_rate) %>%
    layer_dense(units = 2, activation = "softmax")
  
  model <- keras_model(inputs = inputs, outputs = x)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = param$learning_rate),
    loss = "categorical_crossentropy",
    metrics = list(
      metric_auc(name = "auc"),
      metric_categorical_accuracy(name = "accuracy")
    )
  )
  
  model
}

# -----------------------------
# Run models
# -----------------------------
input_length <- dim(x_train)[2]
all_results <- list()

for (i in seq_len(nrow(top20_models))) {
  
  cat("Running ResNet model", i, "\n")
  
  param <- top20_models[i, ]
  
  set_random_seed(15)
  
  resnet_model <- build_resnet_model(param, input_length)
  
  history <- resnet_model %>% fit(
    x = x_train,
    y = dummy_y_train,
    batch_size = param$batch_size,
    epochs = 150,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = class_weights,
    callbacks = callbacks,
    verbose = 1
  )
  
  eval <- resnet_model %>% evaluate(
    x_test, dummy_y_test,
    verbose = 0,
    return_dict = TRUE
  )
  
  pred_probs <- resnet_model %>% predict(x_test, verbose = 0)
  prob_smelt <- pred_probs[, 2]
  
  true_labels <- factor(
    ifelse(dummy_y_test[, 1] == 1, "Alewife", "Rainbow Smelt"),
    levels = c("Alewife", "Rainbow Smelt")
  )
  
  # -----------------------------
  # Threshold sweep
  # Same idea as tutor's CNN script
  # -----------------------------
  thresholds <- seq(0.01, 0.99, by = 0.01)
  threshold_results <- data.frame()
  
  for (t in thresholds) {
    
    species_pred <- factor(
      ifelse(prob_smelt >= t, "Rainbow Smelt", "Alewife"),
      levels = c("Alewife", "Rainbow Smelt")
    )
    
    cm_tmp <- confusionMatrix(
      data = species_pred,
      reference = true_labels,
      positive = "Rainbow Smelt"
    )
    
    threshold_results <- rbind(
      threshold_results,
      data.frame(
        threshold = t,
        accuracy = as.numeric(cm_tmp$overall["Accuracy"]),
        sensitivity = as.numeric(cm_tmp$byClass["Sensitivity"]),
        specificity = as.numeric(cm_tmp$byClass["Specificity"]),
        balanced_accuracy = as.numeric(cm_tmp$byClass["Balanced Accuracy"]),
        precision = as.numeric(cm_tmp$byClass["Precision"]),
        recall = as.numeric(cm_tmp$byClass["Recall"]),
        f1 = as.numeric(cm_tmp$byClass["F1"])
      )
    )
  }
  
  best_row <- which.max(threshold_results$balanced_accuracy)
  best_threshold <- threshold_results$threshold[best_row]
  
  species_pred <- factor(
    ifelse(prob_smelt >= best_threshold, "Rainbow Smelt", "Alewife"),
    levels = c("Alewife", "Rainbow Smelt")
  )
  
  cm <- confusionMatrix(
    data = species_pred,
    reference = true_labels,
    positive = "Rainbow Smelt"
  )
  
  roc_obj <- roc(
    response = true_labels,
    predictor = prob_smelt,
    levels = c("Alewife", "Rainbow Smelt"),
    direction = "<"
  )
  
  test_auc <- as.numeric(auc(roc_obj))
  
  # keep your AUC-based epoch rule
  auc_hist <- history$metrics$val_auc
  best_auc <- max(auc_hist)
  tol <- 0.001
  candidate_epochs <- which(auc_hist >= best_auc - tol)
  best_epoch <- max(candidate_epochs)
  
  metrics_tbl <- tibble(
    model = i,
    model_id = if ("model_id" %in% names(param)) param$model_id else NA_integer_,
    filters1 = param$filters1,
    filters2 = param$filters2,
    kernel_size = param$kernel_size,
    dropout_rate = param$dropout_rate,
    dense_units = param$dense_units,
    batch_size = param$batch_size,
    learning_rate = param$learning_rate,
    best_epoch = best_epoch,
    val_loss = history$metrics$val_loss[best_epoch],
    val_auc = history$metrics$val_auc[best_epoch],
    test_loss = eval[["loss"]],
    test_auc_keras = eval[["auc"]],
    test_accuracy_keras = eval[["accuracy"]],
    test_auc_pROC = test_auc,
    best_threshold = best_threshold,
    accuracy = as.numeric(cm$overall["Accuracy"]),
    sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    specificity = as.numeric(cm$byClass["Specificity"]),
    balanced_accuracy = as.numeric(cm$byClass["Balanced Accuracy"]),
    precision = as.numeric(cm$byClass["Precision"]),
    recall = as.numeric(cm$byClass["Recall"]),
    f1 = as.numeric(cm$byClass["F1"])
  )
  
  print(metrics_tbl)
  
  all_results[[i]] <- metrics_tbl
  
  write.csv(
    metrics_tbl,
    file.path(test_dir, paste0("resnet_model_", i, ".csv")),
    row.names = FALSE
  )
  
  write.csv(
    threshold_results,
    file.path(test_dir, paste0("resnet_model_", i, "_thresholds.csv")),
    row.names = FALSE
  )
  
  keras3::clear_session()
  gc()
}

all_results_tbl <- bind_rows(all_results) %>%
  arrange(desc(test_auc_pROC), desc(balanced_accuracy))

print(all_results_tbl)

write.csv(
  all_results_tbl,
  file.path(test_dir, "top20_resnet_results.csv"),
  row.names = FALSE
)

saveRDS(
  all_results_tbl,
  file.path(test_dir, "top20_resnet_results.rds")
)