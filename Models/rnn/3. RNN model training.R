library(dplyr)
library(tensorflow)
library(caret)
library(keras3)
library(pROC)
library(tibble)

# =========================================================
# Local pipeline for RNN:
# 1. test top 20 models
# 2. rank from best to worst
# 3. save results to Models/rnn/drac/test
# 4. refit best model
# 5. plot(model_history)
# =========================================================

set.seed(15)

cat("Current working directory:", getwd(), "\n")

# -------------------------
# Output folder
# -------------------------
test_dir <- "Models/rnn/drac/test"
if (!dir.exists(test_dir)) dir.create(test_dir, recursive = TRUE)

# -------------------------
# Load data
# -------------------------
x_train <- readRDS("Data/RNN_data/x_train.rds")
x_validate <- readRDS("Data/RNN_data/x_validate.rds")
x_test <- readRDS("Data/RNN_data/x_test.rds")

y_train <- readRDS("Data/RNN_data/y_train.rds")
dummy_y_train <- readRDS("Data/RNN_data/dummy_y_train.rds")
dummy_y_val <- readRDS("Data/RNN_data/dummy_y_val.rds")
dummy_y_test <- readRDS("Data/RNN_data/dummy_y_test.rds")

# -------------------------
# Load top 20 files
# top20_models: only model_id
# top20_configs: full config table including best_epoch_loss
# -------------------------
top20_models <- readRDS("Models/rnn/drac/training/training_metrics_full/top20_models.rds")
top20_configs <- readRDS("Models/rnn/drac/training/training_metrics_full/top20_configs.rds")

cat("\nTop 20 models preview:\n")

print(top20_models)

cat("\nTop 20 configs preview:\n")
print(top20_configs)

# -------------------------
# Sanity checks
# -------------------------
if (!"model_id" %in% names(top20_models)) {
  stop("Column 'model_id' not found in top20_models.")
}
if (!"model_id" %in% names(top20_configs)) {
  stop("Column 'model_id' not found in top20_configs.")
}
if (!"best_epoch_loss" %in% names(top20_configs)) {
  stop("Column 'best_epoch_loss' not found in top20_configs.")
}
if (nrow(top20_models) != nrow(top20_configs)) {
  stop("top20_models and top20_configs do not have the same number of rows.")
}

# -------------------------
# Class weights
# -------------------------
class_counts <- table(y_train)
cw <- as.numeric(class_counts["Alewife"] / class_counts["Rainbow Smelt"])
class_weight_list <- list("0" = 1, "1" = cw)

cat("\nClass counts:\n")
print(class_counts)

cat("\nClass weights:\n")
print(class_weight_list)

# -------------------------
# Input shape
# -------------------------
input_shape_use <- c(dim(x_train)[2], dim(x_train)[3])

# -------------------------
# Helper: build RNN model
# -------------------------
build_rnn_model <- function(param_row, input_shape_use) {
  rnn <- keras_model_sequential() %>%
    layer_lstm(
      input_shape = input_shape_use,
      units = param_row$lstmunits[[1]]
    ) %>%
    layer_activation_leaky_relu()
  
  if ("use_batchnorm" %in% names(param_row) && isTRUE(param_row$use_batchnorm[[1]])) {
    rnn <- rnn %>% layer_batch_normalization()
  }
  
  if ("dropout1" %in% names(param_row) && param_row$dropout1[[1]] > 0) {
    rnn <- rnn %>% layer_dropout(rate = param_row$dropout1[[1]])
  }
  
  rnn <- rnn %>%
    layer_dense(
      units = param_row$neuron1[[1]],
      activity_regularizer = regularizer_l2(l = param_row$regrate[[1]])
    ) %>%
    layer_activation_leaky_relu() %>%
    layer_dense(units = 2, activation = "softmax")
  
  lr_use <- if ("lr" %in% names(param_row)) param_row$lr[[1]] else 1e-4
  
  rnn %>% compile(
    optimizer = optimizer_adam(learning_rate = lr_use),
    loss = "categorical_crossentropy",
    metrics = list(
      metric_auc(name = "auc"),
      metric_categorical_accuracy(name = "accuracy")
    )
  )
  
  rnn
}

# -------------------------
# Storage
# -------------------------
all_results <- vector("list", nrow(top20_models))

# =========================================================
# Test top 20 models
# =========================================================
for (i in seq_len(nrow(top20_models))) {
  
  model_id_use <- top20_models$model_id[[i]]
  params <- top20_configs[i, ]
  config_model_id <- params$model_id[[1]]
  best_epoch_use <- params$best_epoch_loss[[1]]
  
  if (model_id_use != config_model_id) {
    stop(paste(
      "Mismatch at row", i,
      ": top20_models model_id =", model_id_use,
      "but top20_configs model_id =", config_model_id
    ))
  }
  
  cat("\n=====================================\n")
  cat("Testing model", i, "/ model_id =", model_id_use, "\n")
  print(params)
  cat("Best epoch:", best_epoch_use, "\n")
  
  if (is.null(best_epoch_use) || is.na(best_epoch_use) || best_epoch_use <= 0) {
    stop(paste("Invalid best_epoch for model_id =", model_id_use))
  }
  
  set.seed(15)
  set_random_seed(15)
  
  rnn <- build_rnn_model(params, input_shape_use)
  
  history <- rnn %>% fit(
    x = x_train,
    y = dummy_y_train,
    batch_size = params$batchsize[[1]],
    epochs = best_epoch_use,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = class_weight_list,
    verbose = 0
  )
  
  eval <- rnn %>% evaluate(
    x_test, dummy_y_test,
    verbose = 0,
    return_dict = TRUE
  )
  
  pred_probs <- rnn %>% predict(x_test, verbose = 0)
  prob_smelt <- pred_probs[, 2]
  
  # fixed threshold = 0.5
  species_pred <- factor(
    ifelse(prob_smelt >= 0.5, "Rainbow Smelt", "Alewife"),
    levels = c("Alewife", "Rainbow Smelt")
  )
  
  true_labels <- factor(
    ifelse(dummy_y_test[, 1] == 1, "Alewife", "Rainbow Smelt"),
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
  
  metrics_tbl <- tibble(
    model = i,
    model_id = model_id_use,
    best_epoch = best_epoch_use,
    val_loss = history$metrics$val_loss[best_epoch_use],
    val_auc = history$metrics$val_auc[best_epoch_use],
    test_loss = eval[["loss"]],
    test_auc_keras = eval[["auc"]],
    test_accuracy_keras = eval[["accuracy"]],
    test_auc_pROC = test_auc,
    accuracy = as.numeric(cm$overall["Accuracy"]),
    sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    specificity = as.numeric(cm$byClass["Specificity"]),
    balanced_accuracy = as.numeric(cm$byClass["Balanced Accuracy"]),
    precision = as.numeric(cm$byClass["Precision"]),
    recall = as.numeric(cm$byClass["Recall"]),
    f1 = as.numeric(cm$byClass["F1"]),
    lstmunits = params$lstmunits[[1]],
    neuron1 = params$neuron1[[1]],
    batchsize = params$batchsize[[1]],
    lr = if ("lr" %in% names(params)) params$lr[[1]] else NA_real_,
    dropout1 = if ("dropout1" %in% names(params)) params$dropout1[[1]] else NA_real_,
    regrate = params$regrate[[1]],
    use_batchnorm = if ("use_batchnorm" %in% names(params)) params$use_batchnorm[[1]] else NA
  )
  
  print(metrics_tbl)
  
  all_results[[i]] <- metrics_tbl
  
  keras3::clear_session()
  gc()
}

# =========================================================
# Rank top 20 results from best to worst
# =========================================================
all_results_tbl <- bind_rows(all_results) %>%
  arrange(desc(balanced_accuracy), desc(test_auc_pROC))

cat("\n=====================================\n")
cat("Final ranked top 20 RNN results:\n")
print(all_results_tbl)

write.csv(
  all_results_tbl,
  file.path(test_dir, "top_20_rnn_results.csv"),
  row.names = FALSE
)

saveRDS(
  all_results_tbl,
  file.path(test_dir, "top_20_rnn_results.rds")
)

# =========================================================
# Extract best model (first row after ranking)
# =========================================================
best_model_id <- all_results_tbl$model_id[[1]]

best_param <- top20_configs %>%
  filter(model_id == best_model_id) %>%
  mutate(best_epoch = best_epoch_loss)

if (nrow(best_param) != 1) {
  stop("Could not uniquely identify best model in top20_configs.")
}

cat("\n=====================================\n")
cat("Best model selected:\n")
print(best_param)

# =========================================================
# Refit best model using your code style
# =========================================================
set.seed(15)

# Input shape
input_shape_use <- c(dim(x_train)[2], dim(x_train)[3])

# Class weights in your original style
class_counts <- table(y_train)
cw <- as.numeric(class_counts["Alewife"] / class_counts["Rainbow Smelt"])
class_weight_list <- list("0" = 1, "1" = cw)

print(class_counts)
print(class_weight_list)

set_random_seed(15)

rnn <- keras_model_sequential() %>%
  layer_lstm(
    input_shape = input_shape_use,
    units = best_param$lstmunits[[1]]
  ) %>%
  layer_activation_leaky_relu()

if ("use_batchnorm" %in% names(best_param) && isTRUE(best_param$use_batchnorm[[1]])) {
  rnn <- rnn %>% layer_batch_normalization()
}

if ("dropout1" %in% names(best_param) && best_param$dropout1[[1]] > 0) {
  rnn <- rnn %>% layer_dropout(rate = best_param$dropout1[[1]])
}

rnn <- rnn %>%
  layer_dense(
    units = best_param$neuron1[[1]],
    activity_regularizer = regularizer_l2(l = best_param$regrate[[1]])
  ) %>%
  layer_activation_leaky_relu() %>%
  layer_dense(units = 2, activation = "softmax")

rnn %>% compile(
  loss = loss_categorical_crossentropy(),
  optimizer = optimizer_adam(learning_rate = best_param$lr[[1]]),
  metrics = c("accuracy", metric_auc(name = "auc"))
)

model_history <- rnn %>% fit(
  x_train, dummy_y_train,
  batch_size = best_param$batchsize[[1]],
  epochs = best_param$best_epoch[[1]],
  validation_data = list(x_validate, dummy_y_val),
  class_weight = class_weight_list,
  verbose = 1
)

# -------------------------
# Evaluate on test set
# -------------------------
test_results <- rnn %>% evaluate(
  x_test, dummy_y_test,
  verbose = 0
)

# -------------------------
# Predict on test set
# -------------------------
test_pred_prob <- rnn %>% predict(x_test)
test_pred_class <- max.col(test_pred_prob) - 1
test_true_class <- max.col(dummy_y_test) - 1

pred_species <- factor(
  ifelse(test_pred_class == 0, "Alewife", "Rainbow Smelt"),
  levels = c("Alewife", "Rainbow Smelt")
)

true_species <- factor(
  ifelse(test_true_class == 0, "Alewife", "Rainbow Smelt"),
  levels = c("Alewife", "Rainbow Smelt")
)

# -------------------------
# Confusion matrix
# -------------------------
cm <- table(Predicted = pred_species, True = true_species)

cat("\nConfusion matrix:\n")
print(cm)

TN <- cm["Alewife", "Alewife"]
FN <- cm["Alewife", "Rainbow Smelt"]
FP <- cm["Rainbow Smelt", "Alewife"]
TP <- cm["Rainbow Smelt", "Rainbow Smelt"]

accuracy <- (TP + TN) / sum(cm)
accuracy
sensitivity_smelt <- TP / (TP + FN)
sensitivity_smelt
specificity_alewife <- TN / (TN + FP)
specificity_alewife
precision_smelt <- TP / (TP + FP)
precision_smelt
balanced_accuracy <- (sensitivity_smelt + specificity_alewife) / 2
balanced_accuracy
performance_table <- data.frame(
  Metric = c(
    "Accuracy",
    "AUC (Area Under the ROC Curve)",
    "Loss",
    "Balanced Accuracy"
  ),
  Value = round(c(
    test_results$accuracy,
    test_results$auc,
    test_results$loss,
    balanced_accuracy
  ), 3)
)

cat("\nTest performance table:\n")
print(performance_table)

cat("\nDerived metrics:\n")
cat("Sensitivity (Smelt):", round(sensitivity_smelt, 3), "\n")
cat("Specificity (Alewife):", round(specificity_alewife, 3), "\n")
cat("Precision (Smelt):", round(precision_smelt, 3), "\n")

# -------------------------
# Save best model summary
# -------------------------
best_result <- all_results_tbl %>%
  slice(1)

write.csv(
  best_result,
  file.path(test_dir, "best_rnn_model_result.csv"),
  row.names = FALSE
)

saveRDS(
  best_result,
  file.path(test_dir, "best_rnn_model_result.rds")
)

# -------------------------
# Plot training history
# Use your style
# -------------------------
png(
  filename = file.path(test_dir, "best_rnn_training_history.png"),
  width = 1200,
  height = 1600,
  res = 150
)
plot(model_history)
dev.off()

plot(model_history)