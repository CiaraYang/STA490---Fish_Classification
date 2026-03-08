# =========================================================
# 03_train_final_resnet.R
# Train final 1D ResNet and evaluate on test set
# =========================================================

library(dplyr)
library(tensorflow)
library(caret)
library(keras3)
library(pROC)
library(tibble)
library(tidyr)
library(ggplot2)

## ---- Load data ----
load("Data/fish_cnn_data.RData")
load("Data/resnet_tuning_results.RData")

print(best_param_resnet)

## ---- Class weights ----
train_tab <- table(y_train)
cw <- as.numeric(train_tab[1] / train_tab[2])
class_weights <- list("0" = 1, "1" = cw)

## ---- Early stopping ----
callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
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

## ---- Build model ----
build_resnet_model <- function(input_length,
                               filters1,
                               filters2,
                               kernel_size,
                               dropout_rate,
                               dense_units) {
  
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
    optimizer = optimizer_adam(learning_rate = 1e-4),
    loss = "binary_crossentropy",
    metrics = list(
      metric_auc(name = "auc"),
      metric_binary_accuracy(name = "accuracy")
    )
  )
  
  model
}

## ---- Create model ----
input_length <- dim(x_train)[2]

set_random_seed(15)

resnet_model <- build_resnet_model(
  input_length = input_length,
  filters1 = best_param_resnet$filters1,
  filters2 = best_param_resnet$filters2,
  kernel_size = best_param_resnet$kernel_size,
  dropout_rate = best_param_resnet$dropout_rate,
  dense_units = best_param_resnet$dense_units
)

summary(resnet_model)

## ---- Train final model ----
resnet_history <- resnet_model %>% fit(
  x = x_train,
  y = dummy_y_train,
  batch_size = best_param_resnet$batch_size,
  epochs = 200,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = class_weights,
  callbacks = callbacks,
  verbose = 1
)

## ---- Evaluate on test set ----
eval <- resnet_model %>% evaluate(x_test, dummy_y_test, verbose = 0)
print(eval)

## ---- Predict ----
pred_probs <- resnet_model %>% predict(x_test)
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

metrics_tbl_resnet <- tibble(
  loss = as.numeric(eval[["loss"]]),
  auc_from_keras = as.numeric(eval[["auc"]]),
  auc_from_pROC = test_auc,
  accuracy = as.numeric(cm$overall["Accuracy"]),
  sensitivity = as.numeric(cm$byClass["Sensitivity"]),
  specificity = as.numeric(cm$byClass["Specificity"]),
  balanced_accuracy = as.numeric(cm$byClass["Balanced Accuracy"]),
  precision = as.numeric(cm$byClass["Precision"]),
  recall = as.numeric(cm$byClass["Recall"]),
  f1 = as.numeric(cm$byClass["F1"])
)

print(metrics_tbl_resnet)

## ---- Training history table ----
lr_history <- rep(1e-4, length(resnet_history$metrics$loss))

history_df_resnet <- tibble(
  epoch = seq_along(resnet_history$metrics$loss),
  loss = resnet_history$metrics$loss,
  val_loss = resnet_history$metrics$val_loss,
  auc = resnet_history$metrics$auc,
  val_auc = resnet_history$metrics$val_auc,
  accuracy = resnet_history$metrics$accuracy,
  val_accuracy = resnet_history$metrics$val_accuracy,
  learning_rate = lr_history
)

print(history_df_resnet)

## ---- Plot training history ----
history_long_resnet <- history_df_resnet %>%
  pivot_longer(
    cols = -epoch,
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    panel = case_when(
      metric %in% c("loss", "val_loss") ~ "Loss",
      metric %in% c("auc", "val_auc") ~ "AUC",
      metric %in% c("accuracy", "val_accuracy") ~ "Accuracy",
      metric == "learning_rate" ~ "Learning Rate"
    )
  )

training_history_plot_resnet <- ggplot(history_long_resnet,
                                       aes(x = epoch, y = value, color = metric)) +
  geom_line(linewidth = 0.8) +
  geom_point(size = 1.2) +
  facet_wrap(~ panel, scales = "free_y", ncol = 1) +
  theme_bw() +
  labs(
    title = "ResNet Training History",
    x = "Epoch",
    y = NULL,
    color = "Metric"
  )

print(training_history_plot_resnet)

## ---- Save outputs ----
write.csv(metrics_tbl_resnet, "Data/resnet_test_metrics.csv", row.names = FALSE)
write.csv(history_df_resnet, "Data/resnet_training_history.csv", row.names = FALSE)

save(
  resnet_model, resnet_history, cm, metrics_tbl_resnet, pred_probs,
  history_df_resnet, training_history_plot_resnet,
  file = "Data/final_resnet_model_results.RData"
)