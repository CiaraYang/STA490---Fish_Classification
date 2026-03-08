# =========================================================
# 03_train_final_cnn.R
# Train final 1D CNN and evaluate on test set
# =========================================================

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)
library(keras3)
library(pROC)

## ---- Load prepared data and best tuning result ----
load("Data/fish_cnn_data.RData")
load("Data/cnn_tuning_results.RData")

## ---- Inspect best parameters ----
print(best_param)

## ---- Class weights ----
train_tab <- table(y_train)

# levels were set as c("Alewife", "Rainbow Smelt")
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

## ---- Input length ----
input_length <- dim(x_train)[2]

## ---- Build final model ----
set_random_seed(15)

cnn <- keras_model_sequential() %>%
  layer_conv_1d(
    input_shape = c(input_length, 1),
    filters = best_param$filters1,
    kernel_size = best_param$kernel_size,
    activation = "relu",
    padding = "same",
    strides = 1
  ) %>%
  layer_dropout(rate = best_param$droprate1) %>%
  layer_batch_normalization() %>%
  
  layer_conv_1d(
    filters = best_param$filters2,
    kernel_size = best_param$kernel_size,
    activation = "relu",
    padding = "same",
    strides = 1
  ) %>%
  layer_dropout(rate = best_param$droprate2) %>%
  layer_batch_normalization() %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  
  layer_conv_1d(
    filters = best_param$filters3,
    kernel_size = best_param$kernel_size,
    activation = "relu",
    padding = "same",
    strides = 1
  ) %>%
  layer_dropout(rate = best_param$droprate3) %>%
  layer_batch_normalization() %>%
  
  layer_conv_1d(
    filters = best_param$filters4,
    kernel_size = best_param$kernel_size,
    activation = "relu",
    padding = "same",
    strides = 1
  ) %>%
  layer_dropout(rate = best_param$droprate4) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_batch_normalization() %>%
  
  layer_conv_1d(
    filters = best_param$filters5,
    kernel_size = best_param$kernel_size,
    activation = "relu",
    padding = "same",
    strides = 1
  ) %>%
  layer_dropout(rate = best_param$droprate5) %>%
  layer_batch_normalization() %>%
  
  layer_flatten() %>%
  layer_dense(units = 2, activation = "softmax")

cnn %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-4),
  loss = "binary_crossentropy",
  metrics = list(
    metric_auc(name = "auc")
  )
)

summary(cnn)

## ---- Train final model ----
cnn_history <- cnn %>% fit(
  x = x_train,
  y = dummy_y_train,
  batch_size = best_param$batch_size,
  epochs = 200,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = class_weights,
  callbacks = callbacks,
  verbose = 1
)

## ---- Evaluate on test set ----
eval <- cnn %>% evaluate(x_test, dummy_y_test, verbose = 0)
print(eval)

## ---- Predict on test set ----
pred_probs <- cnn %>% predict(x_test)

# class-2 probability = Rainbow Smelt probability
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