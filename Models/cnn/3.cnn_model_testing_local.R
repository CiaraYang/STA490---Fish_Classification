library(dplyr)
library(tensorflow)
library(caret)
library(keras3)
library(pROC)
library(tibble)

# output folder
test_dir <- "Models/cnn/drac/test"
if (!dir.exists(test_dir)) dir.create(test_dir, recursive = TRUE)

# load data
x_train <- readRDS("Data/x_train.rds")
x_validate <- readRDS("Data/x_validate.rds")
x_test <- readRDS("Data/x_test.rds")

y_train <- readRDS("Data/y_train.rds")
y_validate <- readRDS("Data/y_validate.rds")
y_test <- readRDS("Data/y_test.rds")

dummy_y_train <- readRDS("Data/dummy_y_train.rds")
dummy_y_val <- readRDS("Data/dummy_y_val.rds")
dummy_y_test <- readRDS("Data/dummy_y_test.rds")

# load tuning result and best parameter
load("Data/cnn_tuning_results.RData")

# class weights
train_tab <- table(y_train)
class_weights_vec <- as.numeric(sum(train_tab) / (length(train_tab) * train_tab))
class_weights <- as.list(class_weights_vec)
names(class_weights) <- as.character(0:(length(class_weights) - 1))

# early stopping
callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
    min_delta = 1e-3,
    patience = 30,
    restore_best_weights = TRUE
  )
)

# build cnn
build_cnn_model <- function(param, input_length) {
  keras_model_sequential() %>%
    layer_conv_1d(
      input_shape = c(input_length, 1),
      filters = param$filters1,
      kernel_size = param$kernel_size,
      activation = "relu",
      padding = "same"
    ) %>%
    layer_dropout(rate = param$droprate1) %>%
    layer_batch_normalization() %>%
    
    layer_conv_1d(
      filters = param$filters2,
      kernel_size = param$kernel_size,
      activation = "relu",
      padding = "same"
    ) %>%
    layer_dropout(rate = param$droprate2) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    
    layer_conv_1d(
      filters = param$filters3,
      kernel_size = param$kernel_size,
      activation = "relu",
      padding = "same"
    ) %>%
    layer_dropout(rate = param$droprate3) %>%
    layer_batch_normalization() %>%
    
    layer_conv_1d(
      filters = param$filters4,
      kernel_size = param$kernel_size,
      activation = "relu",
      padding = "same"
    ) %>%
    layer_dropout(rate = param$droprate4) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_batch_normalization() %>%
    
    layer_conv_1d(
      filters = param$filters5,
      kernel_size = param$kernel_size,
      activation = "relu",
      padding = "same"
    ) %>%
    layer_dropout(rate = param$droprate5) %>%
    layer_batch_normalization() %>%
    
    layer_flatten() %>%
    layer_dense(units = 2, activation = "softmax")
}

# fit best model
input_length <- dim(x_train)[2]

set_random_seed(15)

cnn <- build_cnn_model(best_param, input_length)

cnn %>% compile(
  optimizer = optimizer_adam(learning_rate = 1e-4),
  loss = "categorical_crossentropy",
  metrics = list(
    metric_auc(name = "auc"),
    metric_categorical_accuracy(name = "accuracy")
  )
)

history <- cnn %>% fit(
  x = x_train,
  y = dummy_y_train,
  batch_size = best_param$batch_size,
  epochs = 150,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = class_weights,
  callbacks = callbacks,
  verbose = 1
)

# evaluate on test set
eval <- cnn %>% evaluate(
  x_test, dummy_y_test,
  verbose = 0,
  return_dict = TRUE
)

pred_probs <- cnn %>% predict(x_test, verbose = 0)

prob_smelt <- pred_probs[, 2]

true_labels <- factor(
  ifelse(dummy_y_test[, 1] == 1, "Alewife", "Rainbow Smelt"),
  levels = c("Alewife", "Rainbow Smelt")
)

# ROC and AUC
roc_obj <- roc(
  response = true_labels,
  predictor = prob_smelt,
  levels = c("Alewife", "Rainbow Smelt"),
  direction = "<"
)

test_auc <- as.numeric(auc(roc_obj))
best_epoch <- which.min(history$metrics$val_loss)

# try all thresholds from 0.00 to 1.00
thresholds <- seq(0, 1, by = 0.01)

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

print(threshold_results)

# select best threshold by balanced accuracy
best_row <- which.max(threshold_results$balanced_accuracy)
best_threshold <- threshold_results$threshold[best_row]

cat("Best threshold:", best_threshold, "\n")

# final prediction using best threshold
species_pred <- factor(
  ifelse(prob_smelt >= best_threshold, "Rainbow Smelt", "Alewife"),
  levels = c("Alewife", "Rainbow Smelt")
)

cm <- confusionMatrix(
  data = species_pred,
  reference = true_labels,
  positive = "Rainbow Smelt"
)

best_result <- tibble(
  val_loss = min(history$metrics$val_loss),
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
  f1 = as.numeric(cm$byClass["F1"]),
  filters1 = best_param$filters1,
  filters2 = best_param$filters2,
  filters3 = best_param$filters3,
  filters4 = best_param$filters4,
  filters5 = best_param$filters5,
  kernel_size = best_param$kernel_size,
  batch_size = best_param$batch_size,
  droprate1 = best_param$droprate1,
  droprate2 = best_param$droprate2,
  droprate3 = best_param$droprate3,
  droprate4 = best_param$droprate4,
  droprate5 = best_param$droprate5
)

print(best_param)
print(best_result)
print(cm)

write.csv(best_result, file.path(test_dir, "best_model_result.csv"), row.names = FALSE)
write.csv(threshold_results, file.path(test_dir, "threshold_results.csv"), row.names = FALSE)

saveRDS(best_result, file.path(test_dir, "best_model_result.rds"))
saveRDS(threshold_results, file.path(test_dir, "threshold_results.rds"))

save(
  cnn, history, cm, best_param, best_result, threshold_results, pred_probs, roc_obj,
  file = file.path(test_dir, "best_model_result.RData")
)

ggplot(threshold_results, aes(x = threshold)) +
  geom_line(aes(y = sensitivity, color = "Sensitivity"), size = 1) +
  geom_line(aes(y = specificity, color = "Specificity"), size = 1) +
  geom_line(aes(y = balanced_accuracy, color = "Balanced Accuracy"), size = 1) +
  labs(
    title = "Threshold vs Performance Metrics",
    x = "Threshold",
    y = "Metric Value",
    color = "Metric"
  ) +
  geom_vline(xintercept = best_threshold, linetype = "dashed", color = "grey") +
  labs(
    title = "Threshold vs Balanced Accuracy",
    x = "Threshold",
    y = "Balanced Accuracy"
  ) +
  theme_minimal()
