#server version
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

# load top 20 configs
top20_models <- readRDS(
  "Models/cnn/drac/training/training_metrics_full/top20_configs.rds"
)

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

# run models
input_length <- dim(x_train)[2]
all_results <- list()

for (i in seq_len(nrow(top20_models))) {
  
  cat("Running model", i, "\n")
  
  param <- top20_models[i, ]
  
  set_random_seed(15)
  
  cnn <- build_cnn_model(param, input_length)
  
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
    batch_size = param$batch_size,
    epochs = 200,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = class_weights,
    callbacks = callbacks,
    verbose = 1
  )
  
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
  
  best_epoch <- which.min(history$metrics$val_loss)
  
  metrics_tbl <- tibble(
    model = i,
    val_loss = min(history$metrics$val_loss),
    val_auc = history$metrics$val_auc[best_epoch],
    test_loss = eval[["loss"]],
    test_auc_keras = eval[["auc"]],
    test_accuracy_keras = eval[["accuracy"]],
    test_auc_pROC = test_auc,
    best_threshold = best_threshold,
    accuracy = cm$overall["Accuracy"],
    sensitivity = cm$byClass["Sensitivity"],
    specificity = cm$byClass["Specificity"],
    balanced_accuracy = cm$byClass["Balanced Accuracy"],
    precision = cm$byClass["Precision"],
    recall = cm$byClass["Recall"],
    f1 = cm$byClass["F1"]
  )
  
  print(metrics_tbl)
  
  all_results[[i]] <- metrics_tbl
  
  write.csv(
    metrics_tbl,
    file.path(test_dir, paste0("model_", i, ".csv")),
    row.names = FALSE
  )
}

all_results_tbl <- bind_rows(all_results)

all_results_tbl <- all_results_tbl %>%
  arrange(desc(test_auc_pROC))

print(all_results_tbl)

write.csv(
  all_results_tbl,
  file.path(test_dir, "top20_results.csv"),
  row.names = FALSE
)

saveRDS(
  all_results_tbl,
  file.path(test_dir, "top20_results.rds")
)