#drac version
library(dplyr)
library(tensorflow)
library(caret)
library(keras3)
library(pROC)
library(tibble)

# output folder
test_dir <- "Models/dnn/drac/test"
if (!dir.exists(test_dir)) dir.create(test_dir, recursive = TRUE)

# load data
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

# load top 20 configs
top20_models <- readRDS("Models/dnn/drac/training/training_metrics_full/top20_dnn_loss_with_config.rds")

# class weights
train_tab <- table(y_train)
class_weights_vec <- as.numeric(sum(train_tab) / (length(train_tab) * train_tab))
class_weights <- as.list(class_weights_vec)
names(class_weights) <- as.character(0:(length(class_weights) - 1))

# callbacks
callbacks <- list(
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
    patience = 30,
    min_lr = 1e-5
  )
)

# build dnn
build_dnn_model <- function(param, input_dim) {
  keras_model_sequential() %>%
    layer_dense(
      units = param$hidden1,
      input_shape = c(input_dim),
      kernel_regularizer = regularizer_l2(param$l2)
    ) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_dropout(rate = param$dropout) %>%
    layer_dense(
      units = param$hidden2,
      kernel_regularizer = regularizer_l2(param$l2)
    ) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_dropout(rate = param$dropout) %>%
    layer_dense(units = 1, activation = "sigmoid")
}

# run models
input_dim <- ncol(x_train)
all_results <- list()

for (i in seq_len(nrow(top20_models))) {
  
  cat("Running model", i, "of", nrow(top20_models), "\n")
  
  param <- top20_models[i, ]
  
  set_random_seed(15)
  
  dnn <- build_dnn_model(param, input_dim)
  
  dnn %>% compile(
    optimizer = optimizer_adam(learning_rate = param$lr),
    loss = "binary_crossentropy",
    metrics = list(
      metric_auc(name = "auc"),
      metric_binary_accuracy(name = "accuracy")
    )
  )
  
  history <- dnn %>% fit(
    x = x_train,
    y = y_train,
    batch_size = param$batch_size,
    epochs = 100,
    validation_data = list(x_validate, y_validate),
    class_weight = class_weights,
    callbacks = callbacks,
    verbose = 1
  )
  
  eval <- dnn %>% evaluate(
    x_test, y_test,
    verbose = 0,
    return_dict = TRUE
  )
  
  pred_probs <- dnn %>% predict(x_test, verbose = 0)
  prob_smelt <- as.numeric(pred_probs)
  
  species_pred <- factor(
    ifelse(prob_smelt >= 0.5, "Rainbow Smelt", "Alewife"),
    levels = c("Alewife", "Rainbow Smelt")
  )
  
  true_labels <- factor(
    ifelse(y_test == 1, "Rainbow Smelt", "Alewife"),
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
  
  best_epoch <- which.max(history$metrics$val_auc)
  
  metrics_tbl <- tibble(
    rank = i,
    model_id = param$model_id,
    hidden1 = param$hidden1,
    hidden2 = param$hidden2,
    dropout = param$dropout,
    l2 = param$l2,
    lr = param$lr,
    batch_size = param$batch_size,
    tuning_val_loss = if ("val_loss" %in% names(param)) param$val_loss else NA_real_,
    tuning_val_auc = if ("val_auc" %in% names(param)) param$val_auc else NA_real_,
    best_epoch = best_epoch,
    retrain_val_loss = min(history$metrics$val_loss),
    retrain_val_auc = max(history$metrics$val_auc),
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
    f1 = as.numeric(cm$byClass["F1"])
  )
  
  print(metrics_tbl)
  
  all_results[[i]] <- metrics_tbl
  
  saveRDS(
    metrics_tbl,
    file.path(test_dir, paste0("model_", i, ".rds"))
  )
}

all_results_tbl <- bind_rows(all_results) %>%
  arrange(desc(test_auc_pROC))

print(all_results_tbl)

saveRDS(
  all_results_tbl,
  file.path(test_dir, "top20_dnn_retrain_results.rds")
)

