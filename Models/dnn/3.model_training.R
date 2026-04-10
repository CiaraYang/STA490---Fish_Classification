#server version
library(dplyr)
library(tensorflow)
library(caret)
library(keras3)
library(pROC)
library(tibble)

set_random_seed(15)

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

# make sure y is numeric 0/1
y_train <- as.numeric(y_train)
y_validate <- as.numeric(y_validate)
y_test <- as.numeric(y_test)

# load top 20 configs
top20_models <- readRDS("Models/dnn/drac/training/training_metrics_full/top20_dnn_loss_with_config.rds")

# focal loss
binary_focal_loss <- function(alpha = 0.75, gamma = 2.0) {
  function(y_true, y_pred) {
    y_true <- tf$cast(y_true, tf$float32)
    y_pred <- tf$cast(y_pred, tf$float32)
    
    eps <- tf$constant(1e-7, dtype = tf$float32)
    y_pred <- tf$clip_by_value(y_pred, eps, 1 - eps)
    
    alpha_t <- y_true * alpha + (1 - y_true) * (1 - alpha)
    p_t <- y_true * y_pred + (1 - y_true) * (1 - y_pred)
    
    loss <- -alpha_t * tf$pow(1 - p_t, gamma) * tf$math$log(p_t)
    tf$reduce_mean(loss)
  }
}

# callbacks
callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
    mode = "min",
    min_delta = 0.001,
    patience = 30,
    restore_best_weights = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    mode = "min",
    factor = 0.5,
    patience = 5,
    min_lr = 1e-5
  )
)

# build dnn
build_dnn_model <- function(param, input_dim) {
  keras_model_sequential(input_shape = c(input_dim)) %>%
    layer_dense(
      units = param$hidden1,
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
    loss = binary_focal_loss(
      alpha = as.numeric(param$alpha_pos),
      gamma = as.numeric(param$gamma)
    ),
    metrics = list(
      metric_auc(name = "auc"),
      metric_binary_accuracy(name = "accuracy")
    )
  )
  
  history <- dnn %>% fit(
    x = x_train,
    y = y_train,
    batch_size = as.integer(param$batch_size),
    epochs = 200,
    validation_data = list(x_validate, y_validate),
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
  
  best_epoch <- which.min(history$metrics$val_loss)
  
  metrics_tbl <- tibble(
    rank = i,
    model_id = as.integer(param$model_id),
    hidden1 = as.integer(param$hidden1),
    hidden2 = as.integer(param$hidden2),
    dropout = as.numeric(param$dropout),
    l2 = as.numeric(param$l2),
    lr = as.numeric(param$lr),
    batch_size = as.integer(param$batch_size),
    alpha_pos = as.numeric(param$alpha_pos),
    gamma = as.numeric(param$gamma),
    tuning_val_loss = if ("val_loss" %in% names(param)) as.numeric(param$val_loss) else NA_real_,
    tuning_val_auc = if ("val_auc" %in% names(param)) as.numeric(param$val_auc) else NA_real_,
    best_epoch = best_epoch,
    retrain_val_loss = min(history$metrics$val_loss),
    retrain_val_auc = max(history$metrics$val_auc),
    test_loss = as.numeric(eval[["loss"]]),
    test_auc_keras = as.numeric(eval[["auc"]]),
    test_accuracy_keras = as.numeric(eval[["accuracy"]]),
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

# saveRDS(
#   all_results_tbl,
#   file.path(test_dir, "top20_dnn_retrain_results.rds")
# )


