library(dplyr)
library(tensorflow)
library(caret)
library(keras3)
library(pROC)
library(tibble)

set_random_seed(15)

# load data
x_train <- readRDS("Data/x_train.rds")
x_validate <- readRDS("Data/x_validate.rds")
x_test <- readRDS("Data/x_test.rds")

y_train <- readRDS("Data/y_train.rds")
y_validate <- readRDS("Data/y_validate.rds")
y_test <- readRDS("Data/y_test.rds")

# reshape
x_train <- matrix(x_train[, , 1], nrow = dim(x_train)[1], ncol = dim(x_train)[2])
x_validate <- matrix(x_validate[, , 1], nrow = dim(x_validate)[1], ncol = dim(x_validate)[2])
x_test <- matrix(x_test[, , 1], nrow = dim(x_test)[1], ncol = dim(x_test)[2])

y_train <- as.numeric(y_train)
y_validate <- as.numeric(y_validate)
y_test <- as.numeric(y_test)

# load best model config 
top20_models <- readRDS("Models/dnn/drac/training/training_metrics_full/top20_dnn_loss_with_config.rds")
param <- top20_models[1, ]   

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
    patience = 30,
    restore_best_weights = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_loss",
    factor = 0.5,
    patience = 5,
    min_lr = 1e-5
  )
)

# build model
build_dnn_model <- function(param, input_dim) {
  keras_model_sequential(input_shape = c(input_dim)) %>%
    layer_dense(units = param$hidden1, kernel_regularizer = regularizer_l2(param$l2)) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_dropout(rate = param$dropout) %>%
    layer_dense(units = param$hidden2, kernel_regularizer = regularizer_l2(param$l2)) %>%
    layer_batch_normalization() %>%
    layer_activation("relu") %>%
    layer_dropout(rate = param$dropout) %>%
    layer_dense(units = 1, activation = "sigmoid")
}

input_dim <- ncol(x_train)

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

# train
history <- dnn %>% fit(
  x_train, y_train,
  batch_size = as.integer(param$batch_size),
  epochs = 200,
  validation_data = list(x_validate, y_validate),
  callbacks = callbacks,
  verbose = 1
)

# predict
pred_probs <- dnn %>% predict(x_test)
prob_smelt <- as.numeric(pred_probs)

species_pred <- factor(
  ifelse(prob_smelt >= 0.5, "Rainbow Smelt", "Alewife"),
  levels = c("Alewife", "Rainbow Smelt")
)

true_labels <- factor(
  ifelse(y_test == 1, "Rainbow Smelt", "Alewife"),
  levels = c("Alewife", "Rainbow Smelt")
)

# result print
cm <- confusionMatrix(
  data = species_pred,
  reference = true_labels,
  positive = "Rainbow Smelt"
)

print(cm)
