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

# tune threshold on validation set
val_probs <- dnn %>% predict(x_validate, verbose = 0)
prob_smelt_val <- as.numeric(val_probs)

thresholds <- seq(0.1, 0.9, by = 0.01)

threshold_results <- data.frame()

for (t in thresholds) {
  pred_val <- factor(
    ifelse(prob_smelt_val >= t, "Rainbow Smelt", "Alewife"),
    levels = c("Alewife", "Rainbow Smelt")
  )
  
  true_val <- factor(
    ifelse(y_validate == 1, "Rainbow Smelt", "Alewife"),
    levels = c("Alewife", "Rainbow Smelt")
  )
  
  cm_val <- confusionMatrix(
    data = pred_val,
    reference = true_val,
    positive = "Rainbow Smelt"
  )
  
  threshold_results <- rbind(
    threshold_results,
    data.frame(
      threshold = t,
      sensitivity = as.numeric(cm_val$byClass["Sensitivity"]),
      specificity = as.numeric(cm_val$byClass["Specificity"]),
      balanced_accuracy = as.numeric(cm_val$byClass["Balanced Accuracy"])
    )
  )
}

print(threshold_results)

best_threshold <- threshold_results$threshold[
  which.max(threshold_results$balanced_accuracy)
]

cat("Best threshold:", best_threshold, "\n")

# predict on test set
pred_probs <- dnn %>% predict(x_test, verbose = 0)
prob_smelt <- as.numeric(pred_probs)

species_pred <- factor(
  ifelse(prob_smelt >= best_threshold, "Rainbow Smelt", "Alewife"),
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
  theme_minimal()
