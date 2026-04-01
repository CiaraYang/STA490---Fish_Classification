# local version
library(dplyr)
library(keras3)
library(tensorflow)
library(caret)
library(pROC)
library(tibble)

set_random_seed(15)

# load processed data from Data/
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

# select best hyperparameters from tuning
best_param <- tuning_results[1, ]

# input feature dimension
input_dim <- ncol(x_train)

# focal loss parameters
class_counts <- table(y_train)
alpha_pos <- as.numeric(class_counts["0"] / sum(class_counts))
gamma_val <- 2.0

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

# build final DNN model
final_model <- keras_model_sequential() |>
  layer_dense(
    units = best_param$hidden1,
    input_shape = c(input_dim),
    kernel_regularizer = regularizer_l2(best_param$l2)
  ) |>
  layer_batch_normalization() |>
  layer_activation("relu") |>
  layer_dropout(rate = best_param$dropout) |>
  layer_dense(
    units = best_param$hidden2,
    kernel_regularizer = regularizer_l2(best_param$l2)
  ) |>
  layer_batch_normalization() |>
  layer_activation("relu") |>
  layer_dropout(rate = best_param$dropout) |>
  layer_dense(units = 1, activation = "sigmoid")

# compile model
final_model |>
  compile(
    optimizer = optimizer_adam(learning_rate = best_param$lr),
    loss = binary_focal_loss(alpha = alpha_pos, gamma = gamma_val),
    metrics = c("accuracy", metric_auc(name = "auc"))
  )

# callbacks
final_callbacks <- list(
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

# train final model
final_history <- final_model |>
  fit(
    x = x_train,
    y = y_train,
    validation_data = list(x_validate, y_validate),
    epochs = 100,
    batch_size = best_param$batch_size,
    callbacks = final_callbacks,
    verbose = 1
  )

# evaluate on test set
eval_result <- final_model |>
  evaluate(x_test, y_test, verbose = 0, return_dict = TRUE)
print(eval_result)

# predict probabilities
pred_probs <- final_model |>
  predict(x_test, verbose = 0)

# extract probability of Rainbow Smelt
prob_smelt <- as.numeric(pred_probs)

# convert probabilities to predicted class
species_pred <- factor(
  ifelse(prob_smelt >= 0.5, "Rainbow Smelt", "Alewife"),
  levels = c("Alewife", "Rainbow Smelt")
)

# create true labels
true_labels <- factor(
  ifelse(y_test == 1, "Rainbow Smelt", "Alewife"),
  levels = c("Alewife", "Rainbow Smelt")
)

# confusion matrix
cm <- confusionMatrix(
  data = species_pred,
  reference = true_labels,
  positive = "Rainbow Smelt"
)
print(cm)

# ROC and AUC
roc_obj <- roc(
  response = true_labels,
  predictor = prob_smelt,
  levels = c("Alewife", "Rainbow Smelt"),
  direction = "<"
)

test_auc <- as.numeric(auc(roc_obj))
print(test_auc)

# best validation epoch based on loss
best_epoch <- which.min(final_history$metrics$val_loss)

# summarize evaluation metrics
metrics_tbl_dnn <- tibble(
  alpha_pos = alpha_pos,
  gamma = gamma_val,
  best_epoch = best_epoch,
  best_val_loss = final_history$metrics$val_loss[best_epoch],
  best_val_acc = final_history$metrics$val_accuracy[best_epoch],
  best_val_auc = final_history$metrics$val_auc[best_epoch],
  test_loss = as.numeric(eval_result[["loss"]]),
  accuracy = as.numeric(eval_result[["accuracy"]]),
  auc_from_keras = as.numeric(eval_result[["auc"]]),
  auc_from_pROC = test_auc,
  sensitivity = as.numeric(cm$byClass["Sensitivity"]),
  specificity = as.numeric(cm$byClass["Specificity"]),
  balanced_accuracy = as.numeric(cm$byClass["Balanced Accuracy"]),
  precision = as.numeric(cm$byClass["Precision"]),
  recall = as.numeric(cm$byClass["Recall"]),
  f1 = as.numeric(cm$byClass["F1"])
)

print(metrics_tbl_dnn)