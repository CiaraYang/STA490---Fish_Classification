library(dplyr)
library(keras3)
library(tensorflow)
library(caret)
library(pROC)
library(tibble)

set_random_seed(15)

# select best hyperparameters from tuning
best_param <- tuning_results[1, ]

# input feature dimension
input_dim <- ncol(x_train)

# compute class weights to handle imbalance
train_tab <- table(y_train)
class_weights <- list()
class_weights[["0"]] <- as.numeric(sum(train_tab) / (2 * train_tab["0"]))
class_weights[["1"]] <- as.numeric(sum(train_tab) / (2 * train_tab["1"]))

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

# compile model with binary classification setup
final_model |>
  compile(
    optimizer = optimizer_adam(learning_rate = best_param$lr),
    loss = "binary_crossentropy",
    metrics = c("accuracy", metric_auc(name = "auc"))
  )

# callbacks for early stopping and learning rate reduction
final_callbacks <- list(
  callback_early_stopping(
    monitor = "val_auc",
    mode = "max",
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

# train final model
final_history <- final_model |>
  fit(
    x = x_train,
    y = y_train,
    validation_data = list(x_validate, y_validate),
    epochs = 100,
    batch_size = best_param$batch_size,
    class_weight = class_weights,
    callbacks = final_callbacks,
    verbose = 1
  )

# evaluate on test set
eval_result <- final_model |>
  evaluate(x_test, y_test, verbose = 0)
print(eval_result)

# predict probabilities
pred_probs <- final_model |>
  predict(x_test, verbose = 0)

# extract probability of Rainbow Smelt
prob_smelt <- as.numeric(pred_probs)

# convert probabilities to predicted class using threshold 0.5
species_pred <- factor(
  ifelse(prob_smelt >= 0.5, "Rainbow Smelt", "Alewife"),
  levels = c("Alewife", "Rainbow Smelt")
)

# create true labels in same format
true_labels <- factor(
  ifelse(y_test == 1, "Rainbow Smelt", "Alewife"),
  levels = c("Alewife", "Rainbow Smelt")
)

# confusion matrix for classification performance
cm <- confusionMatrix(
  data = species_pred,
  reference = true_labels,
  positive = "Rainbow Smelt"
)
print(cm)

# compute ROC curve and AUC
roc_obj <- roc(
  response = true_labels,
  predictor = prob_smelt,
  levels = c("Alewife", "Rainbow Smelt"),
  direction = "<"
)

test_auc <- as.numeric(auc(roc_obj))
print(test_auc)

# summarize evaluation metrics
metrics_tbl_dnn <- tibble(
  loss = as.numeric(eval_result[["loss"]]),
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