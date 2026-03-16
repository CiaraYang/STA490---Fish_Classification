library(dplyr)
library(keras3)
library(tensorflow)
library(caret)
library(pROC)
library(tibble)

set_random_seed(15)

best_param <- tuning_results[1, ]

input_dim <- ncol(x_train_dnn)

train_tab <- table(y_train)

class_weights <- list()
class_weights[["0"]] <- as.numeric(sum(train_tab) / (2 * train_tab["Alewife"]))
class_weights[["1"]] <- as.numeric(sum(train_tab) / (2 * train_tab["Rainbow Smelt"]))

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
  layer_dense(units = 2, activation = "softmax")

final_model |>
  compile(
    optimizer = optimizer_adam(learning_rate = best_param$lr),
    loss = "categorical_crossentropy",
    metrics = c("accuracy", metric_auc(name = "auc"))
  )

final_callbacks <- list(
  callback_early_stopping(
    monitor = "val_auc",
    mode = "max",
    patience = 10,
    restore_best_weights = TRUE
  ),
  callback_reduce_lr_on_plateau(
    monitor = "val_auc",
    mode = "max",
    factor = 0.5,
    patience = 4,
    min_lr = 1e-5
  )
)

final_history <- final_model |>
  fit(
    x = x_train_dnn,
    y = dummy_y_train,
    validation_data = list(x_validate_dnn, dummy_y_val),
    epochs = 100,
    batch_size = best_param$batch_size,
    class_weight = class_weights,
    callbacks = final_callbacks,
    verbose = 1
  )

eval_result <- final_model |>
  evaluate(x_test_dnn, dummy_y_test, verbose = 0)

print(eval_result)

pred_probs <- final_model |>
  predict(x_test_dnn, verbose = 0)

prob_smelt <- pred_probs[, 2]

pred_class_idx <- apply(pred_probs, 1, which.max)

species_pred <- factor(
  ifelse(pred_class_idx == 1, "Alewife", "Rainbow Smelt"),
  levels = c("Alewife", "Rainbow Smelt")
)

true_labels <- factor(y_test, levels = c("Alewife", "Rainbow Smelt"))

cm <- confusionMatrix(
  data = species_pred,
  reference = true_labels,
  positive = "Rainbow Smelt"
)

print(cm)

roc_obj <- roc(
  response = true_labels,
  predictor = prob_smelt,
  levels = c("Alewife", "Rainbow Smelt"),
  direction = "<"
)

test_auc <- as.numeric(auc(roc_obj))
print(test_auc)

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