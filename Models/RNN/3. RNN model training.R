# =========================================================
# 03_fit_best_rnn_model.R
# Train the best RNN model and evaluate on test set
# =========================================================

library(dplyr)
library(tensorflow)
library(keras3)

set.seed(15)

# -------------------------
# Best hyperparameters
# -------------------------
best_param <- tibble(
  regrate = 1e-06,
  lstmunits = 256,
  neuron1 = 64,
  batchsize = 100,
  best_epoch = 15
)

# -------------------------
# Input shape
# -------------------------
input_shape_use <- c(dim(x_train)[2], dim(x_train)[3])

# -------------------------
# Class weights
# -------------------------
class_counts <- table(y_train)
cw <- as.numeric(class_counts["Alewife"] / class_counts["Rainbow Smelt"])
class_weight_list <- list("0" = 1, "1" = cw)

print(class_counts)
print(class_weight_list)

# -------------------------
# Build model
# -------------------------
set_random_seed(15)

rnn <- keras_model_sequential() %>%
  layer_lstm(
    input_shape = input_shape_use,
    units = best_param$lstmunits
  ) %>%
  layer_activation_leaky_relu() %>%
  layer_batch_normalization() %>%
  layer_dense(
    units = best_param$neuron1,
    activity_regularizer = regularizer_l2(l = best_param$regrate)
  ) %>%
  layer_activation_leaky_relu() %>%
  layer_dense(units = 2, activation = "softmax")

# -------------------------
# Compile
# -------------------------
rnn %>% compile(
  loss = loss_categorical_crossentropy(),
  optimizer = optimizer_adam(learning_rate = 1e-4),
  metrics = c("accuracy", metric_auc(name = "auc"))
)

# -------------------------
# Fit best model
# -------------------------
model_history <- rnn %>% fit(
  x_train, dummy_y_train,
  batch_size = best_param$batchsize,
  epochs = best_param$best_epoch,
  validation_data = list(x_validate, dummy_y_val),
  class_weight = class_weight_list,
  verbose = 1
)

# -------------------------
# Evaluate on test set
# -------------------------
test_results <- rnn %>% evaluate(
  x_test, dummy_y_test,
  verbose = 0
)

# -------------------------
# Predict on test set
# -------------------------
test_pred_prob <- rnn %>% predict(x_test)
test_pred_class <- max.col(test_pred_prob) - 1
test_true_class <- max.col(dummy_y_test) - 1

# map 0/1 back to species names
pred_species <- factor(
  ifelse(test_pred_class == 0, "Alewife", "Rainbow Smelt"),
  levels = c("Alewife", "Rainbow Smelt")
)

true_species <- factor(
  ifelse(test_true_class == 0, "Alewife", "Rainbow Smelt"),
  levels = c("Alewife", "Rainbow Smelt")
)

# -------------------------
# Confusion matrix
# -------------------------
cm <- table(Predicted = pred_species, True = true_species)

cat("\nConfusion matrix:\n")
print(cm)

# -------------------------
# Derived metrics
# Take Rainbow Smelt as the positive class
# -------------------------
TN <- cm["Alewife", "Alewife"]
FN <- cm["Alewife", "Rainbow Smelt"]
FP <- cm["Rainbow Smelt", "Alewife"]
TP <- cm["Rainbow Smelt", "Rainbow Smelt"]

accuracy <- (TP + TN) / sum(cm)
sensitivity_smelt <- TP / (TP + FN)
specificity_alewife <- TN / (TN + FP)
precision_smelt <- TP / (TP + FP)
balanced_accuracy <- (sensitivity_smelt + specificity_alewife) / 2

# -------------------------
# Performance table
# -------------------------
performance_table <- data.frame(
  Metric = c(
    "Accuracy",
    "AUC (Area Under the ROC Curve)",
    "Loss",
    "Balanced Accuracy"
  ),
  Value = round(c(
    test_results$accuracy,
    test_results$auc,
    test_results$loss,
    balanced_accuracy
  ), 3)
)

cat("\nTest performance table:\n")
print(performance_table)

# -------------------------
# Additional interpretation metrics
# -------------------------
cat("\nDerived metrics:\n")
cat("Sensitivity (Smelt):", round(sensitivity_smelt, 3), "\n")
cat("Specificity (Alewife):", round(specificity_alewife, 3), "\n")
cat("Precision (Smelt):", round(precision_smelt, 3), "\n")



# -------------------------
# Track-level prediction by majority vote
# -------------------------
# -------------------------
# 5. Track-level majority vote
# Use average Smelt probability to break ties
# -------------------------
seq_pred_df <- test_seq_info %>%
  mutate(
    pred_class = test_pred_class,
    true_class = test_true_class,
    pred_species = ifelse(pred_class == 0, "Alewife", "Rainbow Smelt"),
    true_species = ifelse(true_class == 0, "Alewife", "Rainbow Smelt"),
    smelt_prob = test_pred_prob[, 2]
  )

track_pred_df <- seq_pred_df %>%
  group_by(track_id) %>%
  summarise(
    true_species = first(true_species),
    n_seq = n(),
    n_pred_alewife = sum(pred_species == "Alewife"),
    n_pred_smelt = sum(pred_species == "Rainbow Smelt"),
    mean_smelt_prob = mean(smelt_prob),
    pred_species = case_when(
      n_pred_smelt > n_pred_alewife ~ "Rainbow Smelt",
      n_pred_alewife > n_pred_smelt ~ "Alewife",
      mean_smelt_prob > 0.5 ~ "Rainbow Smelt",
      TRUE ~ "Alewife"
    ),
    .groups = "drop"
  ) %>%
  mutate(
    pred_species = factor(pred_species, levels = c("Alewife", "Rainbow Smelt")),
    true_species = factor(true_species, levels = c("Alewife", "Rainbow Smelt"))
  )

# -------------------------
# 6. Track-level confusion matrix
# -------------------------
track_cm <- table(
  Predicted = track_pred_df$pred_species,
  True = track_pred_df$true_species
)

cat("\nTrack-level confusion matrix:\n")
print(track_cm)

# -------------------------
# 7. Track-level metrics
# Take Rainbow Smelt as the positive class
# -------------------------
track_TN <- track_cm["Alewife", "Alewife"]
track_FN <- track_cm["Alewife", "Rainbow Smelt"]
track_FP <- track_cm["Rainbow Smelt", "Alewife"]
track_TP <- track_cm["Rainbow Smelt", "Rainbow Smelt"]

track_accuracy <- (track_TP + track_TN) / sum(track_cm)
track_sensitivity_smelt <- track_TP / (track_TP + track_FN)
track_specificity_alewife <- track_TN / (track_TN + track_FP)
track_precision_smelt <- track_TP / (track_TP + track_FP)
track_balanced_accuracy <- (track_sensitivity_smelt + track_specificity_alewife) / 2

track_performance_table <- data.frame(
  Metric = c(
    "Accuracy",
    "Sensitivity (Smelt)",
    "Specificity (Alewife)",
    "Precision (Smelt)",
    "Balanced Accuracy"
  ),
  Value = round(c(
    track_accuracy,
    track_sensitivity_smelt,
    track_specificity_alewife,
    track_precision_smelt,
    track_balanced_accuracy
  ), 3)
)

cat("\nTrack-level performance table:\n")
print(track_performance_table)

