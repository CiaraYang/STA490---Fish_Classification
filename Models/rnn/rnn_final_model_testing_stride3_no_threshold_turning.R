# =========================================================
# 03_rnn_final_model.R
# Final RNN model with selected hyperparameters (model 501)
# =========================================================

library(dplyr)
library(tensorflow)
library(keras3)
library(abind)

# -------------------------
# Load data
# -------------------------
x_train        <- readRDS("Data/RNN_data/stride3/x_train_stride3.rds")
x_validate     <- readRDS("Data/RNN_data/stride3/x_validate_stride3.rds")
x_test         <- readRDS("Data/RNN_data/stride3/x_test_stride3.rds")
y_train        <- readRDS("Data/RNN_data/stride3/y_train_stride3.rds")
dummy_y_train  <- readRDS("Data/RNN_data/stride3/dummy_y_train_stride3.rds")
dummy_y_val    <- readRDS("Data/RNN_data/stride3/dummy_y_val_stride3.rds")
dummy_y_test   <- readRDS("Data/RNN_data/stride3/dummy_y_test_stride3.rds")

# -------------------------
# Combine train + validation
# -------------------------
x_trainval       <- abind(x_train, x_validate, along = 1)
dummy_y_trainval <- rbind(dummy_y_train, dummy_y_val)
input_shape_use  <- c(dim(x_trainval)[2], dim(x_trainval)[3])

# -------------------------
# Class weights (based on train only)
# -------------------------
class_counts <- table(y_train)
cw <- as.numeric(class_counts["Alewife"] / class_counts["Rainbow Smelt"])
class_weight_list <- list("0" = 1, "1" = cw)

# -------------------------
# Selected hyperparameters (model 501)
# -------------------------
lstmunits      <- 128
neuron1        <- 64
batchsize      <- 64
lr             <- 1e-04
dropout1       <- 0
regrate        <- 1e-05
use_batchnorm  <- FALSE
best_epochs    <- 37
model_id       <- 501

# -------------------------
# Build model
# -------------------------
set_random_seed(15)

rnn <- keras_model_sequential() %>%
  layer_lstm(
    input_shape = input_shape_use,
    units = lstmunits
  ) %>%
  layer_activation_leaky_relu()

if (use_batchnorm) {
  rnn <- rnn %>% layer_batch_normalization()
}

if (dropout1 > 0) {
  rnn <- rnn %>% layer_dropout(rate = dropout1)
}

rnn <- rnn %>%
  layer_dense(
    units = neuron1,
    activity_regularizer = regularizer_l2(l = regrate)
  ) %>%
  layer_activation_leaky_relu() %>%
  layer_dense(units = 2, activation = "softmax")

rnn %>% compile(
  optimizer = optimizer_adam(learning_rate = lr),
  loss = loss_categorical_crossentropy(),
  metrics = c("accuracy", metric_auc(name = "auc"))
)

# -------------------------
# Train on train + validation for exactly best_epochs epochs
# -------------------------
rnn_history <- rnn %>% fit(
  x_trainval,
  dummy_y_trainval,
  batch_size = batchsize,
  epochs = best_epochs,
  class_weight = class_weight_list,
  verbose = 2
)

# -------------------------
# Evaluate on test set using selected threshold
# -------------------------
test_pred_prob  <- predict(rnn, x_test)
test_pred_class <- ifelse(test_pred_prob[, 2] >= 0.5, 1, 0)
test_true_class <- max.col(dummy_y_test) - 1

tp <- sum(test_true_class == 1 & test_pred_class == 1)
tn <- sum(test_true_class == 0 & test_pred_class == 0)
fp <- sum(test_true_class == 0 & test_pred_class == 1)
fn <- sum(test_true_class == 1 & test_pred_class == 0)

test_sensitivity <- ifelse((tp + fn) > 0, tp / (tp + fn), NA)
test_specificity <- ifelse((tn + fp) > 0, tn / (tn + fp), NA)
test_bal_acc     <- mean(c(test_sensitivity, test_specificity), na.rm = TRUE)

test_accuracy <- (tp + tn) / (tp + tn + fp + fn)

auc_metric <- metric_auc()
auc_metric$update_state(dummy_y_test, test_pred_prob)
test_auc <- as.numeric(auc_metric$result())

cat("Test sensitivity:", round(test_sensitivity, 4), "\n")
cat("Test specificity:", round(test_specificity, 4), "\n")
cat("Test balanced accuracy:", round(test_bal_acc, 4), "\n")
cat("Test accuracy:", round(test_accuracy, 4), "\n")
cat("Test AUC:", round(test_auc, 4), "\n")

# -------------------------
# Save model and results
# -------------------------
save_model(rnn, "Models/rnn/final_rnn_model_no_threshold_tuning_501.keras")

saveRDS(
  list(
    model_id         = model_id,
    best_epoch       = best_epochs,
    lstmunits        = lstmunits,
    neuron1          = neuron1,
    batchsize        = batchsize,
    lr               = lr,
    dropout1         = dropout1,
    regrate          = regrate,
    use_batchnorm    = use_batchnorm,
    test_sensitivity = test_sensitivity,
    test_specificity = test_specificity,
    test_bal_acc     = test_bal_acc,
    test_accuracy    = test_accuracy,
    test_auc         = test_auc,
    confusion_matrix = matrix(
      c(tn, fp, fn, tp),
      nrow = 2,
      byrow = TRUE,
      dimnames = list(
        "Actual" = c("Alewife", "Rainbow Smelt"),
        "Predicted" = c("Alewife", "Rainbow Smelt")
      )
    )
  ),
  "Models/rnn/final_rnn_test_results_no_threshold_tuning_501.rds"
)

print("Final model training and evaluation complete!")

