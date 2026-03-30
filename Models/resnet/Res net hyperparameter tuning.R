# =========================================================
# 02_tune_resnet.R
# Hyperparameter tuning for 1D ResNet
# =========================================================

library(dplyr)
library(tensorflow)
# library(keras3)
library(caret)

set.seed(15)
options(stringsAsFactors = FALSE)

## ---- Load prepared data from updated data_process.R ----
x_train <- readRDS("Data/x_train.rds")
x_validate <- readRDS("Data/x_validate.rds")
x_test <- readRDS("Data/x_test.rds")

y_train <- readRDS("Data/y_train.rds")
y_validate <- readRDS("Data/y_validate.rds")
y_test <- readRDS("Data/y_test.rds")

dummy_y_train <- readRDS("Data/dummy_y_train.rds")
dummy_y_val <- readRDS("Data/dummy_y_val.rds")
dummy_y_test <- readRDS("Data/dummy_y_test.rds")

cat("x_train shape:", dim(x_train), "\n")
cat("x_validate shape:", dim(x_validate), "\n")
cat("x_test shape:", dim(x_test), "\n")
print(table(y_train))
print(table(y_validate))
print(table(y_test))

## ---- Class weights ----
# train_tab <- table(y_train)
# cw <- as.numeric(train_tab[1] / train_tab[2])
# class_weights <- list("0" = 1, "1" = cw)

# train_tab <- table(y_train)
# cw_vals <- as.numeric(sum(train_tab) / (2 * train_tab))
# class_weights <- setNames(as.list(cw_vals), c("0", "1"))

train_tab <- table(y_train)
cw_vals <- as.numeric(sum(train_tab) / (2 * train_tab))
class_weights <- setNames(as.list(cw_vals), names(train_tab))

## ---- Early stopping ----
# callbacks <- list(
#   callback_early_stopping(
#     monitor = "val_loss",
#     min_delta = 1e-3,
#     patience = 30,
#     restore_best_weights = TRUE
#   )
# )

callbacks <- list(
  callback_early_stopping(
    monitor = "val_auc",
    mode = "max",
    min_delta = 1e-3,
    patience = 12,
    restore_best_weights = TRUE
  )
)

## ---- Residual block ----
residual_block_1d <- function(x, filters, kernel_size, dropout_rate = 0) {
  shortcut <- x
  
  out <- x %>%
    layer_conv_1d(
      filters = filters,
      kernel_size = kernel_size,
      padding = "same",
      activation = "relu"
    ) %>%
    layer_batch_normalization() %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_conv_1d(
      filters = filters,
      kernel_size = kernel_size,
      padding = "same",
      activation = NULL
    ) %>%
    layer_batch_normalization()
  
  input_channels <- x$shape[[3]]
  if (!is.null(input_channels) && input_channels != filters) {
    shortcut <- shortcut %>%
      layer_conv_1d(
        filters = filters,
        kernel_size = 1,
        padding = "same",
        activation = NULL
      ) %>%
      layer_batch_normalization()
  }
  
  out <- layer_add(list(out, shortcut)) %>%
    layer_activation("relu")
  
  out
}

## ---- Model builder ----
build_resnet_model <- function(input_length,
                               filters1,
                               filters2,
                               kernel_size,
                               dropout_rate,
                               dense_units,
                               learning_rate) {
  
  inputs <- layer_input(shape = c(input_length, 1))
  
  x <- inputs %>%
    layer_conv_1d(
      filters = filters1,
      kernel_size = kernel_size,
      padding = "same",
      activation = "relu"
    ) %>%
    layer_batch_normalization()
  
  x <- residual_block_1d(x, filters = filters1, kernel_size = kernel_size, dropout_rate = dropout_rate)
  x <- residual_block_1d(x, filters = filters1, kernel_size = kernel_size, dropout_rate = dropout_rate)
  
  x <- x %>%
    layer_max_pooling_1d(pool_size = 2)
  
  x <- residual_block_1d(x, filters = filters2, kernel_size = kernel_size, dropout_rate = dropout_rate)
  x <- residual_block_1d(x, filters = filters2, kernel_size = kernel_size, dropout_rate = dropout_rate)
  
  # oftmax makes all class probabilities add to 1, and categorical 
  # crossentropy measures how well the model puts probability mass on the correct class.
  x <- x %>%
    layer_global_average_pooling_1d() %>%
    layer_dense(units = dense_units, activation = "relu") %>%
    layer_dropout(rate = dropout_rate) %>%
    layer_dense(units = 2, activation = "softmax")
  
  model <- keras_model(inputs = inputs, outputs = x)
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = learning_rate),
    loss = "categorical_crossentropy",
    metrics = list(
      metric_auc(name = "auc"),
      metric_categorical_accuracy(name = "accuracy")
    )
  )
  
  model
}


## ---- Hyperparameter grid ----
# make the model search a bit simpler and more targeted.
filters1 <- c(8, 16, 32)
filters2 <- c(16, 32, 64)
kernel_size <- c(3, 5, 9)
dropout_rate <- c(0.1, 0.2, 0.3)
dense_units <- c(16, 32, 64)
batch_size <- c(64, 128)
learning_rate <- c(1e-3, 3e-4, 1e-4)

grid.search.full <- expand.grid(
  filters1 = filters1,
  filters2 = filters2,
  kernel_size = kernel_size,
  dropout_rate = dropout_rate,
  dense_units = dense_units,
  batch_size = batch_size,
  learning_rate = learning_rate
)

set.seed(15)
n_subset <- 30
grid.search.subset <- grid.search.full[
  sample(1:nrow(grid.search.full), n_subset, replace = FALSE),
]

val_loss <- rep(NA_real_, n_subset)
best_epoch_auc <- rep(NA_integer_, n_subset)
val_auc <- rep(NA_real_, n_subset)
val_accuracy <- rep(NA_real_, n_subset)
val_loss_at_best_auc <- rep(NA_real_, n_subset)
val_accuracy_at_best_auc <- rep(NA_real_, n_subset)

input_length <- dim(x_train)[2]

## ---- Run tuning ----
for (i in 1:n_subset) {
  cat("Processing ResNet Model #", i, "\n")
  
  set_random_seed(15)
  
  resnet_model <- build_resnet_model(
    input_length = input_length,
    filters1 = grid.search.subset$filters1[i],
    filters2 = grid.search.subset$filters2[i],
    kernel_size = grid.search.subset$kernel_size[i],
    dropout_rate = grid.search.subset$dropout_rate[i],
    dense_units = grid.search.subset$dense_units[i],
    learning_rate = grid.search.subset$learning_rate[i]
  )
  
  history <- resnet_model %>% fit(
    x = x_train,
    y = dummy_y_train,
    batch_size = grid.search.subset$batch_size[i],
    epochs = 100,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = class_weights,
    callbacks = callbacks,
    verbose = 0
  )
  
  auc_hist <- history$metrics$val_auc
  loss_hist <- history$metrics$val_loss
  acc_hist <- history$metrics$val_accuracy
  
  best_epoch_auc[i] <- which.max(auc_hist)
  val_auc[i] <- auc_hist[best_epoch_auc[i]]
  val_loss_at_best_auc[i] <- loss_hist[best_epoch_auc[i]]
  val_accuracy_at_best_auc[i] <- acc_hist[best_epoch_auc[i]]
  val_loss[i] <- min(loss_hist)
}

  ## ---- Collect results ----
tuning_results_resnet <- grid.search.subset %>%
  mutate(
    best_epoch = best_epoch_auc,
    val_auc = val_auc,
    val_loss_at_best_auc = val_loss_at_best_auc,
    val_accuracy = val_accuracy_at_best_auc,
    val_loss_min = val_loss
  ) %>%
  arrange(desc(val_auc), val_loss_at_best_auc)

print(tuning_results_resnet)

best_param_resnet <- tuning_results_resnet[1, ]

cat("\nBest ResNet parameters:\n")
print(best_param_resnet)

write.csv(tuning_results_resnet, "Resnet/resnet_tuning_results.csv", row.names = FALSE)
save(tuning_results_resnet, 
     best_param_resnet,
     file = "Data/resnet_tuning_results.RData")
