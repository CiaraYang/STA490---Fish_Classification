# =========================================================
# 02_tune_cnn.R
# Hyperparameter tuning for fish 1D CNN
# =========================================================

## ---- Libraries ----
library(dplyr)
library(tidymodels)
library(tensorflow)
library(caret)
library(rsample)
library(keras3)

## ---- Load prepared data ----
x_train <- readRDS("Data/x_train.rds")
x_validate <- readRDS("Data/x_validate.rds")
x_test <- readRDS("Data/x_test.rds")

y_train <- readRDS("Data/y_train.rds")
y_validate <- readRDS("Data/y_validate.rds")
y_test <- readRDS("Data/y_test.rds")

dummy_y_train <- readRDS("Data/dummy_y_train.rds")
dummy_y_val <- readRDS("Data/dummy_y_val.rds")
dummy_y_test <- readRDS("Data/dummy_y_test.rds")

## ---- Check data ----
cat("x_train shape:", dim(x_train), "\n")
cat("x_validate shape:", dim(x_validate), "\n")
cat("x_test shape:", dim(x_test), "\n")
print(table(y_train))
print(table(y_validate))
print(table(y_test))

## ---- Class weight calculation ----
train_tab <- table(y_train)
class_weights_vec <- as.numeric(sum(train_tab) / (length(train_tab) * train_tab))
class_weights <- as.list(class_weights_vec)
names(class_weights) <- as.character(0:(length(class_weights) - 1))

## ---- Early stopping ----
callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
    min_delta = 1e-3,
    patience = 30,
    restore_best_weights = TRUE
  )
)

## ---- Parameter grid ----
filters1 <- c(8, 16, 32)
filters2 <- c(8, 16, 32)
filters3 <- c(8, 16, 32)
filters4 <- c(8, 16, 32)
filters5 <- c(8, 16, 32)

kernel_size <- c(3, 5, 7)
batch_size <- c(32, 64, 128)

droprate1 <- c(0, 0.1, 0.2)
droprate2 <- c(0, 0.1, 0.2)
droprate3 <- c(0, 0.1, 0.2)
droprate4 <- c(0, 0.1, 0.2)
droprate5 <- c(0, 0.1, 0.2)

grid.search.full <- expand.grid(
  filters1 = filters1,
  filters2 = filters2,
  filters3 = filters3,
  filters4 = filters4,
  filters5 = filters5,
  kernel_size = kernel_size,
  batch_size = batch_size,
  droprate1 = droprate1,
  droprate2 = droprate2,
  droprate3 = droprate3,
  droprate4 = droprate4,
  droprate5 = droprate5
)

set.seed(15)
n_subset <- 20
grid.search.subset <- grid.search.full[sample(1:nrow(grid.search.full), n_subset, replace = FALSE), ]

## ---- Storage ----
val_loss <- rep(NA, n_subset)
best_epoch_loss <- rep(NA, n_subset)
val_auc <- rep(NA, n_subset)

input_length <- dim(x_train)[2]

## ---- Model builder ----
build_cnn_model <- function(filters1, filters2, filters3, filters4, filters5,
                            kernel_size,
                            droprate1, droprate2, droprate3, droprate4, droprate5,
                            input_length) {
  
  model <- keras_model_sequential() %>%
    layer_conv_1d(
      input_shape = c(input_length, 1),
      filters = filters1,
      kernel_size = kernel_size,
      activation = "relu",
      padding = "same",
      strides = 1
    ) %>%
    layer_dropout(rate = droprate1) %>%
    layer_batch_normalization() %>%
    
    layer_conv_1d(
      filters = filters2,
      kernel_size = kernel_size,
      activation = "relu",
      padding = "same",
      strides = 1
    ) %>%
    layer_dropout(rate = droprate2) %>%
    layer_batch_normalization() %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    
    layer_conv_1d(
      filters = filters3,
      kernel_size = kernel_size,
      activation = "relu",
      padding = "same",
      strides = 1
    ) %>%
    layer_dropout(rate = droprate3) %>%
    layer_batch_normalization() %>%
    
    layer_conv_1d(
      filters = filters4,
      kernel_size = kernel_size,
      activation = "relu",
      padding = "same",
      strides = 1
    ) %>%
    layer_dropout(rate = droprate4) %>%
    layer_max_pooling_1d(pool_size = 2) %>%
    layer_batch_normalization() %>%
    
    layer_conv_1d(
      filters = filters5,
      kernel_size = kernel_size,
      activation = "relu",
      padding = "same",
      strides = 1
    ) %>%
    layer_dropout(rate = droprate5) %>%
    layer_batch_normalization() %>%
    
    layer_flatten() %>%
    layer_dense(units = 2, activation = "softmax")
  
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = 1e-4),
    loss = "categorical_crossentropy",
    metrics = list(metric_auc(name = "auc"))
  )
  
  return(model)
}

## ---- Run search ----
for (i in 1:n_subset) {
  cat("Processing Model #", i, "\n")
  
  
  cnn <- build_cnn_model(
    filters1 = grid.search.subset$filters1[i],
    filters2 = grid.search.subset$filters2[i],
    filters3 = grid.search.subset$filters3[i],
    filters4 = grid.search.subset$filters4[i],
    filters5 = grid.search.subset$filters5[i],
    kernel_size = grid.search.subset$kernel_size[i],
    droprate1 = grid.search.subset$droprate1[i],
    droprate2 = grid.search.subset$droprate2[i],
    droprate3 = grid.search.subset$droprate3[i],
    droprate4 = grid.search.subset$droprate4[i],
    droprate5 = grid.search.subset$droprate5[i],
    input_length = input_length
  )
  
  cnn_history <- cnn %>% fit(
    x_train, dummy_y_train,
    batch_size = grid.search.subset$batch_size[i],
    epochs = 200,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = class_weights,
    callbacks = callbacks,
    verbose = 0
  )
  
  val_loss[i] <- min(cnn_history$metrics$val_loss)
  best_epoch_loss[i] <- which.min(cnn_history$metrics$val_loss)
  val_auc[i] <- max(cnn_history$metrics$val_auc)
}

## ---- Collect results ----
tuning_results <- grid.search.subset %>%
  mutate(
    val_loss = val_loss,
    best_epoch = best_epoch_loss,
    val_auc = val_auc
  ) %>%
  arrange(val_loss, desc(val_auc))

print(tuning_results)

## ---- Best model ----
best_row <- which.min(tuning_results$val_loss)
best_param <- tuning_results[best_row, ]

cat("\nBest parameter row:\n")
print(best_param)

## ---- Save results ----
write.csv(tuning_results, "Data/cnn_tuning_results.csv", row.names = FALSE)
save(tuning_results, best_param, file = "Data/cnn_tuning_results.RData")

