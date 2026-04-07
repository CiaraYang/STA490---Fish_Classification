# =========================================================
# 02_rnn_hyperparameter_tuning.R
# RNN hyperparameter tuning for current dataset
# =========================================================

library(dplyr)
library(tensorflow)
library(keras3)

set.seed(15)

# -------------------------
# Check input shape
# -------------------------
print(dim(x_train))
print(dim(x_validate))

input_shape_use <- c(dim(x_train)[2], dim(x_train)[3]) # should be 5 and 91

# -------------------------
# Class weights
# -------------------------
class_counts <- table(y_train)
print(class_counts)

cw <- as.numeric(class_counts["Alewife"] / class_counts["Rainbow Smelt"])
class_weight_list <- list("0" = 1, "1" = cw)

print(class_weight_list)

# -------------------------
# Early stopping
# -------------------------
callbacks <- list(
  callback_early_stopping(
    monitor = "val_loss",
    min_delta = 1e-2,
    patience = 25,
    restore_best_weights = TRUE
  )
)

# -------------------------
# Hyperparameter grid
# -------------------------
regrate <- c(1e-6, 1e-5, 1e-4)
lstmunits <- c(256, 128, 64)
neuron1 <- c(256, 128, 64, 32, 16)
batchsize <- c(100, 500, 1000, 1500)

grid.search.full <- expand.grid(
  regrate = regrate,
  lstmunits = lstmunits,
  neuron1 = neuron1,
  batchsize = batchsize
)

set.seed(15)
x <- sample(1:nrow(grid.search.full), 20, replace = FALSE)
grid.search.subset <- grid.search.full[x, ]

val_loss <- vector(length = nrow(grid.search.subset))
best_epoch <- vector(length = nrow(grid.search.subset))
val_auc <- vector(length = nrow(grid.search.subset))

# choose which models to run this time
# Do this for 1:10 then 11:20 because it takes too long on its own
model_indices <- 11:20 # Change to 11:20 later

for (i in model_indices) {
  print(sprintf("Processing Model #%d", i))
  set_random_seed(15)
  
  rnn <- keras_model_sequential() %>%
    layer_lstm(
      input_shape = input_shape_use,
      units = grid.search.subset$lstmunits[i]
    ) %>%
    layer_activation_leaky_relu() %>%
    layer_batch_normalization() %>%
    layer_dense(
      units = grid.search.subset$neuron1[i],
      activity_regularizer = regularizer_l2(l = grid.search.subset$regrate[i])
    ) %>%
    layer_activation_leaky_relu() %>%
    layer_dense(units = 2, activation = "softmax")
  
  rnn %>% compile(
    optimizer = optimizer_adam(learning_rate = 1e-4),
    loss = loss_categorical_crossentropy(),
    metrics = c("accuracy", metric_auc(name = "auc"))
  )
  
  rnn_history <- rnn %>% fit(
    x_train, dummy_y_train,
    batch_size = grid.search.subset$batchsize[i],
    epochs = 200,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = class_weight_list,
    callbacks = callbacks,
    verbose = 1
  )
  
  val_loss[i] <- min(rnn_history$metrics$val_loss)
  best_epoch[i] <- which.min(rnn_history$metrics$val_loss)
  val_auc[i] <- max(rnn_history$metrics$val_auc)
}

results <- grid.search.subset %>%
  mutate(
    val_loss = val_loss,
    best_epoch = best_epoch,
    val_auc = val_auc
  )

print(results)

best_model_index <- which.min(results$val_loss)
print(best_model_index)
print(results[best_model_index, ])
