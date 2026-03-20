# =========================================================
# 02_rnn_hyperparameter_tuning.R
# RNN hyperparameter tuning for current dataset
# =========================================================

library(dplyr)
library(tidymodels)
library(tensorflow)
library(keras3)

set.seed(15)

# -------------------------
# Check input shape
# -------------------------
print(dim(x_train))
print(dim(x_validate))

input_shape_use <- c(dim(x_train)[2], dim(x_train)[3])   # should be c(92, 1)

# -------------------------
# Class weights for imbalanced data
# -------------------------
# y_train should already be a factor with levels:
# c("Alewife", "Rainbow Smelt")

class_counts <- table(y_train)
print(class_counts)

# In your encoding:
# Alewife -> 0
# Rainbow Smelt -> 1
# Give larger weight to minority class
cw <- as.numeric(class_counts[1] / class_counts[2])

# If Rainbow Smelt is minority, cw > 1
# If not, you may want to reverse it depending on counts
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

# For newer Macs
optimizers <- keras::keras$optimizers

# -------------------------
# Hyperparameter grid
# -------------------------
regrate   <- c(1e-6, 1e-5, 1e-4)
lstmunits <- c(256, 128, 64)
neuron1   <- c(256, 128, 64, 32, 16)
batchsize <- c(100, 500, 1000, 1500)

grid.search.full <- expand.grid(
  regrate = regrate,
  lstmunits = lstmunits,
  neuron1 = neuron1,
  batchsize = batchsize
)

# Randomly select 20 combinations
set.seed(15)
idx <- sample(1:nrow(grid.search.full), 20, replace = FALSE)
grid.search.subset <- grid.search.full[idx, ]

print(grid.search.subset)

# -------------------------
# Store results
# -------------------------
val_loss   <- rep(NA_real_, nrow(grid.search.subset))
best_epoch <- rep(NA_integer_, nrow(grid.search.subset))
val_auc    <- rep(NA_real_, nrow(grid.search.subset))

# Optional: save fitted histories if you want
# histories <- vector("list", nrow(grid.search.subset))

# -------------------------
# Hyperparameter tuning loop
# -------------------------
for(i in 1:nrow(grid.search.subset)) {
  
  cat("\n")
  print(sprintf("Processing Model #%d", i))
  print(grid.search.subset[i, ])
  
  set_random_seed(15)
  
  rnn <- keras_model_sequential()
  
  rnn %>%
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
    metrics = c("accuracy", tf$keras$metrics$AUC(name = "auc"))
  )
  
  rnn_history <- rnn %>% fit(
    x = x_train,
    y = dummy_y_train,
    batch_size = grid.search.subset$batchsize[i],
    epochs = 200,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = class_weight_list,
    callbacks = callbacks,
    verbose = 1
  )
  
  val_loss[i]   <- min(rnn_history$metrics$val_loss)
  best_epoch[i] <- which.min(rnn_history$metrics$val_loss)
  val_auc[i]    <- max(rnn_history$metrics$val_auc)
  
  # histories[[i]] <- rnn_history
}
# -------------------------
# Summarize results
# -------------------------
tuning_results <- grid.search.subset %>%
  mutate(
    val_loss = val_loss,
    best_epoch = best_epoch,
    val_auc = val_auc
  ) %>%
  arrange(val_loss)

print(tuning_results)

best_idx <- which.min(val_loss)

cat("\n============================\n")
cat("Best model index:", best_idx, "\n")
cat("Best validation loss:", val_loss[best_idx], "\n")
cat("Best epoch:", best_epoch[best_idx], "\n")
cat("Best validation AUC:", val_auc[best_idx], "\n")
print(grid.search.subset[best_idx, ])
cat("============================\n")

# -------------------------
# Save tuning results
# -------------------------
save(
  grid.search.full,
  grid.search.subset,
  tuning_results,
  val_loss,
  best_epoch,
  val_auc,
  input_shape_use,
  class_weight_list,
  file = "Data/rnn_tuning_results.RData"
)

