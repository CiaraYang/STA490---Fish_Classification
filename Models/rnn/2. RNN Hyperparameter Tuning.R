# =========================================================
# 02_rnn_hyperparameter_tuning.R
# Total combinations = 14,400
# =========================================================

library(optparse)
library(dplyr)
library(tensorflow)
library(keras3)

# -------------------------
# Command line arguments
# -------------------------
option_list <- list(
  make_option(c("-b", "--batch"), type = "integer", default = 1,
              help = "Batch ID [default = %default]", metavar = "integer")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

nbatch <- opt$batch

set.seed(15)

# -------------------------
# Load data
# -------------------------
x_train <- readRDS("Data/RNN_data/x_train.rds")
x_validate <- readRDS("Data/RNN_data/x_validate.rds")

y_train <- readRDS("Data/RNN_data/y_train.rds")
y_validate <- readRDS("Data/RNN_data/y_validate.rds")

dummy_y_train <- readRDS("Data/RNN_data/dummy_y_train.rds")
dummy_y_val <- readRDS("Data/RNN_data/dummy_y_val.rds")

# -------------------------
# Check input shape
# -------------------------
cat("x_train dim:\n")
print(dim(x_train))

cat("x_validate dim:\n")
print(dim(x_validate))

input_shape_use <- c(dim(x_train)[2], dim(x_train)[3])

# -------------------------
# Class weights
# -------------------------
class_counts <- table(y_train)
cat("Training class counts:\n")
print(class_counts)

cw <- as.numeric(class_counts["Alewife"] / class_counts["Rainbow Smelt"])
class_weight_list <- list("0" = 1, "1" = cw)

cat("Class weight list:\n")
print(class_weight_list)

# -------------------------
# grid (14,400)
# -------------------------
lstmunits <- c(16, 32, 64, 128, 256, 384)   # 6
neuron1   <- c(16, 32, 64, 128, 256)        # 5
batchsize <- c(32, 64, 128, 256)            # 4
lr        <- c(1e-4, 2e-4, 3e-4, 5e-4, 1e-3) # 5
dropout1  <- c(0.0, 0.1, 0.2, 0.3)          # 4
regrate   <- c(1e-7, 1e-6, 1e-5)            # 3
use_batchnorm <- c(TRUE, FALSE)             # 2

grid.search.full <- expand.grid(
  lstmunits = lstmunits,
  neuron1 = neuron1,
  batchsize = batchsize,
  lr = lr,
  dropout1 = dropout1,
  regrate = regrate,
  use_batchnorm = use_batchnorm
)

cat("Total number of hyperparameter combinations:\n")
print(nrow(grid.search.full))

# -------------------------
# Batch setup
# -------------------------
models_per_batch <-2000

start_idx <- (nbatch - 1) * models_per_batch + 1
end_idx <- min(nbatch * models_per_batch, nrow(grid.search.full))

if (start_idx > nrow(grid.search.full)) {
  stop("Batch index exceeds total number of models in grid.search.full.")
}

model_indices <- start_idx:end_idx

cat("Running batch:\n")
print(nbatch)

cat("Model indices in this batch:\n")
print(c(start_idx, end_idx))

# -------------------------
# Output directory
# -------------------------
output_dir <- paste0(
  "Models/rnn/drac/training/training_metrics_full/training_metrics_b",
  nbatch
)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

# -------------------------
# Training loop
# One model -> one output file
# -------------------------
for (i in model_indices) {
  
  cat(sprintf("\nProcessing Model #%d\n", i))
  
  output_file <- paste0(output_dir, "/training_output_", i, ".rds")
  
  if (file.exists(output_file)) {
    cat("Output already exists. Skipping.\n")
    next
  }
  
  this_config <- grid.search.full[i, ]
  print(this_config)
  
  set_random_seed(15)
  
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
  # Build model
  # -------------------------
  rnn <- keras_model_sequential()
  
  rnn <- rnn %>%
    layer_lstm(
      units = this_config$lstmunits,
      input_shape = input_shape_use,
      dropout = this_config$dropout1
    ) %>%
    layer_activation_leaky_relu()
  
  if (this_config$use_batchnorm) {
    rnn <- rnn %>% layer_batch_normalization()
  }
  
  rnn <- rnn %>%
    layer_dense(
      units = this_config$neuron1,
      activity_regularizer = regularizer_l2(l = this_config$regrate)
    ) %>%
    layer_activation_leaky_relu() %>%
    layer_dense(units = 2, activation = "softmax")
  
  # -------------------------
  # Compile
  # -------------------------
  rnn %>% compile(
    optimizer = optimizer_adam(learning_rate = this_config$lr),
    loss = loss_categorical_crossentropy(),
    metrics = c("accuracy", metric_auc(name = "auc"))
  )
  
  # -------------------------
  # Fit
  # -------------------------
  rnn_history <- rnn %>% fit(
    x_train, dummy_y_train,
    batch_size = this_config$batchsize,
    epochs = 200,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = class_weight_list,
    callbacks = callbacks,
    verbose = 0
  )
  
  # -------------------------
  # Save summary metrics
  # -------------------------
  aux_row <- tibble(
    val_loss = min(rnn_history$metrics$val_loss),
    best_epoch_loss = which.min(rnn_history$metrics$val_loss),
    val_auc = max(rnn_history$metrics$val_auc),
    model_id = i
  )
  
  saveRDS(aux_row, output_file)
  
  cat("Saved:\n")
  print(output_file)
  
  rm(rnn, rnn_history, aux_row, this_config)
  gc()
}

cat(paste0("\nFinished batch ", nbatch, "\n"))

