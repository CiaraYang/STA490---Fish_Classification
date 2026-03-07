# load packages
library(keras3)
library(tensorflow)

# set seed
set_random_seed(15)

# input dimension
input_dim <- ncol(x_train_dnn)

# build DNN model
build_dnn_model <- function(input_dim,
                            hidden1 = 128,
                            hidden2 = 64,
                            dropout = 0.3,
                            l2_lambda = 0.001,
                            lr = 0.001) {
  
  model <- keras_model_sequential() |>
    layer_dense(
      units = hidden1,
      input_shape = input_dim,
      kernel_regularizer = regularizer_l2(l2_lambda)
    ) |>
    layer_batch_normalization() |>
    layer_activation("relu") |>
    layer_dropout(rate = dropout) |>
    layer_dense(
      units = hidden2,
      kernel_regularizer = regularizer_l2(l2_lambda)
    ) |>
    layer_batch_normalization() |>
    layer_activation("relu") |>
    layer_dropout(rate = dropout) |>
    layer_dense(units = 2, activation = "softmax")
  
  model |>
    compile(
      optimizer = optimizer_adam(learning_rate = lr),
      loss = "categorical_crossentropy",
      metrics = c("accuracy", metric_auc(name = "auc"))
    )
}

# define hyperparameter grid
grid_full <- expand.grid(
  hidden1 = c(64, 128, 256),
  hidden2 = c(32, 64, 128),
  dropout = c(0.2, 0.3, 0.4),
  l2 = c(0.0001, 0.001),
  lr = c(0.001, 0.0005),
  batch_size = c(32, 64),
  stringsAsFactors = FALSE
)

# total number of combinations
nrow(grid_full)

# randomly sample tuning combinations
set.seed(15)
n_try <- 20
grid_sub <- grid_full[sample(nrow(grid_full), n_try, replace = FALSE), ]

# define callbacks
make_callbacks <- function() {
  list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 8,
      restore_best_weights = TRUE
    ),
    callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      factor = 0.5,
      patience = 4,
      min_lr = 1e-5
    )
  )
}

# store tuning results
tuning_results <- data.frame()

# run hyperparameter tuning
for (i in seq_len(nrow(grid_sub))) {
  
  cat("run", i, "of", nrow(grid_sub), "\n")
  
  model <- build_dnn_model(
    input_dim = input_dim,
    hidden1 = grid_sub$hidden1[i],
    hidden2 = grid_sub$hidden2[i],
    dropout = grid_sub$dropout[i],
    l2_lambda = grid_sub$l2[i],
    lr = grid_sub$lr[i]
  )
  
  history <- model |>
    fit(
      x = x_train_dnn,
      y = dummy_y_train,
      validation_data = list(x_validate_dnn, dummy_y_val),
      epochs = 60,
      batch_size = grid_sub$batch_size[i],
      callbacks = make_callbacks(),
      verbose = 0
    )
  
  val_loss_vec <- history$metrics$val_loss
  val_acc_vec <- history$metrics$val_accuracy
  val_auc_vec <- history$metrics$val_auc
  
  if (length(val_loss_vec) == 0) next
  
  best_epoch <- which.min(val_loss_vec)
  
  tuning_results <- rbind(
    tuning_results,
    data.frame(
      run_id = i,
      hidden1 = grid_sub$hidden1[i],
      hidden2 = grid_sub$hidden2[i],
      dropout = grid_sub$dropout[i],
      l2 = grid_sub$l2[i],
      lr = grid_sub$lr[i],
      batch_size = grid_sub$batch_size[i],
      best_epoch = best_epoch,
      best_val_loss = val_loss_vec[best_epoch],
      best_val_acc = val_acc_vec[best_epoch],
      best_val_auc = val_auc_vec[best_epoch]
    )
  )
}

# rank models
tuning_results <- tuning_results[
  order(-tuning_results$best_val_auc,
        -tuning_results$best_val_acc,
        tuning_results$best_val_loss),
]

# show top results
head(tuning_results, 10)