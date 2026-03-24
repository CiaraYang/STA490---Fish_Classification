library(keras3)
library(tensorflow)

set_random_seed(15)

input_dim <- ncol(x_train)

class_counts <- table(y_train)

class_weight <- list(
  "0" = as.numeric(sum(class_counts) / (2 * class_counts["0"])),
  "1" = as.numeric(sum(class_counts) / (2 * class_counts["1"]))
)

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
    layer_dense(units = 1, activation = "sigmoid")
  
  model |>
    compile(
      optimizer = optimizer_adam(learning_rate = lr),
      loss = "binary_crossentropy",
      metrics = c("accuracy", metric_auc(name = "auc"))
    )
}

grid_full <- expand.grid(
  hidden1 = c(32, 64, 128),
  hidden2 = c(16, 32, 64),
  dropout = c(0.3, 0.4, 0.5),
  l2 = c(0.001, 0.005),
  lr = c(0.001, 0.0005),
  batch_size = c(32, 64),
  stringsAsFactors = FALSE
)

nrow(grid_full)

set.seed(15)
n_try <- 20
grid_sub <- grid_full[sample(nrow(grid_full), n_try, replace = FALSE), ]

make_callbacks <- function() {
  list(
    callback_early_stopping(
      monitor = "val_loss",
      patience = 30,
      restore_best_weights = TRUE
    ),
    callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      factor = 0.5,
      patience = 30,
      min_lr = 1e-5
    )
  )
}

tuning_results <- data.frame()

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
      x = x_train,
      y = y_train,
      validation_data = list(x_validate, y_validate),
      epochs = 200,
      batch_size = grid_sub$batch_size[i],
      callbacks = make_callbacks(),
      class_weight = class_weight,
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

tuning_results <- tuning_results[
  order(-tuning_results$best_val_auc,
        tuning_results$best_val_loss),
]

head(tuning_results, 10)