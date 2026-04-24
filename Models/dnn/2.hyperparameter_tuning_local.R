library(keras3)
library(tensorflow)

set_random_seed(15)

# load processed data from Data/
x_train <- readRDS("Data/x_train.rds")
x_validate <- readRDS("Data/x_validate.rds")
x_test <- readRDS("Data/x_test.rds")

y_train <- readRDS("Data/y_train.rds")
y_validate <- readRDS("Data/y_validate.rds")
y_test <- readRDS("Data/y_test.rds")

# convert CNN arrays to DNN matrices
x_train <- matrix(x_train[, , 1], nrow = dim(x_train)[1], ncol = dim(x_train)[2])
x_validate <- matrix(x_validate[, , 1], nrow = dim(x_validate)[1], ncol = dim(x_validate)[2])
x_test <- matrix(x_test[, , 1], nrow = dim(x_test)[1], ncol = dim(x_test)[2])

# make sure y is numeric 0/1
y_train <- as.numeric(y_train)
y_validate <- as.numeric(y_validate)
y_test <- as.numeric(y_test)

# input dimension
input_dim <- ncol(x_train)

# class counts
class_counts <- table(y_train)
print(class_counts)

# focal loss parameters
alpha_pos <- as.numeric(class_counts["0"] / sum(class_counts))
gamma_val <- 2.0

binary_focal_loss <- function(alpha = 0.75, gamma = 2.0) {
  function(y_true, y_pred) {
    y_true <- tf$cast(y_true, tf$float32)
    y_pred <- tf$cast(y_pred, tf$float32)
    
    eps <- tf$constant(1e-7, dtype = tf$float32)
    y_pred <- tf$clip_by_value(y_pred, eps, 1 - eps)
    
    alpha_t <- y_true * alpha + (1 - y_true) * (1 - alpha)
    p_t <- y_true * y_pred + (1 - y_true) * (1 - y_pred)
    
    loss <- -alpha_t * tf$pow(1 - p_t, gamma) * tf$math$log(p_t)
    tf$reduce_mean(loss)
  }
}

# build DNN model
build_dnn_model <- function(input_dim,
                            hidden1 = 128,
                            hidden2 = 64,
                            dropout = 0.3,
                            l2_lambda = 0.001,
                            lr = 0.001,
                            alpha = 0.75,
                            gamma = 2.0) {
  
  model <- keras_model_sequential() |>
    layer_dense(
      units = hidden1,
      input_shape = c(input_dim),
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
      loss = binary_focal_loss(alpha = alpha, gamma = gamma),
      metrics = c("accuracy", metric_auc(name = "auc"))
    )
  
  model
}

# hyperparameter grid
grid_full <- expand.grid(
  hidden1 = c(32, 64, 128),
  hidden2 = c(16, 32, 64),
  dropout = c(0.3, 0.4, 0.5),
  l2 = c(0.001, 0.005),
  lr = c(0.001, 0.0005),
  batch_size = c(32, 64),
  stringsAsFactors = FALSE
)

set.seed(15)
n_try <- 20
grid_sub <- grid_full[sample(nrow(grid_full), n_try, replace = FALSE), ]

# callbacks
make_callbacks <- function() {
  list(
    callback_early_stopping(
      monitor = "val_loss",
      mode = "min",
      min_delta = 0.001,
      patience = 30,
      restore_best_weights = TRUE
    ),
    callback_reduce_lr_on_plateau(
      monitor = "val_loss",
      mode = "min",
      factor = 0.5,
      patience = 5,
      min_lr = 1e-5
    )
  )
}

# store tuning results
tuning_results <- data.frame()

for (i in seq_len(nrow(grid_sub))) {
  
  cat("run", i, "of", nrow(grid_sub), "\n")
  
  model <- build_dnn_model(
    input_dim = input_dim,
    hidden1 = grid_sub$hidden1[i],
    hidden2 = grid_sub$hidden2[i],
    dropout = grid_sub$dropout[i],
    l2_lambda = grid_sub$l2[i],
    lr = grid_sub$lr[i],
    alpha = alpha_pos,
    gamma = gamma_val
  )
  
  history <- model |>
    fit(
      x = x_train,
      y = y_train,
      validation_data = list(x_validate, y_validate),
      epochs = 100,
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
      alpha_pos = alpha_pos,
      gamma = gamma_val,
      best_epoch = best_epoch,
      best_val_loss = val_loss_vec[best_epoch],
      best_val_acc = val_acc_vec[best_epoch],
      best_val_auc = val_auc_vec[best_epoch]
    )
  )
}

tuning_results <- tuning_results[
  order(tuning_results$best_val_loss,
        -tuning_results$best_val_auc),
]

row.names(tuning_results) <- NULL

head(tuning_results, 10)
