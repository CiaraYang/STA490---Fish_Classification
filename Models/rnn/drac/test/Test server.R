library(dplyr)
library(tensorflow)
library(caret)
library(keras3)
library(pROC)
library(tibble)

# output folder
test_dir <- "Models/rnn/drac/test"
if (!dir.exists(test_dir)) dir.create(test_dir, recursive = TRUE)

# load data
x_train <- readRDS("Data/RNN_data/x_train.rds")
x_validate <- readRDS("Data/RNN_data/x_validate.rds")
x_test <- readRDS("Data/RNN_data/x_test.rds")

y_train <- readRDS("Data/RNN_data/y_train.rds")
dummy_y_train <- readRDS("Data/RNN_data/dummy_y_train.rds")
dummy_y_val <- readRDS("Data/RNN_data/dummy_y_val.rds")
dummy_y_test <- readRDS("Data/RNN_data/dummy_y_test.rds")

# load top 20
top20_models <- readRDS("Models/rnn/drac/training/training_metrics_full/top20_models.rds")
top20_configs <- readRDS("Models/rnn/drac/training/training_metrics_full/top20_configs.rds")

# class weights
class_counts <- table(y_train)
cw <- as.numeric(class_counts["Alewife"] / class_counts["Rainbow Smelt"])
class_weight_list <- list("0" = 1, "1" = cw)

# input shape
input_shape_use <- c(dim(x_train)[2], dim(x_train)[3])

all_results <- vector("list", nrow(top20_models))

for (i in seq_len(nrow(top20_models))) {
  
  model_id_use <- top20_models$model_id[i]
  best_epoch_use <- top20_models$best_epoch_loss[i]
  params <- top20_configs[i, ]
  
  cat("\n=====================================\n")
  cat("Testing model", i, "/ model_id =", model_id_use, "\n")
  print(params)
  cat("Best epoch:", best_epoch_use, "\n")
  
  set_random_seed(15)
  
  rnn <- keras_model_sequential() %>%
    layer_lstm(
      input_shape = input_shape_use,
      units = params$lstmunits
    ) %>%
    layer_activation_leaky_relu()
  
  if ("use_batchnorm" %in% names(params) && isTRUE(params$use_batchnorm)) {
    rnn <- rnn %>% layer_batch_normalization()
  }
  
  if ("dropout1" %in% names(params) && params$dropout1 > 0) {
    rnn <- rnn %>% layer_dropout(rate = params$dropout1)
  }
  
  rnn <- rnn %>%
    layer_dense(
      units = params$neuron1,
      activity_regularizer = regularizer_l2(l = params$regrate)
    ) %>%
    layer_activation_leaky_relu() %>%
    layer_dense(units = 2, activation = "softmax")
  
  lr_use <- if ("lr" %in% names(params)) params$lr else 1e-4
  
  rnn %>% compile(
    optimizer = optimizer_adam(learning_rate = lr_use),
    loss = "categorical_crossentropy",
    metrics = list(
      metric_auc(name = "auc"),
      metric_categorical_accuracy(name = "accuracy")
    )
  )
  
  history <- rnn %>% fit(
    x = x_train,
    y = dummy_y_train,
    batch_size = params$batchsize,
    epochs = best_epoch_use,
    validation_data = list(x_validate, dummy_y_val),
    class_weight = class_weight_list,
    verbose = 0
  )
  
  eval <- rnn %>% evaluate(
    x_test, dummy_y_test,
    verbose = 0,
    return_dict = TRUE
  )
  
  pred_probs <- rnn %>% predict(x_test, verbose = 0)
  prob_smelt <- pred_probs[, 2]
  
  species_pred <- factor(
    ifelse(prob_smelt >= 0.5, "Rainbow Smelt", "Alewife"),
    levels = c("Alewife", "Rainbow Smelt")
  )
  
  true_labels <- factor(
    ifelse(dummy_y_test[, 1] == 1, "Alewife", "Rainbow Smelt"),
    levels = c("Alewife", "Rainbow Smelt")
  )
  
  cm <- confusionMatrix(
    data = species_pred,
    reference = true_labels,
    positive = "Rainbow Smelt"
  )
  
  roc_obj <- roc(
    response = true_labels,
    predictor = prob_smelt,
    levels = c("Alewife", "Rainbow Smelt"),
    direction = "<"
  )
  
  test_auc <- as.numeric(auc(roc_obj))
  
  metrics_tbl <- tibble(
    model = i,
    model_id = model_id_use,
    best_epoch = best_epoch_use,
    val_loss = history$metrics$val_loss[best_epoch_use],
    val_auc = history$metrics$val_auc[best_epoch_use],
    test_loss = eval[["loss"]],
    test_auc_keras = eval[["auc"]],
    test_accuracy_keras = eval[["accuracy"]],
    test_auc_pROC = test_auc,
    accuracy = as.numeric(cm$overall["Accuracy"]),
    sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    specificity = as.numeric(cm$byClass["Specificity"]),
    balanced_accuracy = as.numeric(cm$byClass["Balanced Accuracy"]),
    precision = as.numeric(cm$byClass["Precision"]),
    recall = as.numeric(cm$byClass["Recall"]),
    f1 = as.numeric(cm$byClass["F1"]),
    lstmunits = params$lstmunits,
    neuron1 = params$neuron1,
    batchsize = params$batchsize,
    lr = if ("lr" %in% names(params)) params$lr else NA_real_,
    dropout1 = if ("dropout1" %in% names(params)) params$dropout1 else NA_real_,
    regrate = params$regrate,
    use_batchnorm = if ("use_batchnorm" %in% names(params)) params$use_batchnorm else NA
  )
  
  print(metrics_tbl)
  all_results[[i]] <- metrics_tbl
  
  keras3::clear_session()
  gc()
}

# final ranking: best to worst
all_results_tbl <- bind_rows(all_results) %>%
  arrange(desc(test_auc_pROC), desc(balanced_accuracy))

print(all_results_tbl)

write.csv(
  all_results_tbl,
  file.path(test_dir, "top_20_rnn_results.csv"),
  row.names = FALSE
)

saveRDS(
  all_results_tbl,
  file.path(test_dir, "top_20_rnn_results.rds")
)