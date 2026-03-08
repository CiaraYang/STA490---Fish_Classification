library(class)
library(ggplot2)

# load processed data
load("Data/knn_data.RData")

# range of k values
k_values <- 1:100

val_acc <- numeric(length(k_values))

# tune k using validation set
for(i in seq_along(k_values)){
  
  k <- k_values[i]
  
  pred <- knn(
    train = x_train,
    test = x_validate,
    cl = y_train,
    k = k
  )
  
  val_acc[i] <- mean(pred == y_validate)
}

tuning_results <- data.frame(
  k = k_values,
  Accuracy = val_acc
)

# plot tuning curve
ggplot(tuning_results, aes(x = k, y = Accuracy)) +
  geom_line(size = 1) +
  geom_point(size = 3) +
  labs(
    title = "Validation Accuracy vs k (KNN)",
    x = "Number of Neighbors (k)",
    y = "Validation Accuracy"
  ) +
  theme_bw()

# best k
best_k <- tuning_results$k[which.max(tuning_results$Accuracy)]

best_k



# train prediction
pred_train <- knn(
  train = x_train,
  test = x_train,
  cl = y_train,
  k = best_k
)

train_acc <- mean(pred_train == y_train)

# validation prediction
pred_val <- knn(
  train = x_train,
  test = x_validate,
  cl = y_train,
  k = best_k
)

val_acc <- mean(pred_val == y_validate)

# test prediction
pred_test <- knn(
  train = x_train,
  test = x_test,
  cl = y_train,
  k = best_k
)

test_acc <- mean(pred_test == y_test)

train_acc
val_acc
test_acc