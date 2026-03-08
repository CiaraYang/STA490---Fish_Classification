library(class)

# load processed data
load("Data/knn_data.RData")

# choose k (example)
k <- 5

# train and predict
pred_test <- knn(
  train = x_train,
  test = x_test,
  cl = y_train,
  k = k
)

# confusion matrix
table(Predicted = pred_test, True = y_test)

# accuracy
accuracy <- mean(pred_test == y_test)

accuracy