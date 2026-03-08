library(dplyr)

# load processed dataset
load("Data/fish_model_data_month_signal.RData")

# combine all observations
all_df <- bind_rows(train_df, validate_df, test_df)

# find acoustic signal columns
signal_cols <- grep("^x", names(all_df), value = TRUE)

# split again into train / validation / test
train_df <- train_df
validate_df <- validate_df
test_df <- test_df

# extract signal matrices
signal_train <- as.matrix(train_df[, signal_cols])
signal_validate <- as.matrix(validate_df[, signal_cols])
signal_test <- as.matrix(test_df[, signal_cols])

# compute scaling statistics using training data
signal_means <- apply(signal_train, 2, mean)
signal_sds <- apply(signal_train, 2, sd)
signal_sds[signal_sds == 0] <- 1

# standardize signals
signal_train <- scale(signal_train, center = signal_means, scale = signal_sds)
signal_validate <- scale(signal_validate, center = signal_means, scale = signal_sds)
signal_test <- scale(signal_test, center = signal_means, scale = signal_sds)

# month dummy variables
meta_formula <- ~ Month - 1

meta_train <- model.matrix(meta_formula, data = train_df)
meta_validate <- model.matrix(meta_formula, data = validate_df)
meta_test <- model.matrix(meta_formula, data = test_df)

# combine acoustic signal and month
x_train <- cbind(signal_train, meta_train)
x_validate <- cbind(signal_validate, meta_validate)
x_test <- cbind(signal_test, meta_test)

# labels
y_train <- train_df$species
y_validate <- validate_df$species
y_test <- test_df$species

# save processed data
save(
  x_train, x_validate, x_test,
  y_train, y_validate, y_test,
  file = "Data/knn_data.RData"
)