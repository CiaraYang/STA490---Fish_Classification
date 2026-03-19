# =========================================================
# 01_build_model_data.R
# Interpolation version
# =========================================================

library(readr)
library(dplyr)
library(tidyr)
library(caret)
library(keras3)

set.seed(15)
options(stringsAsFactors = FALSE)

# -------------------------
# Load merged data
# -------------------------
full_data <- readRDS("~/Documents/School/STA490/Fish/full_data_merged.rds")

# provisional track id
full_data <- full_data %>%
  mutate(
    track_id = paste(Region_name, Year, Month, Location, kHz, sep = "_")
  )

# -------------------------
# Track summaries
# -------------------------
tracks <- full_data %>%
  filter(!is.na(Depth), !is.na(value), !is.na(variable)) %>%
  mutate(variable_num = as.numeric(variable)) %>%
  filter(!is.na(variable_num)) %>%
  group_by(track_id, Region_name, Year, Month, Location, kHz) %>%
  summarise(
    min_depth = min(Depth, na.rm = TRUE),
    max_depth = max(Depth, na.rm = TRUE),
    max_diff_depth = max_depth - min_depth,
    n_pings = n(),
    .groups = "drop"
  )

# -------------------------
# Thermocline table
# -------------------------
Year = c("2024","2025")
Month = c("June","Sept")
kHz = c("070","200")
Location = c("South","North")

csv_grid_2024 = expand.grid(
  Year = factor(2024),
  Month = Month,
  kHz = kHz,
  Location = Location
)

csv_grid_2025 = expand.grid(
  Year = factor(2025),
  Month = "June",
  kHz = kHz,
  Location = Location
)

csv_grid = rbind(csv_grid_2024, csv_grid_2025)

thermo_depth_csv = tibble(
  csv_grid,
  thermo_depth = c(rep(c(12.3,20), times = 4), 12,12,12,12)
) %>%
  mutate(
    Year = as.character(Year),
    Month = as.character(Month),
    Location = as.character(Location),
    kHz = as.character(kHz)
  )

# -------------------------
# Species labels
# -------------------------
tracks_labeled <- tracks %>%
  mutate(
    Year = as.character(Year),
    Month = as.character(Month),
    Location = as.character(Location),
    kHz = as.character(kHz)
  ) %>%
  left_join(thermo_depth_csv, by = c("Year", "Month", "Location", "kHz")) %>%
  mutate(
    crosses_thermo = !is.na(thermo_depth) &
      (min_depth < thermo_depth & thermo_depth < max_depth),
    mid_depth = (min_depth + max_depth) / 2,
    species = case_when(
      crosses_thermo ~ "Crossing",
      is.na(thermo_depth) ~ NA_character_,
      mid_depth <= thermo_depth ~ "Alewife",
      mid_depth > thermo_depth ~ "Rainbow Smelt",
      TRUE ~ NA_character_
    )
  )

# -------------------------
# Keep only good tracks
# threshold = 1 m
# binary classification only
# -------------------------
tracks_keep <- tracks_labeled %>%
  filter(max_diff_depth < 1) %>%
  filter(species %in% c("Alewife", "Rainbow Smelt"))

table(tracks_keep$species)

# -------------------------
# Keep ping-level rows for retained tracks
# -------------------------
data_keep <- full_data %>%
  mutate(
    track_id = paste(Region_name, Year, Month, Location, kHz, sep = "_")
  ) %>%
  filter(track_id %in% tracks_keep$track_id) %>%
  filter(!is.na(value)) %>%
  filter(kHz == "070") %>%
  left_join(tracks_keep %>% select(Region_name, Year, Month, Location, kHz, species)) %>%
  mutate(variable = paste0("F", variable)) %>%
  pivot_wider(id_cols = c("Ping_index","Region_name","Year","Month","kHz","Location","species"),
              names_from = "variable", values_from = "value")

# -------------------------
# Add Fish Length
# -------------------------

# Step 1: Calculate mean TS across all frequency columns (cols 8-98), ignoring NAs
data_keep <- data_keep %>%
  mutate(TS_mean = rowMeans(select(., 8:98), na.rm = TRUE))

# Step 2: Calculate fish length from TS_mean
data_keep <- data_keep %>%
  mutate(fish_length = case_when(
    species == "Alewife"       ~ 10^((TS_mean + 64.2) / 20.5),
    species == "Rainbow Smelt" ~ 10^((TS_mean + 72) / 20),
    TRUE ~ NA_real_
  ))

# -------------------------
# Linearization and Normalization TS within each ping
# -------------------------
freq_cols <- grep("^F", names(data_keep), value = TRUE)

# Step 1: Ensure frequency columns are numeric first
data_keep[, freq_cols] <- lapply(data_keep[, freq_cols], as.numeric)

# Step 2: Convert dB to linear sound intensity (10^(dB/10))
data_linear <- data_keep
data_linear[, freq_cols] <- 10^(data_keep[, freq_cols] / 10)

# Step 3: Normalize each row (ping) to [0, 1]
data_normalized <- data_linear
data_normalized[, freq_cols] <- t(apply(data_linear[, freq_cols], 1, function(row) {
  (row - min(row, na.rm = TRUE)) / (max(row, na.rm = TRUE) - min(row, na.rm = TRUE))
}))

# Step 4: Carry fish_length and TS_mean into data_normalized
data_normalized$TS_mean <- data_keep$TS_mean
data_normalized$fish_length <- data_keep$fish_length



# -------------------------
# Define feature columns
# -------------------------
# Add fish length as a feature (Optional)
feature_cols <- c(grep("^F", names(data_normalized), value = TRUE), "fish_length")

# -------------------------
# Train / validation / test split
# -------------------------
train_index <- createDataPartition(data_normalized$species, p = 0.7, list = FALSE)
train_df <- data_normalized[train_index, ]
temp_df  <- data_normalized[-train_index, ]

val_index <- createDataPartition(temp_df$species, p = 0.5, list = FALSE)
validate_df <- temp_df[val_index, ]
test_df     <- temp_df[-val_index, ]

# -------------------------
# Build feature matrices
# -------------------------
x_train_mat    <- as.matrix(train_df[, feature_cols])
x_validate_mat <- as.matrix(validate_df[, feature_cols])
x_test_mat     <- as.matrix(test_df[, feature_cols])

# Standardize using training set only
train_means <- apply(x_train_mat, 2, mean)
train_sds   <- apply(x_train_mat, 2, sd)
train_sds[train_sds == 0] <- 1

x_train_mat    <- scale(x_train_mat,    center = train_means, scale = train_sds)
x_validate_mat <- scale(x_validate_mat, center = train_means, scale = train_sds)
x_test_mat     <- scale(x_test_mat,     center = train_means, scale = train_sds)

# Shape: (n, n_features, 1) — works for both CNN and RNN/LSTM
x_train    <- array(as.numeric(x_train_mat),
                    dim = c(nrow(x_train_mat), ncol(x_train_mat), 1))
x_validate <- array(as.numeric(x_validate_mat),
                    dim = c(nrow(x_validate_mat), ncol(x_validate_mat), 1))
x_test     <- array(as.numeric(x_test_mat),
                    dim = c(nrow(x_test_mat), ncol(x_test_mat), 1))

# -------------------------
# Labels
# -------------------------
train_df$species    <- factor(train_df$species,    levels = c("Alewife", "Rainbow Smelt"))
validate_df$species <- factor(validate_df$species, levels = c("Alewife", "Rainbow Smelt"))
test_df$species     <- factor(test_df$species,     levels = c("Alewife", "Rainbow Smelt"))

y_train    <- train_df$species
y_validate <- validate_df$species
y_test     <- test_df$species

dummy_y_train <- to_categorical(as.integer(y_train)    - 1, num_classes = 2)
dummy_y_val   <- to_categorical(as.integer(y_validate) - 1, num_classes = 2)
dummy_y_test  <- to_categorical(as.integer(y_test)     - 1, num_classes = 2)

# -------------------------
# Save — shared by both CNN and RNN pipelines
# -------------------------
save(
  x_train, x_validate, x_test,
  dummy_y_train, dummy_y_val, dummy_y_test,
  y_train, y_validate, y_test,
  train_df, validate_df, test_df,
  train_means, train_sds,
  feature_cols,
  file = "~/Documents/School/STA490/Fish/RNN/fish_rnn_data.RData"
)