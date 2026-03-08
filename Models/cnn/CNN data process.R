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
full_data <- readRDS("Data/full_data_merged.rds")

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
    track_id = paste(Region_name, Year, Month, Location, kHz, sep = "_"),
    variable_num = as.numeric(variable)
  ) %>%
  filter(track_id %in% tracks_keep$track_id) %>%
  filter(!is.na(variable_num), !is.na(value))

# -------------------------
# Interpolation settings
# -------------------------
grid_length <- 128

# helper: interpolate one track to fixed length
interp_track <- function(x, y, grid_length = 128) {
  ord <- order(x)
  x <- x[ord]
  y <- y[ord]
  
  # remove duplicated x values if any
  keep <- !duplicated(x)
  x <- x[keep]
  y <- y[keep]
  
  # need at least 2 points for interpolation
  if (length(x) < 2) {
    return(rep(NA_real_, grid_length))
  }
  
  xout <- seq(min(x), max(x), length.out = grid_length)
  
  approx(
    x = x,
    y = y,
    xout = xout,
    method = "linear",
    rule = 2
  )$y
}

# -------------------------
# Build fixed-length feature matrix
# -------------------------
track_feature_list <- data_keep %>%
  group_by(track_id) %>%
  arrange(variable_num, .by_group = TRUE) %>%
  summarise(
    signal = list(interp_track(variable_num, value, grid_length = grid_length)),
    .groups = "drop"
  )

# convert list-column to matrix
signal_mat <- do.call(rbind, track_feature_list$signal)
colnames(signal_mat) <- paste0("x", seq_len(grid_length))

signal_df <- bind_cols(
  track_feature_list %>% select(track_id),
  as.data.frame(signal_mat)
)

# -------------------------
# Join labels
# -------------------------
model_df <- tracks_keep %>%
  select(track_id, species, kHz, Year, Month, Location) %>%
  left_join(signal_df, by = "track_id")

# remove failed interpolation rows
model_df_complete <- model_df %>%
  filter(if_all(starts_with("x"), ~ !is.na(.)))

dim(model_df)
dim(model_df_complete)
table(model_df_complete$species)

# -------------------------
# Train / validation / test split
# -------------------------
train_index <- createDataPartition(model_df_complete$species, p = 0.7, list = FALSE)
train_df <- model_df_complete[train_index, ]
temp_df  <- model_df_complete[-train_index, ]

val_index <- createDataPartition(temp_df$species, p = 0.5, list = FALSE)
validate_df <- temp_df[val_index, ]
test_df <- temp_df[-val_index, ]

# -------------------------
# Build feature matrices
# -------------------------
feature_cols <- grep("^x", names(model_df_complete), value = TRUE)

x_train_mat <- as.matrix(train_df[, feature_cols])
x_validate_mat <- as.matrix(validate_df[, feature_cols])
x_test_mat <- as.matrix(test_df[, feature_cols])

# standardize using training set only
train_means <- apply(x_train_mat, 2, mean)
train_sds <- apply(x_train_mat, 2, sd)
train_sds[train_sds == 0] <- 1

x_train_mat <- scale(x_train_mat, center = train_means, scale = train_sds)
x_validate_mat <- scale(x_validate_mat, center = train_means, scale = train_sds)
x_test_mat <- scale(x_test_mat, center = train_means, scale = train_sds)

# CNN shape: (n, length, 1)
x_train <- array(as.numeric(x_train_mat),
                 dim = c(nrow(x_train_mat), ncol(x_train_mat), 1))
x_validate <- array(as.numeric(x_validate_mat),
                    dim = c(nrow(x_validate_mat), ncol(x_validate_mat), 1))
x_test <- array(as.numeric(x_test_mat),
                dim = c(nrow(x_test_mat), ncol(x_test_mat), 1))

# labels
train_df$species <- factor(train_df$species, levels = c("Alewife", "Rainbow Smelt"))
validate_df$species <- factor(validate_df$species, levels = c("Alewife", "Rainbow Smelt"))
test_df$species <- factor(test_df$species, levels = c("Alewife", "Rainbow Smelt"))

y_train <- train_df$species
y_validate <- validate_df$species
y_test <- test_df$species

dummy_y_train <- to_categorical(as.integer(y_train) - 1, num_classes = 2)
dummy_y_val <- to_categorical(as.integer(y_validate) - 1, num_classes = 2)
dummy_y_test <- to_categorical(as.integer(y_test) - 1, num_classes = 2)

# save
save(
  x_train, x_validate, x_test,
  dummy_y_train, dummy_y_val, dummy_y_test,
  y_train, y_validate, y_test,
  train_df, validate_df, test_df,
  feature_cols, grid_length,
  file = "Data/fish_cnn_data.RData"
)
