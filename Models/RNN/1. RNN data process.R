# =========================================================
# 01_build_model_data_ping_70k.R
# Build model data using:
# - raw ping-level acoustic signal (no interpolation)
# - Month metadata
# - 70 kHz transducer only
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

# create track id
full_data <- full_data %>%
  mutate(
    Year = as.character(Year),
    Month = as.character(Month),
    Location = as.character(Location),
    kHz = as.character(kHz),
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
    n_pings = n_distinct(Ping_index),
    .groups = "drop"
  )

# -------------------------
# Thermocline table
# -------------------------
Year = c("2024", "2025")
Month = c("June", "Sept")
kHz = c("070", "200")
Location = c("South", "North")

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
  thermo_depth = c(rep(c(12.3, 20), times = 4), 12, 12, 12, 12)
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
# Track filtering
# Keep only stable tracks and target species
# Then focus only on 70 kHz as requested
# -------------------------
tracks_keep <- tracks_labeled %>%
  filter(max_diff_depth < 1) %>%
  filter(species %in% c("Alewife", "Rainbow Smelt")) %>%
  filter(kHz == "070")

cat("\nTracks kept (70 kHz only):\n")
print(table(tracks_keep$species))

# -------------------------
# Build ping-level dataframe
# Each row = one ping
# No interpolation / no grid transformation
# -------------------------
data_keep <- full_data %>%
  filter(track_id %in% tracks_keep$track_id) %>%
  filter(kHz == "070") %>%
  filter(!is.na(value)) %>%
  left_join(
    tracks_keep %>%
      dplyr::select("track_id", "species"),
    by = "track_id"
  ) %>%
  mutate(variable = paste0("F", variable)) %>%
  dplyr::select(
    "Ping_index", "track_id", "Region_name", "Year", "Month",
    "kHz", "Location", "species", "variable", "value"
  ) %>%
  tidyr::pivot_wider(
    names_from = variable,
    values_from = value
  )

# -------------------------
# Remove incomplete rows
# Keep only pings with all frequency values observed
# -------------------------
freq_cols <- grep("^F", names(data_keep), value = TRUE)

model_df_complete <- data_keep %>%
  filter(species %in% c("Alewife", "Rainbow Smelt")) %>%
  filter(if_all(all_of(freq_cols), ~ !is.na(.))) %>%
  mutate(
    species = factor(species, levels = c("Alewife", "Rainbow Smelt")),
    Month = factor(Month)
  )

cat("\nPing-level data dimensions:\n")
print(dim(model_df_complete))

cat("\nPing-level class counts:\n")
print(table(model_df_complete$species))

# -------------------------
# Optional: species-based normalization
# Marco mentioned using Mia's normalization formula.
# Replace this section with Mia's exact formula when available.
# For now, data are left unchanged.
# -------------------------

# Example placeholder:
# model_df_complete <- model_df_complete %>%
#   group_by(species) %>%
#   mutate(across(all_of(freq_cols), ~ (.-mean(., na.rm = TRUE)) / sd(., na.rm = TRUE))) %>%
#   ungroup()

# -------------------------
# Train / validation / test split
# Stratified by species at track level
# Then assign all pings from those tracks
# -------------------------
track_df <- model_df_complete %>%
  distinct(track_id, species)

train_index <- createDataPartition(track_df$species, p = 0.7, list = FALSE)

train_tracks <- track_df$track_id[train_index]
temp_track_df <- track_df[-train_index, ]

val_index <- createDataPartition(temp_track_df$species, p = 0.5, list = FALSE)

validate_tracks <- temp_track_df$track_id[val_index]
test_tracks <- temp_track_df$track_id[-val_index]

train_df <- model_df_complete %>%
  filter(track_id %in% train_tracks)

validate_df <- model_df_complete %>%
  filter(track_id %in% validate_tracks)

test_df <- model_df_complete %>%
  filter(track_id %in% test_tracks)

cat("\nTrain / validation / test sizes:\n")
print(c(
  train = nrow(train_df),
  validation = nrow(validate_df),
  test = nrow(test_df)
))

cat("\nTrain class counts:\n")
print(table(train_df$species))

cat("\nValidation class counts:\n")
print(table(validate_df$species))

cat("\nTest class counts:\n")
print(table(test_df$species))

# -------------------------
# Feature matrices
# -------------------------
signal_train <- as.matrix(train_df[, freq_cols])
signal_validate <- as.matrix(validate_df[, freq_cols])
signal_test <- as.matrix(test_df[, freq_cols])

# standardize signal using training stats
signal_means <- apply(signal_train, 2, mean, na.rm = TRUE)
signal_sds <- apply(signal_train, 2, sd, na.rm = TRUE)
signal_sds[signal_sds == 0] <- 1

signal_train <- scale(signal_train, center = signal_means, scale = signal_sds)
signal_validate <- scale(signal_validate, center = signal_means, scale = signal_sds)
signal_test <- scale(signal_test, center = signal_means, scale = signal_sds)

# -------------------------
# Month metadata matrix
# Keep this if you still want Month as extra input for DNN
# -------------------------
meta_formula <- ~ Month - 1

meta_train <- model.matrix(meta_formula, data = train_df)
meta_validate <- model.matrix(meta_formula, data = validate_df)
meta_test <- model.matrix(meta_formula, data = test_df)

# -------------------------
# Combine signal + Month for DNN
# -------------------------
x_train_dnn <- cbind(signal_train, meta_train)
x_validate_dnn <- cbind(signal_validate, meta_validate)
x_test_dnn <- cbind(signal_test, meta_test)

# -------------------------
# CNN signal arrays
# Shape: n_samples x n_frequencies x 1
# -------------------------
x_train <- array(
  as.numeric(signal_train),
  dim = c(nrow(signal_train), ncol(signal_train), 1)
)

x_validate <- array(
  as.numeric(signal_validate),
  dim = c(nrow(signal_validate), ncol(signal_validate), 1)
)

x_test <- array(
  as.numeric(signal_test),
  dim = c(nrow(signal_test), ncol(signal_test), 1)
)

# -------------------------
# Labels
# -------------------------
y_train <- train_df$species
y_validate <- validate_df$species
y_test <- test_df$species

dummy_y_train <- to_categorical(as.integer(y_train) - 1, 2)
dummy_y_val <- to_categorical(as.integer(y_validate) - 1, 2)
dummy_y_test <- to_categorical(as.integer(y_test) - 1, 2)

# -------------------------
# Save
# -------------------------
save(
  x_train, x_validate, x_test,
  x_train_dnn, x_validate_dnn, x_test_dnn,
  dummy_y_train, dummy_y_val, dummy_y_test,
  y_train, y_validate, y_test,
  train_df, validate_df, test_df,
  freq_cols,
  file = "Data/fish_model_data_ping_70k.RData"
)

cat("\nSaved to: Data/fish_model_data_ping_70k.RData\n")

