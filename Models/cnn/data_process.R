library(readr)
library(dplyr)
library(tidyr)
library(caret)
library(keras3)

set.seed(15)
options(stringsAsFactors = FALSE)

# =========================================================
# 01_build_model_data_cnn.R
# Build ping-level data for 1D CNN
# =========================================================

# -------------------------
# Load data
# -------------------------
full_data <- readRDS("Data/full_data_merged.rds")

# -------------------------
# Create track id
# -------------------------
full_data <- full_data %>%
  mutate(
    track_id = paste(Region_name, Year, Month, Location, kHz, sep = "_")
  )

# -------------------------
# Summarize each track
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
# Build thermocline table
# -------------------------
Year <- c("2024", "2025")
Month <- c("June", "Sept")
kHz <- c("070", "200")
Location <- c("South", "North")

csv_grid_2024 <- expand.grid(
  Year = factor(2024),
  Month = Month,
  kHz = kHz,
  Location = Location
)

csv_grid_2025 <- expand.grid(
  Year = factor(2025),
  Month = "June",
  kHz = kHz,
  Location = Location
)

csv_grid <- rbind(csv_grid_2024, csv_grid_2025)

thermo_depth_csv <- tibble(
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
# Assign species at track level
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
      mid_depth > thermo_depth ~ "Rainbow Smelt"
    )
  )

# -------------------------
# Keep only stable tracks with valid species
# -------------------------
tracks_keep <- tracks_labeled %>%
  filter(max_diff_depth < 1) %>%
  filter(species %in% c("Alewife", "Rainbow Smelt")) %>%
  filter(kHz == "070")

print(table(tracks_keep$species))

# -------------------------
# Split by track_id, not by ping
# -------------------------
track_df <- tracks_keep %>%
  select(track_id, species) %>%
  distinct()

train_track_index <- createDataPartition(track_df$species, p = 0.7, list = FALSE)
train_tracks <- track_df$track_id[train_track_index]

temp_tracks_df <- track_df[-train_track_index, ]

val_track_index <- createDataPartition(temp_tracks_df$species, p = 0.5, list = FALSE)
validate_tracks <- temp_tracks_df$track_id[val_track_index]
test_tracks <- temp_tracks_df$track_id[-val_track_index]

# -------------------------
# Build ping-level dataset
# -------------------------
data_keep <- full_data %>%
  mutate(
    track_id = paste(Region_name, Year, Month, Location, kHz, sep = "_"),
    Year = as.character(Year),
    Month = as.character(Month),
    Location = as.character(Location),
    kHz = as.character(kHz)
  ) %>%
  filter(track_id %in% tracks_keep$track_id) %>%
  filter(kHz == "070") %>%
  filter(!is.na(value)) %>%
  left_join(
    tracks_keep %>%
      select(track_id, Region_name, Year, Month, Location, kHz, species),
    by = c("track_id", "Region_name", "Year", "Month", "Location", "kHz")
  ) %>%
  mutate(variable = paste0("F", variable)) %>%
  pivot_wider(
    id_cols = c(Ping_index, track_id, Region_name, Year, Month, kHz, Location, species),
    names_from = variable,
    values_from = value
  )

# -------------------------
# Keep complete ping-level rows
# -------------------------
signal_cols <- grep("^F", names(data_keep), value = TRUE)

model_df <- data_keep %>%
  filter(species %in% c("Alewife", "Rainbow Smelt")) %>%
  filter(if_all(all_of(signal_cols), ~ !is.na(.))) %>%
  mutate(species = factor(species, levels = c("Alewife", "Rainbow Smelt")))

# -------------------------
# Assign ping-level rows to train/validate/test
# -------------------------
train_df <- model_df %>%
  filter(track_id %in% train_tracks)

validate_df <- model_df %>%
  filter(track_id %in% validate_tracks)

test_df <- model_df %>%
  filter(track_id %in% test_tracks)

# -------------------------
# Extract acoustic features
# -------------------------
signal_train <- as.matrix(train_df[, signal_cols])
signal_validate <- as.matrix(validate_df[, signal_cols])
signal_test <- as.matrix(test_df[, signal_cols])

# -------------------------
# Standardize using training statistics
# -------------------------
signal_means <- apply(signal_train, 2, mean)
signal_sds <- apply(signal_train, 2, sd)
signal_sds[signal_sds == 0] <- 1

x_train <- scale(signal_train, center = signal_means, scale = signal_sds)
x_validate <- scale(signal_validate, center = signal_means, scale = signal_sds)
x_test <- scale(signal_test, center = signal_means, scale = signal_sds)

# -------------------------
# Reshape for 1D CNN
# shape = (samples, timesteps, channels)
# -------------------------
x_train <- array(x_train, dim = c(nrow(x_train), ncol(x_train), 1))
x_validate <- array(x_validate, dim = c(nrow(x_validate), ncol(x_validate), 1))
x_test <- array(x_test, dim = c(nrow(x_test), ncol(x_test), 1))

# -------------------------
# Create labels
# -------------------------
y_train <- ifelse(train_df$species == "Rainbow Smelt", 1, 0)
y_validate <- ifelse(validate_df$species == "Rainbow Smelt", 1, 0)
y_test <- ifelse(test_df$species == "Rainbow Smelt", 1, 0)

dummy_y_train <- to_categorical(y_train, num_classes = 2)
dummy_y_val <- to_categorical(y_validate, num_classes = 2)
dummy_y_test <- to_categorical(y_test, num_classes = 2)

# -------------------------
# Save
# -------------------------
save(
  x_train, x_validate, x_test,
  dummy_y_train, dummy_y_val, dummy_y_test,
  y_train, y_validate, y_test,
  train_df, validate_df, test_df,
  signal_cols,
  file = "Data/fish_model_data_ping_70k.RData"
)
