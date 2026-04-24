# =========================================================
# 01_build_model_data_rnn_70k.R
# Build model data using:
# - raw ping-level acoustic signal (no interpolation)
# - 70 kHz transducer only
# - track-level split
# - RNN samples formed by every 5 consecutive pings within a track
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
# Then focus only on 70 kHz
# -------------------------
tracks_keep <- tracks_labeled %>%
  filter(max_diff_depth < 1) %>%
  filter(species %in% c("Alewife", "Rainbow Smelt")) %>%
  filter(kHz == "070")

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
    tracks_keep %>% dplyr::select(track_id, species),
    by = "track_id"
  ) %>%
  mutate(variable = paste0("F", variable)) %>%
  dplyr::select(
    Ping_index, track_id, Region_name, Year, Month,
    kHz, Location, species, variable, value
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

# -------------------------
# Order pings within track
# Then split by track first
# -------------------------
model_df_complete <- model_df_complete %>%
  mutate(
    Ping_index_num = suppressWarnings(as.numeric(as.character(Ping_index)))
  ) %>%
  group_by(track_id) %>%
  arrange(
    if (all(is.na(Ping_index_num))) row_number() else Ping_index_num,
    .by_group = TRUE
  ) %>%
  mutate(ping_order = row_number()) %>%
  ungroup()

# -------------------------
# Keep only tracks with at least 5 complete pings
# -------------------------
track_df <- model_df_complete %>%
  group_by(track_id, species) %>%
  summarise(n_complete_pings = n(), .groups = "drop") %>%
  filter(n_complete_pings >= 5)

model_df_complete <- model_df_complete %>%
  filter(track_id %in% track_df$track_id)

# -------------------------
# Track count summary table
# 70kHz only:
# original -> after 1m threshold -> after removing <5 pings
# -------------------------
track_summary_table <- tracks_labeled %>%
  filter(kHz == "070", species %in% c("Alewife", "Rainbow Smelt")) %>%
  group_by(species) %>%
  summarise(
    n_70k_original = n_distinct(track_id),
    n_after_1m = n_distinct(track_id[track_id %in% tracks_keep$track_id]),
    n_after_5ping = n_distinct(track_id[track_id %in% track_df$track_id]),
    retain_after_1m = round(n_after_1m / n_70k_original, 3),
    retain_after_5ping = round(n_after_5ping / n_70k_original, 3),
    .groups = "drop"
  )

cat("\nTrack summary table:\n")
print(track_summary_table)

# -------------------------
# Train / validation / test split
# Stratified by species at track level
# -------------------------
train_index <- createDataPartition(track_df$species, p = 0.7, list = FALSE)

train_tracks <- track_df$track_id[train_index]
temp_track_df <- track_df[-train_index, ]

val_index <- createDataPartition(temp_track_df$species, p = 0.5, list = FALSE)

validate_tracks <- temp_track_df$track_id[val_index]
test_tracks <- temp_track_df$track_id[-val_index]

train_df_ping <- model_df_complete %>%
  filter(track_id %in% train_tracks)

validate_df_ping <- model_df_complete %>%
  filter(track_id %in% validate_tracks)

test_df_ping <- model_df_complete %>%
  filter(track_id %in% test_tracks)

# -------------------------
# Table 2a: number of unique fish tracks in each split
# -------------------------
split_track_summary <- bind_rows(
  train_df_ping %>%
    distinct(track_id, species) %>%
    mutate(split = "train"),
  validate_df_ping %>%
    distinct(track_id, species) %>%
    mutate(split = "validation"),
  test_df_ping %>%
    distinct(track_id, species) %>%
    mutate(split = "test")
) %>%
  group_by(split, species) %>%
  summarise(
    n_tracks = n_distinct(track_id),
    .groups = "drop"
  ) %>%
  arrange(split, species)

cat("\nNumber of unique fish tracks in each split:\n")
print(split_track_summary)

# -------------------------
# Function: build 5-ping sequences within each track
# Rule:
# - keep only full groups of 5
# - drop remainder
# - example:
#   6 pings  -> keep 1:5, drop 6
#   12 pings -> keep 1:5 and 6:10, drop 11:12
# -------------------------
build_rnn_windows <- function(df, freq_cols, window_size = 5) {
  
  df_seq <- df %>%
    group_by(track_id) %>%
    arrange(ping_order, .by_group = TRUE) %>%
    mutate(
      n_track_pings = n(),
      n_full_windows = floor(n_track_pings / window_size)
    ) %>%
    filter(n_full_windows > 0) %>%
    mutate(
      keep_limit = n_full_windows * window_size,
      ping_pos = row_number()
    ) %>%
    filter(ping_pos <= keep_limit) %>%
    mutate(
      window_id_within_track = ((ping_pos - 1) %/% window_size) + 1,
      seq_id = paste(track_id, window_id_within_track, sep = "__")
    ) %>%
    ungroup()
  
  seq_info <- df_seq %>%
    group_by(seq_id, track_id, species, Year, Month, Location, kHz, window_id_within_track) %>%
    summarise(n_pings = n(), .groups = "drop") %>%
    filter(n_pings == window_size)
  
  df_seq <- df_seq %>%
    filter(seq_id %in% seq_info$seq_id) %>%
    group_by(seq_id) %>%
    arrange(ping_pos, .by_group = TRUE) %>%
    mutate(step_in_window = row_number()) %>%
    ungroup()
  
  seq_ids <- unique(df_seq$seq_id)
  n_seq <- length(seq_ids)
  n_freq <- length(freq_cols)
  
  x_array <- array(NA_real_, dim = c(n_seq, window_size, n_freq))
  
  for (i in seq_along(seq_ids)) {
    this_seq <- df_seq %>%
      filter(seq_id == seq_ids[i]) %>%
      arrange(step_in_window)
    
    x_array[i, , ] <- as.matrix(this_seq[, freq_cols])
  }
  
  y <- seq_info %>%
    arrange(match(seq_id, seq_ids)) %>%
    pull(species)
  
  y_dummy <- to_categorical(as.integer(y) - 1, 2)
  
  list(
    x = x_array,
    y = y,
    y_dummy = y_dummy,
    seq_info = seq_info %>% arrange(match(seq_id, seq_ids)),
    seq_df = df_seq
  )
}

# -------------------------
# Build RNN sequences
# Output shape: n_samples x 5 x n_frequencies
# -------------------------
train_seq <- build_rnn_windows(train_df_ping, freq_cols, window_size = 5)
validate_seq <- build_rnn_windows(validate_df_ping, freq_cols, window_size = 5)
test_seq <- build_rnn_windows(test_df_ping, freq_cols, window_size = 5)

x_train <- train_seq$x
x_validate <- validate_seq$x
x_test <- test_seq$x

y_train <- train_seq$y
y_validate <- validate_seq$y
y_test <- test_seq$y

dummy_y_train <- train_seq$y_dummy
dummy_y_val <- validate_seq$y_dummy
dummy_y_test <- test_seq$y_dummy

train_seq_info <- train_seq$seq_info
validate_seq_info <- validate_seq$seq_info
test_seq_info <- test_seq$seq_info

# -------------------------
# Print summary
# -------------------------
cat("\nNumber of RNN sequences:\n")
print(c(
  train = dim(x_train)[1],
  validation = dim(x_validate)[1],
  test = dim(x_test)[1]
))

cat("\nRNN input shape (train):\n")
print(dim(x_train))   # n_samples x 5 x n_frequencies

cat("\nTrain class summary:\n")
print(cbind(
  Count = table(y_train),
  Proportion = round(prop.table(table(y_train)), 3)
))

cat("\nValidation class summary:\n")
print(cbind(
  Count = table(y_validate),
  Proportion = round(prop.table(table(y_validate)), 3)
))

cat("\nTest class summary:\n")
print(cbind(
  Count = table(y_test),
  Proportion = round(prop.table(table(y_test)), 3)
))

# -------------------------
# Save
# -------------------------
saveRDS(x_train, "Data/RNN_data/x_train.rds")
saveRDS(x_validate, "Data/RNN_data/x_validate.rds")
saveRDS(x_test, "Data/RNN_data/x_test.rds")

saveRDS(y_train, "Data/RNN_data/y_train.rds")
saveRDS(y_validate, "Data/RNN_data/y_validate.rds")
saveRDS(y_test, "Data/RNN_data/y_test.rds")

saveRDS(dummy_y_train, "Data/RNN_data/dummy_y_train.rds")
saveRDS(dummy_y_val, "Data/RNN_data/dummy_y_val.rds")
saveRDS(dummy_y_test, "Data/RNN_data/dummy_y_test.rds")

cat("Saved all RNN RDS files to Data/RNN_data/\n")

