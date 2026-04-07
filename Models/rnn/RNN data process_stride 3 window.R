# =========================================================
# Build model data for RNN using:
# - complete ping-level acoustic signal
# - 70 kHz transducer only
# - track-level split
# - tracks with at least 5 complete pings only
# - RNN samples formed by 5 consecutive pings
# - stride = 3
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

# -------------------------
# Create track id
# -------------------------
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
# -------------------------
data_keep <- full_data %>%
  filter(track_id %in% tracks_keep$track_id) %>%
  filter(kHz == "070") %>%
  filter(!is.na(value)) %>%
  left_join(
    tracks_keep %>% select(track_id, species),
    by = "track_id"
  ) %>%
  mutate(variable = paste0("F", variable)) %>%
  select(
    Ping_index, track_id, Region_name, Year, Month,
    kHz, Location, species, variable, value
  ) %>%
  pivot_wider(
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
# Also compute how many stride-3 windows each track can produce
# -------------------------
window_size <- 5
stride <- 3

track_df <- model_df_complete %>%
  group_by(track_id, species) %>%
  summarise(n_complete_pings = n(), .groups = "drop") %>%
  mutate(
    n_sequences = ifelse(
      n_complete_pings >= window_size,
      floor((n_complete_pings - window_size) / stride) + 1,
      0
    )
  ) %>%
  filter(n_complete_pings >= 5, n_sequences >= 1)

model_df_complete <- model_df_complete %>%
  filter(track_id %in% track_df$track_id)

# -------------------------
# Track summary table
# 70kHz only:
# original -> after 1m threshold -> after removing <5 complete pings
# -------------------------
track_summary_table <- tracks_labeled %>%
  filter(kHz == "070", species %in% c("Alewife", "Rainbow Smelt")) %>%
  group_by(species) %>%
  summarise(
    n_70k_original = n_distinct(track_id),
    n_after_1m = n_distinct(track_id[track_id %in% tracks_keep$track_id]),
    n_after_5complete = n_distinct(track_id[track_id %in% track_df$track_id]),
    retain_after_1m = round(n_after_1m / n_70k_original, 3),
    retain_after_5complete = round(n_after_5complete / n_70k_original, 3),
    .groups = "drop"
  )

cat("\nTrack summary table:\n")
print(track_summary_table)

# -------------------------
# Number of possible sequences per species
# -------------------------
sequence_capacity_summary <- track_df %>%
  group_by(species) %>%
  summarise(
    n_tracks = n(),
    total_complete_pings = sum(n_complete_pings),
    total_sequences = sum(n_sequences),
    avg_sequences_per_track = round(mean(n_sequences), 2),
    .groups = "drop"
  )

cat("\nSequence capacity summary:\n")
print(sequence_capacity_summary)

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
# Number of unique fish tracks in each split
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
# Standardize signal using training pings only
# -------------------------
signal_train_ping <- as.matrix(train_df_ping[, freq_cols])
signal_validate_ping <- as.matrix(validate_df_ping[, freq_cols])
signal_test_ping <- as.matrix(test_df_ping[, freq_cols])

signal_means <- apply(signal_train_ping, 2, mean, na.rm = TRUE)
signal_sds <- apply(signal_train_ping, 2, sd, na.rm = TRUE)
signal_sds[signal_sds == 0] <- 1

train_df_ping[, freq_cols] <- scale(
  train_df_ping[, freq_cols],
  center = signal_means,
  scale = signal_sds
)

validate_df_ping[, freq_cols] <- scale(
  validate_df_ping[, freq_cols],
  center = signal_means,
  scale = signal_sds
)

test_df_ping[, freq_cols] <- scale(
  test_df_ping[, freq_cols],
  center = signal_means,
  scale = signal_sds
)

# -------------------------
# Function: build 5-ping sequences with stride = 3
# Example:
# 1:5, 4:8, 7:11, ...
# Keep only full windows
# -------------------------
build_rnn_windows_stride <- function(df, freq_cols, window_size = 5, stride = 3) {
  
  x_list <- list()
  seq_info_list <- list()
  seq_counter <- 1
  
  track_ids <- unique(df$track_id)
  
  for (tid in track_ids) {
    one_track <- df %>%
      filter(track_id == tid) %>%
      arrange(ping_order)
    
    n_track_pings <- nrow(one_track)
    
    if (n_track_pings < window_size) next
    
    start_positions <- seq(
      from = 1,
      to = n_track_pings - window_size + 1,
      by = stride
    )
    
    window_counter_within_track <- 1
    
    for (start_pos in start_positions) {
      end_pos <- start_pos + window_size - 1
      
      one_window <- one_track[start_pos:end_pos, ]
      
      if (nrow(one_window) != window_size) next
      
      seq_id <- paste0(tid, "__", window_counter_within_track)
      
      x_list[[length(x_list) + 1]] <- as.matrix(one_window[, freq_cols])
      
      seq_info_list[[length(seq_info_list) + 1]] <- data.frame(
        seq_id = seq_id,
        track_id = tid,
        species = as.character(one_window$species[1]),
        Year = as.character(one_window$Year[1]),
        Month = as.character(one_window$Month[1]),
        Location = as.character(one_window$Location[1]),
        kHz = as.character(one_window$kHz[1]),
        window_id_within_track = window_counter_within_track,
        start_ping_order = one_window$ping_order[1],
        end_ping_order = one_window$ping_order[window_size],
        n_pings = nrow(one_window),
        stringsAsFactors = FALSE
      )
      
      seq_counter <- seq_counter + 1
      window_counter_within_track <- window_counter_within_track + 1
    }
  }
  
  n_seq <- length(x_list)
  n_freq <- length(freq_cols)
  
  if (n_seq == 0) {
    x_array <- array(NA_real_, dim = c(0, window_size, n_freq))
    seq_info <- data.frame()
    y <- factor(character(0), levels = c("Alewife", "Rainbow Smelt"))
    y_dummy <- array(NA_real_, dim = c(0, 2))
    
    return(list(
      x = x_array,
      y = y,
      y_dummy = y_dummy,
      seq_info = seq_info
    ))
  }
  
  x_array <- array(NA_real_, dim = c(n_seq, window_size, n_freq))
  
  for (i in seq_len(n_seq)) {
    x_array[i, , ] <- x_list[[i]]
  }
  
  seq_info <- bind_rows(seq_info_list)
  
  y <- factor(seq_info$species, levels = c("Alewife", "Rainbow Smelt"))
  y_dummy <- to_categorical(as.integer(y) - 1, 2)
  
  list(
    x = x_array,
    y = y,
    y_dummy = y_dummy,
    seq_info = seq_info
  )
}

# -------------------------
# Build RNN sequences
# Output shape: n_samples x 5 x n_frequencies
# -------------------------
train_seq <- build_rnn_windows_stride(
  train_df_ping,
  freq_cols,
  window_size = window_size,
  stride = stride
)

validate_seq <- build_rnn_windows_stride(
  validate_df_ping,
  freq_cols,
  window_size = window_size,
  stride = stride
)

test_seq <- build_rnn_windows_stride(
  test_df_ping,
  freq_cols,
  window_size = window_size,
  stride = stride
)

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
# Sequence count by split and species
# -------------------------
split_sequence_summary <- bind_rows(
  train_seq_info %>% mutate(split = "train"),
  validate_seq_info %>% mutate(split = "validation"),
  test_seq_info %>% mutate(split = "test")
) %>%
  group_by(split, species) %>%
  summarise(
    n_sequences = n(),
    .groups = "drop"
  ) %>%
  arrange(split, species)

cat("\nNumber of sequences in each split:\n")
print(split_sequence_summary)

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
saveRDS(x_train, "Data/RNN_data/x_train_stride3.rds")
saveRDS(x_validate, "Data/RNN_data/x_validate_stride3.rds")
saveRDS(x_test, "Data/RNN_data/x_test_stride3.rds")

saveRDS(y_train, "Data/RNN_data/y_train_stride3.rds")
saveRDS(y_validate, "Data/RNN_data/y_validate_stride3.rds")
saveRDS(y_test, "Data/RNN_data/y_test_stride3.rds")

saveRDS(dummy_y_train, "Data/RNN_data/dummy_y_train_stride3.rds")
saveRDS(dummy_y_val, "Data/RNN_data/dummy_y_val_stride3.rds")
saveRDS(dummy_y_test, "Data/RNN_data/dummy_y_test_stride3.rds")

saveRDS(train_seq_info, "Data/RNN_data/train_seq_info_stride3.rds")
saveRDS(validate_seq_info, "Data/RNN_data/validate_seq_info_stride3.rds")
saveRDS(test_seq_info, "Data/RNN_data/test_seq_info_stride3.rds")

saveRDS(track_df, "Data/RNN_data/track_df_stride3.rds")

cat("\nSaved all stride=3 RNN RDS files to Data/RNN_data/\n")