library(readr)
library(dplyr)
library(tidyr)
library(caret)
library(keras3)

set.seed(15)
options(stringsAsFactors = FALSE)

# build ping-level data for 1D CNN

# load data
full_data <- readRDS("Data/full_data_merged.rds")

# create track id
full_data <- full_data %>%
  mutate(
    track_id = paste(Region_name, Year, Month, Location, kHz, sep = "_")
  )

# summarize each track
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

# build thermocline table
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

# assign species at track level
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

# keep only stable tracks with valid species
tracks_keep <- tracks_labeled %>%
  filter(max_diff_depth < 1) %>%
  filter(species %in% c("Alewife", "Rainbow Smelt")) %>%
  filter(kHz == "070")

print(table(tracks_keep$species))

# split by track_id, not by ping
track_df <- tracks_keep %>%
  select(track_id, species) %>%
  distinct()

train_track_index <- createDataPartition(track_df$species, p = 0.7, list = FALSE)
train_tracks <- track_df$track_id[train_track_index]

temp_tracks_df <- track_df[-train_track_index, ]

val_track_index <- createDataPartition(temp_tracks_df$species, p = 0.5, list = FALSE)
validate_tracks <- temp_tracks_df$track_id[val_track_index]
test_tracks <- temp_tracks_df$track_id[-val_track_index]

# build ping-level dataset
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

# keep complete ping-level rows
signal_cols <- grep("^F", names(data_keep), value = TRUE)

model_df <- data_keep %>%
  filter(species %in% c("Alewife", "Rainbow Smelt")) %>%
  filter(if_all(all_of(signal_cols), ~ !is.na(.))) %>%
  mutate(species = factor(species, levels = c("Alewife", "Rainbow Smelt")))

# assign ping-level rows to train/validate/test
train_df <- model_df %>%
  filter(track_id %in% train_tracks)

validate_df <- model_df %>%
  filter(track_id %in% validate_tracks)

test_df <- model_df %>%
  filter(track_id %in% test_tracks)

# extract acoustic features
signal_train <- as.matrix(train_df[, signal_cols])
signal_validate <- as.matrix(validate_df[, signal_cols])
signal_test <- as.matrix(test_df[, signal_cols])

# reshape for 1D CNN without standardization
x_train <- array(signal_train, dim = c(nrow(signal_train), ncol(signal_train), 1))
x_validate <- array(signal_validate, dim = c(nrow(signal_validate), ncol(signal_validate), 1))
x_test <- array(signal_test, dim = c(nrow(signal_test), ncol(signal_test), 1))

# create labels
y_train <- ifelse(train_df$species == "Rainbow Smelt", 1, 0)
y_validate <- ifelse(validate_df$species == "Rainbow Smelt", 1, 0)
y_test <- ifelse(test_df$species == "Rainbow Smelt", 1, 0)

dummy_y_train <- to_categorical(y_train, num_classes = 2)
dummy_y_val <- to_categorical(y_validate, num_classes = 2)
dummy_y_test <- to_categorical(y_test, num_classes = 2)

saveRDS(x_train, "Data/x_train.rds")
saveRDS(x_validate, "Data/x_validate.rds")
saveRDS(x_test, "Data/x_test.rds")

saveRDS(y_train, "Data/y_train.rds")
saveRDS(y_validate, "Data/y_validate.rds")
saveRDS(y_test, "Data/y_test.rds")

saveRDS(dummy_y_train, "Data/dummy_y_train.rds")
saveRDS(dummy_y_val, "Data/dummy_y_val.rds")
saveRDS(dummy_y_test, "Data/dummy_y_test.rds")

cat("Saved all RDS files to Data/\n")