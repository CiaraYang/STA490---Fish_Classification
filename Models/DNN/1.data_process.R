# libraries
library(readr)
library(dplyr)
library(tidyr)
library(caret)
library(keras3)

set.seed(15)
options(stringsAsFactors = FALSE)

# load merged data
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

# build thermocline lookup table
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

# assign species labels using thermocline depth
tracks_labeled <- tracks %>%
  mutate(
    Year = as.character(Year),
    Month = as.character(Month),
    Location = as.character(Location),
    kHz = as.character(kHz)
  ) %>%
  left_join(thermo_depth_csv, by = c("Year","Month","Location","kHz")) %>%
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

# keep tracks with vertical movement under 1 m
tracks_keep <- tracks_labeled %>%
  filter(max_diff_depth < 1) %>%
  filter(species %in% c("Alewife","Rainbow Smelt"))

table(tracks_keep$species)

# keep rows from retained tracks
data_keep <- full_data %>%
  mutate(
    track_id = paste(Region_name, Year, Month, Location, kHz, sep = "_"),
    variable_num = as.numeric(variable)
  ) %>%
  filter(track_id %in% tracks_keep$track_id) %>%
  filter(!is.na(variable_num), !is.na(value))

# set interpolation length
grid_length <- 128

# interpolate one track to a fixed grid
interp_track <- function(x, y, grid_length = 128) {
  
  ord <- order(x)
  x <- x[ord]
  y <- y[ord]
  
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

# average duplicated frequencies within each track
track_feature_list <- data_keep %>%
  group_by(track_id, variable_num) %>%
  summarise(
    value = mean(value, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  group_by(track_id) %>%
  arrange(variable_num, .by_group = TRUE) %>%
  summarise(
    signal = list(interp_track(variable_num, value, grid_length = grid_length)),
    .groups = "drop"
  )

# convert signals to a matrix
signal_mat <- do.call(rbind, track_feature_list$signal)
signal_mat <- as.matrix(signal_mat)
colnames(signal_mat) <- paste0("x", seq_len(grid_length))

signal_df <- bind_cols(
  track_feature_list %>% select(track_id),
  as.data.frame(signal_mat)
)

# combine labels, month, and acoustic signal
model_df <- tracks_keep %>%
  select(track_id, species, Month) %>%
  left_join(signal_df, by = "track_id")

model_df_complete <- model_df %>%
  filter(if_all(starts_with("x"), ~ !is.na(.))) %>%
  mutate(
    species = factor(species, levels = c("Alewife","Rainbow Smelt")),
    Month = factor(Month)
  )

dim(model_df_complete)
table(model_df_complete$species)

# split into train, validation, and test sets
train_index <- createDataPartition(model_df_complete$species, p = 0.7, list = FALSE)

train_df <- model_df_complete[train_index, ]
temp_df  <- model_df_complete[-train_index, ]

val_index <- createDataPartition(temp_df$species, p = 0.5, list = FALSE)

validate_df <- temp_df[val_index, ]
test_df <- temp_df[-val_index, ]

# extract acoustic features
signal_cols <- grep("^x", names(model_df_complete), value = TRUE)

signal_train <- as.matrix(train_df[, signal_cols])
signal_validate <- as.matrix(validate_df[, signal_cols])
signal_test <- as.matrix(test_df[, signal_cols])

# standardize acoustic signal using training statistics
signal_means <- apply(signal_train, 2, mean)
signal_sds <- apply(signal_train, 2, sd)
signal_sds[signal_sds == 0] <- 1

signal_train <- scale(signal_train, center = signal_means, scale = signal_sds)
signal_validate <- scale(signal_validate, center = signal_means, scale = signal_sds)
signal_test <- scale(signal_test, center = signal_means, scale = signal_sds)

# create month dummy variables
meta_formula <- ~ Month - 1

meta_train <- model.matrix(meta_formula, data = train_df)
meta_validate <- model.matrix(meta_formula, data = validate_df)
meta_test <- model.matrix(meta_formula, data = test_df)

# combine signal and month for DNN
x_train_dnn <- cbind(signal_train, meta_train)
x_validate_dnn <- cbind(signal_validate, meta_validate)
x_test_dnn <- cbind(signal_test, meta_test)

# create signal arrays for CNN
x_train <- array(as.numeric(signal_train),
                 dim = c(nrow(signal_train), ncol(signal_train), 1))

x_validate <- array(as.numeric(signal_validate),
                    dim = c(nrow(signal_validate), ncol(signal_validate), 1))

x_test <- array(as.numeric(signal_test),
                dim = c(nrow(signal_test), ncol(signal_test), 1))

# extract labels
y_train <- train_df$species
y_validate <- validate_df$species
y_test <- test_df$species

# convert labels to one-hot format
dummy_y_train <- to_categorical(as.integer(y_train) - 1, 2)
dummy_y_val <- to_categorical(as.integer(y_validate) - 1, 2)
dummy_y_test <- to_categorical(as.integer(y_test) - 1, 2)

# save processed data
save(
  x_train, x_validate, x_test,
  x_train_dnn, x_validate_dnn, x_test_dnn,
  dummy_y_train, dummy_y_val, dummy_y_test,
  y_train, y_validate, y_test,
  train_df, validate_df, test_df,
  signal_cols, grid_length,
  file = "Data/fish_model_data_month_signal.RData"
)