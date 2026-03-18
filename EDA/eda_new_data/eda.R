# setup
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(forcats)
library(scales)

options(stringsAsFactors = FALSE)
theme_set(theme_bw())

# load merged data
full_data <- readRDS("Data/full_data_merged.rds")
str(full_data)

# keep 70 kHz data (variable == "45")
data_70 <- full_data %>%
  filter(variable == "45") %>%
  filter(!is.na(Depth), !is.na(value))

# keep 200 kHz data (variable == "170")
data_200 <- full_data %>%
  filter(variable == "170") %>%
  filter(!is.na(Depth), !is.na(value))

# compute track-level depth differences
# 70 kHz
tracks_70 <- data_70 %>%
  group_by(Region_name, Year, Month, Location, kHz) %>%
  summarise(
    min_depth = min(Depth, na.rm = TRUE),
    max_depth = max(Depth, na.rm = TRUE),
    max_diff_depth = max_depth - min_depth,
    n_pings = n(),
    .groups = "drop"
  )
# 200 kHz
tracks_200 <- data_200 %>%
  group_by(Region_name, Year, Month, Location, kHz) %>%
  summarise(
    min_depth = min(Depth, na.rm = TRUE),
    max_depth = max(Depth, na.rm = TRUE),
    max_diff_depth = max_depth - min_depth,
    n_pings = n(),
    .groups = "drop"
  )

# compute track-level depth differences
# 70 kHz
track_depth_70 <- tracks_70 %>%
  select(-n_pings)
# 200 kHz
track_depth_200 <- tracks_200 %>%
  select(-n_pings)

# track-level summary table by Year/Month/Location
# 70 kHz
table_tracks_70 <- tracks_70 %>%
  group_by(Year, Month, Location) %>%
  summarise(
    n_tracks = n(),
    mean_diff_depth = mean(max_diff_depth, na.rm = TRUE),
    median_diff_depth = median(max_diff_depth, na.rm = TRUE),
    sd_diff_depth = sd(max_diff_depth, na.rm = TRUE),
    mean_pings = mean(n_pings, na.rm = TRUE),
    .groups = "drop"
  )
table_tracks_70
# 200 kHz
table_tracks_200 <- tracks_200 %>%
  group_by(Year, Month, Location) %>%
  summarise(
    n_tracks = n(),
    mean_diff_depth = mean(max_diff_depth, na.rm = TRUE),
    median_diff_depth = median(max_diff_depth, na.rm = TRUE),
    sd_diff_depth = sd(max_diff_depth, na.rm = TRUE),
    mean_pings = mean(n_pings, na.rm = TRUE),
    .groups = "drop"
  )
table_tracks_200

# total number of tracks
n_total_70 <- nrow(track_depth_70)
n_total_200 <- nrow(track_depth_200)

n_total_70
n_total_200

# summary of max depth difference
summary(track_depth_70$max_diff_depth)
summary(track_depth_200$max_diff_depth)

# histogram of maximum depth difference (70 vs 200)
bind_rows(
  track_depth_70 %>% mutate(freq = "70 kHz"),
  track_depth_200 %>% mutate(freq = "200 kHz")
) %>%
  ggplot(aes(x = max_diff_depth, fill = freq)) +
  geom_histogram(binwidth = 0.1,
                 alpha = 0.5,
                 position = "identity") +
  geom_vline(xintercept = c(0.3, 0.6, 1),
             linetype = "dashed") +
  coord_cartesian(xlim = c(0, 5)) +
  labs(
    x = "Maximum depth difference (m)",
    y = "Number of fish tracks",
    fill = "Frequency",
    title = "Depth Difference Distribution Comparison"
  )

# table of remaining/eliminated tracks under different thresholds (70 kHz)
thresholds <- c(0, 0.3, 0.6, 1)

remaining_70 <- sapply(thresholds, function(t) {
  sum(track_depth_70$max_diff_depth <= t)
})

eliminated_70 <- n_total_70 - remaining_70

percent_retained_70 <- round(remaining_70 / n_total_70 * 100, 1)
percent_removed_70 <- round(eliminated_70 / n_total_70 * 100, 1)

threshold_table_70 <- rbind(
  "Total Tracks" = rep(n_total_70, length(thresholds)),
  "Remaining Tracks" = remaining_70,
  "Eliminated Tracks" = eliminated_70,
  "Percentage Retained" = paste0(percent_retained_70, "%"),
  "Percentage Removed" = paste0(percent_removed_70, "%")
)

colnames(threshold_table_70) <- paste0(thresholds, "m")
threshold_table_70

# table of remaining/eliminated tracks under different thresholds (200 kHz)
remaining_200 <- sapply(thresholds, function(t) {
  sum(track_depth_200$max_diff_depth <= t)
})

eliminated_200 <- n_total_200 - remaining_200

percent_retained_200 <- round(remaining_200 / n_total_200 * 100, 1)
percent_removed_200 <- round(eliminated_200 / n_total_200 * 100, 1)

threshold_table_200 <- rbind(
  "Total Tracks" = rep(n_total_200, length(thresholds)),
  "Remaining Tracks" = remaining_200,
  "Eliminated Tracks" = eliminated_200,
  "Percentage Retained" = paste0(percent_retained_200, "%"),
  "Percentage Removed" = paste0(percent_removed_200, "%")
)

colnames(threshold_table_200) <- paste0(thresholds, "m")
threshold_table_200

# thermocline visualization
Year = c("2024","2025")
Month = c("June","Sept")
kHz = c("070","200")
Location = c("South","North")

csv_grid_2024 = expand.grid(
  Year=factor(2024),
  Month=Month,
  kHz=kHz,
  Location=Location)

csv_grid_2025 = expand.grid(
  Year=factor(2025),
  Month="June",
  kHz=kHz,
  Location=Location)

csv_grid = rbind(csv_grid_2024,csv_grid_2025)

thermo_depth_csv = tibble(
  csv_grid,
  thermo_depth=c(rep(c(12.3,20),times=4),12,12,12,12))

tracks_70 %>%
  mutate(recording_session = paste0(Year," ",Month,", ",Location," - ",kHz," kHz")) %>% 
  filter(max_diff_depth < 1) %>%
  ggplot(aes(x = Region_name)) +
  geom_hline(
    data = thermo_depth_csv %>%
      filter(kHz != "200") %>%
      mutate(recording_session = paste0(Year," ",Month,", ",Location," - ",kHz," kHz")),
    aes(yintercept = thermo_depth),
    linetype = "dashed",
    color = "blue"
  ) +
  geom_errorbar(aes(ymin = min_depth,
                    ymax = max_depth)) +
  scale_y_reverse() +
  facet_wrap(~ recording_session,
             scales = "free_x") +
  theme_bw() +
  theme(axis.text.x = element_blank()) +
  labs(
    title = "Track Min/Max Depth vs Thermocline (70 kHz)",
    y = "Depth (m, 0 at surface)"
  )

tracks_200 %>%
  mutate(recording_session = paste0(Year," ",Month,", ",Location," - ",kHz," kHz")) %>%
  filter(max_diff_depth < 1) %>%
  ggplot(aes(x = Region_name)) +
  geom_hline(
    data = thermo_depth_csv %>%
      filter(kHz == "200") %>%
      mutate(recording_session = paste0(Year," ",Month,", ",Location," - ",kHz," kHz")),
    aes(yintercept = thermo_depth),
    linetype = "dashed",
    color = "blue"
  ) +
  geom_errorbar(aes(ymin = min_depth, ymax = max_depth)) +
  scale_y_reverse() +
  facet_wrap(~ recording_session,
             scales = "free_x") +
  theme_bw() +
  theme(axis.text.x = element_blank()) +
  labs(
    title = "Track Min/Max Depth vs Thermocline (200 kHz)",
    y = "Depth (m, 0 at surface)"
  )

# label data
thermo_lookup <- thermo_depth_csv %>%
  select(Year, Month, thermo_depth) %>%
  distinct() %>%
  mutate(Year = as.character(Year))

full_data_clean <- full_data %>%
  filter(!is.na(Depth), !is.na(value)) %>%
  mutate(Year = as.character(Year))

track_labels_all <- full_data_clean %>%
  group_by(Region_name, Year, Month, Location, kHz) %>%
  summarise(
    min_depth = min(Depth, na.rm = TRUE),
    max_depth = max(Depth, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(
    thermo_lookup,
    by = c("Year", "Month")
  ) %>%
  mutate(
    species = case_when(
      is.na(thermo_depth) ~ NA_character_,
      max_depth < thermo_depth ~ "Alewife",
      min_depth > thermo_depth ~ "Rainbow Smelt",
      min_depth <= thermo_depth & max_depth >= thermo_depth ~ "Crossing"
    )
  ) %>%
  select(Region_name, Year, Month, Location, kHz, species)

full_data <- full_data %>%
  mutate(Year = as.character(Year)) %>%
  left_join(
    track_labels_all,
    by = c("Region_name", "Year", "Month", "Location", "kHz")
  )

data_70 <- data_70 %>%
  mutate(Year = as.character(Year)) %>%
  left_join(
    track_labels_all,
    by = c("Region_name", "Year", "Month", "Location", "kHz")
  )

data_200 <- data_200 %>%
  mutate(Year = as.character(Year)) %>%
  left_join(
    track_labels_all,
    by = c("Region_name", "Year", "Month", "Location", "kHz")
  )

# create labeled full dataset object for downstream summaries
full_data_labeled <- full_data

# track-level summary for imbalance analysis
track_summary <- full_data_labeled %>%
  filter(!is.na(Depth), !is.na(value)) %>%
  group_by(Region_name, Year, Month, Location, kHz) %>%
  summarise(
    species = first(na.omit(species)),
    min_depth = min(Depth, na.rm = TRUE),
    max_depth = max(Depth, na.rm = TRUE),
    max_diff_depth = max_depth - min_depth,
    n_pings = n(),
    .groups = "drop"
  ) %>%
  filter(!is.na(species))

# Look into spieces distribution
full_data_labeled %>%
  distinct(Region_name, Year, Month, Location, kHz, species) %>%
  count(species)

fish_counts_by_kHz <- full_data_labeled %>%
  distinct(Region_name, Year, Month, Location, kHz, species) %>%
  count(kHz, species) %>%
  group_by(kHz) %>%
  mutate(
    percent_within_kHz = round(n / sum(n) * 100, 1)
  ) %>%
  ungroup()

fish_counts_by_kHz

# Look into species distribution under different thresholds
no_threshold_results <- track_summary %>%
  count(species) %>%
  mutate(
    threshold = NA_real_,
    percent_within_threshold = round(n / sum(n) * 100, 1)
  )

threshold_results <- lapply(thresholds, function(t) {
  
  filtered_tracks <- track_summary %>%
    filter(max_diff_depth <= t)
  
  filtered_tracks %>%
    count(species) %>%
    mutate(
      threshold = t,
      percent_within_threshold = round(n / sum(n) * 100, 1)
    )
  
}) %>%
  bind_rows() %>%
  bind_rows(no_threshold_results) %>%
  arrange(threshold, species)

threshold_results

# Look into sample imbalance by transducer(after filtering)
imbalance_by_khz_no_unknown <- track_summary %>%
  filter(species %in% c("alewife", "rainbow smelt")) %>%
  count(kHz, species, name = "n_tracks") %>%
  group_by(kHz) %>%
  mutate(
    total_tracks = sum(n_tracks),
    percent_within_khz = round(n_tracks / total_tracks * 100, 1)
  ) %>%
  ungroup() %>%
  arrange(kHz, desc(n_tracks))

imbalance_by_khz_no_unknown
