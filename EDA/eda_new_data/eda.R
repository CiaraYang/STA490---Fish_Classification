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
  filter(max_diff_depth < 0.6) %>%
  ggplot(aes(x = Region_name)) +
  geom_hline(
    data = thermo_depth_csv %>% filter(kHz != "200"),
    aes(yintercept = thermo_depth),
    linetype="dashed",
    color="blue"
  ) +
  geom_errorbar(aes(ymin=min_depth,
                    ymax=max_depth)) +
  facet_wrap(~Year + Month + Location + kHz,
             scales="free_x") +
  theme_bw() +
  theme(axis.text.x = element_blank())

tracks_200 %>%
  filter(max_diff_depth < 0.6) %>%
  ggplot(aes(x = Region_name)) +
  geom_hline(
    data = thermo_depth_csv %>% filter(kHz == "200"),
    aes(yintercept = thermo_depth),
    linetype = "dashed",
    color = "blue"
  ) +
  geom_errorbar(aes(ymin = min_depth, ymax = max_depth)) +
  facet_wrap(~Year + Month + Location, scales = "free_x") +
  theme_bw() +
  theme(axis.text.x = element_blank()) +
  labs(title = "Track Min/Max Depth vs Thermocline (200 kHz)")







