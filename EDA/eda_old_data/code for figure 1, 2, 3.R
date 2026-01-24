setwd("D:/u of t/sta490/EDA")
load("processed_AnalysisData_no200.RData")
df <- processed_data_no200

library(dplyr)
library(shiny)
library(ggplot2)
library(ggiraph)
library(gridExtra)

df <- df %>%
  mutate(
    fishNum = as.factor(fishNum),
    Angle_major_axis = as.numeric(Angle_major_axis),
    Angle_minor_axis = as.numeric(Angle_minor_axis),
    Distance_minor_axis = as.numeric(Distance_minor_axis),
    Distance_major_axis = as.numeric(Distance_major_axis),
    TS_mean = as.numeric(TS_mean),
    aspectAngle = as.numeric(aspectAngle)
  )

library(lubridate)

df <- df %>%
  mutate(
    dateTimeSample = ymd_hms(dateTimeSample, quiet = TRUE)
  )

library(tidyverse)

glimpse(df)
summary(df)


fish_TS <- df %>%
  group_by(spCode, fishNum) %>%
  summarise(
    TS_mean_avg = mean(TS_mean, na.rm = TRUE),
    .groups = "drop"
  )

ggplot(
  fish_TS %>% filter(spCode %in% c(81, 316)),
  aes(x = TS_mean_avg, fill = factor(spCode))
) +
  scale_fill_discrete(
    labels = c("81" = "LT (81)", "316" = "SMB (316)")
  )+
  geom_density(alpha = 0.5) +
  theme_minimal() +
  labs(
    title = "Fish-level TS_mean distributions (LT vs SMB)",
    x = "Mean TS_mean",
    y = "Density",
    fill = "Species"
  )


library(tidyverse)

library(tidyverse)

df_sub <- df %>%
  filter(spCode %in% c(81, 316))

library(dplyr)
library(tidyr)
library(ggplot2)

# 0) Keep only the two target species and valid fish IDs
df_sub_2sp <- df_sub %>%
  filter(spCode %in% c(81, 316)) %>%
  filter(!is.na(fishNum))

# 1) Pivot ONLY F45 ... F170 (including .5) and build long table
df_long <- df_sub_2sp %>%
  pivot_longer(
    cols = matches("^F(\\d+(\\.5)?)$"),
    names_to = "Region",
    values_to = "TS"
  ) %>%
  mutate(
    Region_num = as.numeric(sub("^F", "", Region)),
    Species = factor(spCode, levels = c(81, 316),
                     labels = c("LT (81)", "SMB (316)")),
    TS = as.numeric(TS)
  ) %>%
  # 2) Remove sentinel / impossible TS values that destroy the scale
  #    (this catches typical float missing codes like -3.4e38)
  mutate(
    TS = ifelse(!is.finite(TS) | TS < -1e5 | TS > 1e5, NA_real_, TS)
  ) %>%
  filter(Region_num >= 45, Region_num <= 170) %>%
  filter(!is.na(TS))

# 3) Species mean profile per region
df_mean <- df_long %>%
  group_by(Species, Region_num) %>%
  summarise(mean_TS = mean(TS, na.rm = TRUE), .groups = "drop")

# 4) Plot: light individual fish lines + bold species mean
p_profile <- ggplot(df_long, aes(x = Region_num, y = TS)) +
  geom_line(aes(group = fishNum, color = Species),
            alpha = 0.03, linewidth = 0.25) +
  geom_line(data = df_mean,
            aes(x = Region_num, y = mean_TS, color = Species),
            linewidth = 1.2) +
  labs(
    title = "Fish-level TS profiles across regions (LT vs SMB)",
    subtitle = "Light lines: individual fish | Bold lines: species mean",
    x = "Region (F45–F170)",
    y = "Target Strength (TS)"
  ) +
  theme_minimal()

p_profile





library(tidyverse)

# --- 0) Filter to the two species
df_81316 <- df %>%
  filter(spCode %in% c(81, 316)) %>%
  mutate(
    Species = factor(spCode, levels = c(81, 316),
                     labels = c("81 (LT)", "316 (SMB)"))
  )

# --- 1) Fish-level counts: how many unique fish per species?
fish_counts <- df_81316 %>%
  distinct(Species, fishNum) %>%
  count(Species, name = "n_fish")

# --- 2) Track-length (rows/pings) per fish
track_lengths <- df_81316 %>%
  count(Species, fishNum, name = "n_pings")

# --- Optional: overall row counts per species (sometimes useful)
row_counts <- df_81316 %>%
  count(Species, name = "n_rows")

# --- Print summary (nice for your report text)
print(fish_counts)
print(row_counts)
print(summary(track_lengths$n_pings))

# --- 3) Plots
p_fish_count <- ggplot(fish_counts, aes(x = Species, y = n_fish, fill = Species)) +
  geom_col(width = 0.6) +
  labs(title = "Class imbalance (unique fish)", x = NULL, y = "Number of fish") +
  theme_minimal() +
  theme(legend.position = "none")

p_track_imbalance <- ggplot(track_lengths, aes(x = n_pings, fill = Species)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  scale_x_log10() +
  labs(title = "Track-length imbalance (pings per fish, log scale)",
       x = "Pings per fish (log10)", y = "Number of fish") +
  theme_minimal()

# If you have patchwork installed:
# install.packages("patchwork")
library(patchwork)

(p_fish_count | p_track_imbalance) +
  plot_annotation(title = "Imbalance check for LT (81) vs SMB (316)")


