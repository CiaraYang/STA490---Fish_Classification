load("Data/old_data.Rdata")
df <- processed_data_no200

library(dplyr)
library(shiny)
library(ggplot2)
library(ggiraph)
library(gridExtra)
library(scales)

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

na_rate <- sort(colMeans(is.na(df)), decreasing = TRUE)
head(na_rate, 15)

# helper function
num_cols <- df %>%
  select(where(is.numeric)) %>%
  names()

freq_cols <- names(df)[grepl("^F\\d+$", names(df))]
freq_cols <- freq_cols[order(as.integer(sub("^F", "", freq_cols)))]

# speed and TS_mean over time for 81 and 316
df_long <- df %>%
  filter(spCode %in% c(81, 316), !is.na(dateTimeSample)) %>%
  select(spCode, dateTimeSample, Speed_4D_mean_unsmoothed, TS_mean) %>%
  pivot_longer(
    cols = c(Speed_4D_mean_unsmoothed, TS_mean),
    names_to = "variable",
    values_to = "value"
  ) %>%
  mutate(
    variable = recode(variable,
                      Speed_4D_mean_unsmoothed = "Speed",
                      TS_mean = "TS_mean"),
    spCode = as.factor(spCode)
  )

df_long$spCode <- factor(
  df_long$spCode,
  levels = c("81", "316"),
)
ggplot(df_long, aes(x = dateTimeSample, y = value, color = spCode)) +
  geom_point(alpha = 0.2, size = 0.5) +
  facet_grid(variable ~ spCode, scales = "free_y") +
  scale_color_manual(
    values = c("81" = "#F8766D", "316" = "#00BFC4"),
    labels = c("81" = "LT (81)", "316" = "SMB (316)")
  ) +
  labs(
    title = "Speed and TS_mean over time",
    x = "Time",
    y = "Value",
    color = "Species"
  ) +
  theme_minimal(base_size = 12) +
  theme(
    strip.text = element_text(face = "bold"),
    legend.position = "right"
  )

