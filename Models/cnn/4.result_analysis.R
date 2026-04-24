library(dplyr)
library(tibble)
library(ggplot2)
library(readr)

options(stringsAsFactors = FALSE)

# =========================================================
# TRACK-LEVEL THERMOCLINE DISTANCE ANALYSIS
# Goal:
# Test whether track groups closer to the thermocline have
# higher misclassification rates than track groups farther away.
# =========================================================

# -------------------------------
# Paths
# -------------------------------
data_dir <- "Data"
test_out_dir <- "Models/cnn/drac/test"
analysis_out_dir <- "Models/cnn/drac/test/thermocline_analysis"

if (!dir.exists(analysis_out_dir)) {
  dir.create(analysis_out_dir, recursive = TRUE)
}

# -------------------------------
# Load saved objects
# -------------------------------
test_df <- readRDS(file.path(data_dir, "test_df.rds"))
track_distance_df <- readRDS(file.path(data_dir, "track_distance_df.rds"))
prediction_outputs <- readRDS(file.path(test_out_dir, "prediction_outputs.rds"))

# -------------------------------
# Basic checks
# -------------------------------
if (nrow(test_df) != nrow(prediction_outputs)) {
  stop("Row mismatch: test_df and prediction_outputs do not have the same number of rows.")
}

required_cols_track <- c(
  "track_id", "Region_name", "Year", "Month", "Location", "kHz", "species",
  "mid_depth", "thermo_depth", "distance_to_thermocline"
)

missing_track_cols <- setdiff(required_cols_track, names(track_distance_df))
if (length(missing_track_cols) > 0) {
  stop(
    paste0(
      "track_distance_df is missing required columns: ",
      paste(missing_track_cols, collapse = ", ")
    )
  )
}

# -------------------------------
# Merge predictions back to test_df
# -------------------------------
test_results <- test_df %>%
  mutate(row_id = seq_len(n())) %>%
  left_join(prediction_outputs, by = "row_id") %>%
  mutate(
    misclassified = true_label != pred_label,
    confidence = pmax(prob_smelt, prob_alewife),
    error_type = case_when(
      true_label == "Alewife" & pred_label == "Rainbow Smelt" ~ "Alewife -> Rainbow Smelt",
      true_label == "Rainbow Smelt" & pred_label == "Alewife" ~ "Rainbow Smelt -> Alewife",
      TRUE ~ "Correct"
    )
  )

# -------------------------------
# Summarize to track level
# -------------------------------
track_error_df <- test_results %>%
  group_by(track_id) %>%
  summarise(
    n_pings = n(),
    n_misclassified = sum(misclassified),
    track_error_rate = mean(misclassified),
    mean_confidence = mean(confidence, na.rm = TRUE),
    mean_prob_smelt = mean(prob_smelt, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  left_join(track_distance_df, by = "track_id")

# -------------------------------
# Check merge quality
# -------------------------------
if (any(is.na(track_error_df$distance_to_thermocline))) {
  warning("Some tracks are missing distance_to_thermocline after merge.")
}

# -------------------------------
# Create near / middle / far groups
# Use tertiles so the groups are balanced
# -------------------------------
q1 <- quantile(track_error_df$distance_to_thermocline, 1/3, na.rm = TRUE)
q2 <- quantile(track_error_df$distance_to_thermocline, 2/3, na.rm = TRUE)

track_error_df <- track_error_df %>%
  mutate(
    thermocline_group = case_when(
      distance_to_thermocline <= q1 ~ "Near",
      distance_to_thermocline <= q2 ~ "Middle",
      TRUE ~ "Far"
    )
  )

# -------------------------------
# Overall summaries
# -------------------------------
overall_ping_summary <- test_results %>%
  summarise(
    n_test_pings = n(),
    n_misclassified = sum(misclassified),
    ping_error_rate = mean(misclassified)
  )

overall_track_summary <- track_error_df %>%
  summarise(
    n_tracks = n(),
    mean_track_error_rate = mean(track_error_rate, na.rm = TRUE),
    median_track_error_rate = median(track_error_rate, na.rm = TRUE),
    mean_distance_to_thermocline = mean(distance_to_thermocline, na.rm = TRUE)
  )

group_summary <- track_error_df %>%
  group_by(thermocline_group) %>%
  summarise(
    n_tracks = n(),
    mean_distance = mean(distance_to_thermocline, na.rm = TRUE),
    median_distance = median(distance_to_thermocline, na.rm = TRUE),
    mean_track_error_rate = mean(track_error_rate, na.rm = TRUE),
    median_track_error_rate = median(track_error_rate, na.rm = TRUE),
    mean_n_pings = mean(n_pings, na.rm = TRUE),
    .groups = "drop"
  )

# -------------------------------
# Statistical tests
# -------------------------------
# 1) Correlation between distance and track error rate
cor_test <- suppressWarnings(
  cor.test(
    x = track_error_df$distance_to_thermocline,
    y = track_error_df$track_error_rate,
    method = "spearman",
    exact = FALSE
  )
)

cor_summary <- tibble(
  method = "Spearman correlation",
  estimate = unname(cor_test$estimate),
  p_value = cor_test$p.value
)

# 2) Simple linear regression
lm_fit <- lm(track_error_rate ~ distance_to_thermocline, data = track_error_df)
lm_coef <- summary(lm_fit)$coefficients

lm_summary_tbl <- tibble(
  term = rownames(lm_coef),
  estimate = lm_coef[, "Estimate"],
  std_error = lm_coef[, "Std. Error"],
  t_value = lm_coef[, "t value"],
  p_value = lm_coef[, "Pr(>|t|)"]
)

# 3) Compare Near vs Far groups
near_far_df <- track_error_df %>%
  filter(thermocline_group %in% c("Near", "Far"))

wilcox_res <- wilcox.test(
  track_error_rate ~ thermocline_group,
  data = near_far_df,
  exact = FALSE
)

near_far_summary <- near_far_df %>%
  group_by(thermocline_group) %>%
  summarise(
    n_tracks = n(),
    mean_track_error_rate = mean(track_error_rate, na.rm = TRUE),
    median_track_error_rate = median(track_error_rate, na.rm = TRUE),
    .groups = "drop"
  )

near_far_test_tbl <- tibble(
  test = "Wilcoxon rank-sum: Near vs Far",
  p_value = wilcox_res$p.value
)

# -------------------------------
# Optional weighted regression
# Weight by number of pings in each track
# -------------------------------
weighted_lm_fit <- lm(
  track_error_rate ~ distance_to_thermocline,
  data = track_error_df,
  weights = n_pings
)

weighted_lm_coef <- summary(weighted_lm_fit)$coefficients

weighted_lm_summary_tbl <- tibble(
  term = rownames(weighted_lm_coef),
  estimate = weighted_lm_coef[, "Estimate"],
  std_error = weighted_lm_coef[, "Std. Error"],
  t_value = weighted_lm_coef[, "t value"],
  p_value = weighted_lm_coef[, "Pr(>|t|)"]
)

# -------------------------------
# Auto interpretation text
# -------------------------------
near_mean <- near_far_summary$mean_track_error_rate[near_far_summary$thermocline_group == "Near"]
far_mean  <- near_far_summary$mean_track_error_rate[near_far_summary$thermocline_group == "Far"]
rho <- cor_summary$estimate[1]
rho_p <- cor_summary$p_value[1]

cat("\n================ TRACK-LEVEL THERMOCLINE ANALYSIS ================\n")
print(overall_ping_summary)
print(overall_track_summary)

cat("\n================ GROUP SUMMARY ================\n")
print(group_summary)

cat("\n================ CORRELATION TEST ================\n")
print(cor_summary)

cat("\n================ LINEAR REGRESSION ================\n")
print(lm_summary_tbl)

cat("\n================ WEIGHTED LINEAR REGRESSION ================\n")
print(weighted_lm_summary_tbl)

cat("\n================ NEAR VS FAR ================\n")
print(near_far_summary)
print(near_far_test_tbl)

cat("\n================ INTERPRETATION ================\n")

cat(
  paste0(
    "At the track-group level, the mean misclassification rate was ",
    round(overall_track_summary$mean_track_error_rate[1], 4),
    ". "
  )
)

cat(
  paste0(
    "The mean track error rate was ",
    round(near_mean, 4),
    " for groups near the thermocline and ",
    round(far_mean, 4),
    " for groups far from the thermocline. "
  )
)

if (!is.na(rho) && !is.na(rho_p)) {
  direction_text <- ifelse(rho < 0, "decreased", "increased")
  cat(
    paste0(
      "The Spearman correlation between distance to thermocline and track error rate was ",
      round(rho, 4),
      " (p = ", signif(rho_p, 4), "), suggesting that error rate ",
      direction_text,
      " as distance from the thermocline increased. "
    )
  )
}

if (!is.na(near_mean) && !is.na(far_mean)) {
  if (near_mean > far_mean) {
    cat("This pattern is consistent with the idea that track groups closer to the thermocline are harder to classify. ")
  } else {
    cat("This pattern does not support the idea that track groups closer to the thermocline are harder to classify. ")
  }
}

cat("\n")

# -------------------------------
# Save tables
# -------------------------------
write.csv(test_results,
          file.path(analysis_out_dir, "test_results_with_predictions.csv"),
          row.names = FALSE)

write.csv(track_error_df,
          file.path(analysis_out_dir, "track_error_df.csv"),
          row.names = FALSE)

write.csv(group_summary,
          file.path(analysis_out_dir, "thermocline_group_summary.csv"),
          row.names = FALSE)

write.csv(cor_summary,
          file.path(analysis_out_dir, "correlation_summary.csv"),
          row.names = FALSE)

write.csv(lm_summary_tbl,
          file.path(analysis_out_dir, "linear_regression_summary.csv"),
          row.names = FALSE)

write.csv(weighted_lm_summary_tbl,
          file.path(analysis_out_dir, "weighted_linear_regression_summary.csv"),
          row.names = FALSE)

write.csv(near_far_summary,
          file.path(analysis_out_dir, "near_far_summary.csv"),
          row.names = FALSE)

write.csv(near_far_test_tbl,
          file.path(analysis_out_dir, "near_far_test.csv"),
          row.names = FALSE)

write.csv(overall_ping_summary,
          file.path(analysis_out_dir, "overall_ping_summary.csv"),
          row.names = FALSE)

write.csv(overall_track_summary,
          file.path(analysis_out_dir, "overall_track_summary.csv"),
          row.names = FALSE)

saveRDS(track_error_df,
        file.path(analysis_out_dir, "track_error_df.rds"))

# -------------------------------
# Plots
# -------------------------------

# 1. Scatter plot with regression line
p1 <- ggplot(track_error_df, aes(x = distance_to_thermocline, y = track_error_rate)) +
  geom_point(alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE) +
  labs(
    title = "Track Error Rate vs Distance to Thermocline",
    x = "Distance to Thermocline",
    y = "Track Misclassification Rate"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(analysis_out_dir, "track_error_vs_distance.png"),
  plot = p1, width = 8, height = 5, dpi = 300
)

# 2. Boxplot by near/middle/far groups
p2 <- ggplot(track_error_df, aes(x = thermocline_group, y = track_error_rate)) +
  geom_boxplot() +
  labs(
    title = "Track Error Rate by Thermocline Distance Group",
    x = "Thermocline Distance Group",
    y = "Track Misclassification Rate"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(analysis_out_dir, "track_error_by_distance_group.png"),
  plot = p2, width = 7, height = 5, dpi = 300
)

# 3. Mean error rate by group
p3 <- ggplot(group_summary, aes(x = thermocline_group, y = mean_track_error_rate)) +
  geom_col() +
  labs(
    title = "Mean Track Error Rate by Thermocline Distance Group",
    x = "Thermocline Distance Group",
    y = "Mean Track Misclassification Rate"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(analysis_out_dir, "mean_track_error_by_group.png"),
  plot = p3, width = 7, height = 5, dpi = 300
)

# 4. Histogram of thermocline distance
p4 <- ggplot(track_error_df, aes(x = distance_to_thermocline)) +
  geom_histogram(bins = 30) +
  labs(
    title = "Distribution of Distance to Thermocline",
    x = "Distance to Thermocline",
    y = "Count of Track Groups"
  ) +
  theme_minimal()

ggsave(
  filename = file.path(analysis_out_dir, "distance_to_thermocline_histogram.png"),
  plot = p4, width = 7, height = 5, dpi = 300
)