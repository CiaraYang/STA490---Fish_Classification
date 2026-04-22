library(dplyr)
library(tibble)
library(ggplot2)
library(readr)

options(stringsAsFactors = FALSE)

# =========================================================
# CNN MISCLASSIFICATION ANALYSIS
# This script reads saved outputs from:
# 1) preprocessing script  -> Data/test_df.rds
# 2) final testing script  -> prediction_outputs.rds, metrics_tbl_cnn.rds
# =========================================================

# ---- Paths ----
out_dir <- "Models/cnn/drac/test"
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)

# ---- Load data ----
test_df <- readRDS("Data/test_df.rds")
prediction_outputs <- readRDS(file.path(out_dir, "prediction_outputs.rds"))
metrics_tbl_cnn <- readRDS(file.path(out_dir, "metrics_tbl_cnn.rds"))
threshold_results <- readRDS(file.path(out_dir, "threshold_results.rds"))

# ---- Check alignment ----
if (nrow(test_df) != nrow(prediction_outputs)) {
  stop("Row mismatch: test_df and prediction_outputs do not have the same number of rows.")
}

# ---- Merge metadata with predictions ----
test_results <- test_df %>%
  mutate(row_id = seq_len(n())) %>%
  left_join(prediction_outputs, by = "row_id") %>%
  mutate(
    misclassified = true_label != pred_label,
    predicted_prob = ifelse(pred_label == "Rainbow Smelt", prob_smelt, prob_alewife),
    error_type = case_when(
      true_label == "Alewife" & pred_label == "Rainbow Smelt" ~ "Alewife -> Rainbow Smelt",
      true_label == "Rainbow Smelt" & pred_label == "Alewife" ~ "Rainbow Smelt -> Alewife",
      TRUE ~ "Correct"
    ),
    confidence = pmax(prob_smelt, prob_alewife)
  )

# ---- Misclassified pings only ----
misclassified_pings <- test_results %>%
  filter(misclassified) %>%
  arrange(confidence)

# ---- Overall summary ----
overall_summary <- tibble(
  n_test_pings = nrow(test_results),
  n_misclassified = sum(test_results$misclassified),
  error_rate = mean(test_results$misclassified),
  accuracy = mean(!test_results$misclassified),
  mean_confidence_all = mean(test_results$confidence, na.rm = TRUE),
  mean_confidence_misclassified = mean(test_results$confidence[test_results$misclassified], na.rm = TRUE),
  mean_confidence_correct = mean(test_results$confidence[!test_results$misclassified], na.rm = TRUE)
)

# ---- Error type summary ----
error_type_summary <- test_results %>%
  filter(misclassified) %>%
  count(error_type, sort = TRUE) %>%
  mutate(proportion = n / sum(n))

# ---- Summary by "track_id" group ----
track_error_summary <- test_results %>%
  group_by(track_id, Region_name, Year, Month, Location, kHz, true_label) %>%
  summarise(
    n_pings = n(),
    n_misclassified = sum(misclassified),
    error_rate = mean(misclassified),
    mean_confidence = mean(confidence, na.rm = TRUE),
    mean_prob_smelt = mean(prob_smelt, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(error_rate), desc(n_misclassified))

# ---- Hard groups: require at least 10 pings so tiny groups do not dominate ----
hard_track_groups <- track_error_summary %>%
  filter(n_pings >= 10, n_misclassified > 0) %>%
  arrange(desc(error_rate), desc(n_misclassified))

# ---- Summary by location ----
location_error_summary <- test_results %>%
  group_by(Location) %>%
  summarise(
    n_pings = n(),
    n_misclassified = sum(misclassified),
    error_rate = mean(misclassified),
    .groups = "drop"
  ) %>%
  arrange(desc(error_rate))

# ---- Summary by month/year ----
month_error_summary <- test_results %>%
  group_by(Year, Month) %>%
  summarise(
    n_pings = n(),
    n_misclassified = sum(misclassified),
    error_rate = mean(misclassified),
    .groups = "drop"
  ) %>%
  arrange(desc(error_rate))

# ---- Confidence bands ----
confidence_summary <- test_results %>%
  mutate(
    confidence_band = case_when(
      confidence < 0.55 ~ "[0.50, 0.55)",
      confidence < 0.65 ~ "[0.55, 0.65)",
      confidence < 0.75 ~ "[0.65, 0.75)",
      confidence < 0.85 ~ "[0.75, 0.85)",
      TRUE ~ "[0.85, 1.00]"
    )
  ) %>%
  group_by(confidence_band) %>%
  summarise(
    n_pings = n(),
    n_misclassified = sum(misclassified),
    error_rate = mean(misclassified),
    .groups = "drop"
  )

# ---- Worst individual regions if useful ----
region_error_summary <- test_results %>%
  group_by(Region_name) %>%
  summarise(
    n_pings = n(),
    n_misclassified = sum(misclassified),
    error_rate = mean(misclassified),
    .groups = "drop"
  ) %>%
  arrange(desc(error_rate), desc(n_misclassified))

# ---- Print concise interpretation ----
cat("\n================ OVERALL SUMMARY ================\n")
print(overall_summary)

cat("\n================ ERROR TYPE SUMMARY ================\n")
print(error_type_summary)

cat("\n================ LOCATION ERROR SUMMARY ================\n")
print(location_error_summary)

cat("\n================ MONTH ERROR SUMMARY ================\n")
print(month_error_summary)

cat("\n================ CONFIDENCE SUMMARY ================\n")
print(confidence_summary)

cat("\n================ TOP HARD TRACK GROUPS ================\n")
print(head(hard_track_groups, 10))

# ---- Auto-written conclusion text ----
cat("\n================ INTERPRETATION ================\n")

overall_error_rate <- overall_summary$error_rate[1]
overall_mis <- overall_summary$n_misclassified[1]
overall_n <- overall_summary$n_test_pings[1]

cat(
  paste0(
    "The final CNN misclassified ", overall_mis, " of ", overall_n,
    " test pings, corresponding to an error rate of ",
    round(overall_error_rate, 4), ". "
  )
)

if (nrow(hard_track_groups) > 0) {
  worst_group <- hard_track_groups[1, ]
  cat(
    paste0(
      "Errors were concentrated in some difficult track groups rather than being perfectly uniform. ",
      "The hardest group was ", worst_group$track_id,
      ", with ", worst_group$n_misclassified, " misclassified pings out of ",
      worst_group$n_pings, " (error rate = ",
      round(worst_group$error_rate, 4), "). "
    )
  )
}

if (nrow(location_error_summary) > 0) {
  hardest_location <- location_error_summary %>% slice(1)
  cat(
    paste0(
      "Among locations, the highest error rate was observed in ",
      hardest_location$Location, " (",
      round(hardest_location$error_rate, 4), "). "
    )
  )
}

mean_conf_mis <- overall_summary$mean_confidence_misclassified[1]
mean_conf_cor <- overall_summary$mean_confidence_correct[1]

if (!is.na(mean_conf_mis) && !is.na(mean_conf_cor)) {
  cat(
    paste0(
      "Misclassified pings had mean prediction confidence ",
      round(mean_conf_mis, 4),
      ", compared with ",
      round(mean_conf_cor, 4),
      " for correctly classified pings. "
    )
  )
}

cat("\n")

# ---- Save outputs ----
write.csv(test_results, file.path(out_dir, "test_results_with_metadata.csv"), row.names = FALSE)
write.csv(misclassified_pings, file.path(out_dir, "misclassified_pings.csv"), row.names = FALSE)
write.csv(overall_summary, file.path(out_dir, "overall_misclassification_summary.csv"), row.names = FALSE)
write.csv(error_type_summary, file.path(out_dir, "error_type_summary.csv"), row.names = FALSE)
write.csv(track_error_summary, file.path(out_dir, "track_error_summary.csv"), row.names = FALSE)
write.csv(hard_track_groups, file.path(out_dir, "hard_track_groups.csv"), row.names = FALSE)
write.csv(location_error_summary, file.path(out_dir, "location_error_summary.csv"), row.names = FALSE)
write.csv(month_error_summary, file.path(out_dir, "month_error_summary.csv"), row.names = FALSE)
write.csv(region_error_summary, file.path(out_dir, "region_error_summary.csv"), row.names = FALSE)
write.csv(confidence_summary, file.path(out_dir, "confidence_summary.csv"), row.names = FALSE)

saveRDS(test_results, file.path(out_dir, "test_results_with_metadata.rds"))
saveRDS(misclassified_pings, file.path(out_dir, "misclassified_pings.rds"))
saveRDS(track_error_summary, file.path(out_dir, "track_error_summary.rds"))

# ---- Plot 1: confidence by correctness ----
p1 <- ggplot(test_results, aes(x = confidence, fill = misclassified)) +
  geom_histogram(position = "identity", alpha = 0.6, bins = 40) +
  labs(
    title = "Prediction Confidence: Correct vs Misclassified Pings",
    x = "Prediction Confidence",
    y = "Count",
    fill = "Misclassified"
  ) +
  theme_minimal()

ggsave(
  file.path(out_dir, "confidence_histogram.png"),
  plot = p1, width = 8, height = 5, dpi = 300
)

# ---- Plot 2: top hard groups ----
plot_tracks <- hard_track_groups %>%
  slice_head(n = 10) %>%
  mutate(group_label = paste0(track_id, " (n=", n_pings, ")"))

p2 <- ggplot(plot_tracks, aes(x = reorder(group_label, error_rate), y = error_rate)) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 10 Hardest Track Groups by Error Rate",
    x = "Track Group",
    y = "Misclassification Rate"
  ) +
  theme_minimal()

ggsave(
  file.path(out_dir, "top_hard_track_groups.png"),
  plot = p2, width = 10, height = 6, dpi = 300
)

# ---- Plot 3: location error rate ----
p3 <- ggplot(location_error_summary, aes(x = Location, y = error_rate)) +
  geom_col() +
  labs(
    title = "Misclassification Rate by Location",
    x = "Location",
    y = "Misclassification Rate"
  ) +
  theme_minimal()

ggsave(
  file.path(out_dir, "location_error_rate.png"),
  plot = p3, width = 7, height = 5, dpi = 300
)

# ---- Plot 4: predicted smelt probability by true class ----
p4 <- ggplot(test_results, aes(x = prob_smelt, fill = true_label)) +
  geom_histogram(position = "identity", alpha = 0.55, bins = 40) +
  labs(
    title = "Predicted Rainbow Smelt Probability by True Class",
    x = "Predicted Probability of Rainbow Smelt",
    y = "Count",
    fill = "True Class"
  ) +
  theme_minimal()

ggsave(
  file.path(out_dir, "prob_smelt_by_true_class.png"),
  plot = p4, width = 8, height = 5, dpi = 300
)
