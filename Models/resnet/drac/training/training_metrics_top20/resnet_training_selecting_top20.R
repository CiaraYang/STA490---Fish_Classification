#!/usr/bin/env Rscript

library(dplyr)
library(readr)
library(tibble)

# -----------------------------
# File paths
# -----------------------------

setwd("../../../../../")

grid_path <- "Models/resnet/grid.search.full.rds"
metrics_path <- "Models/resnet/drac/training/val_metrics_b1_clean.rds"

out_rds <- "Models/resnet/drac/training/top20_resnet_configs.rds"
out_csv <- "Models/resnet/drac/training/top20_resnet_configs.csv"

# -----------------------------
# Read files
# -----------------------------
grid.search.full <- readRDS(grid_path) %>%
  mutate(model_id = row_number())

val_metrics <- readRDS(metrics_path)

n_models <- nrow(grid.search.full)

cat("Grid size:", n_models, "\n")
cat("Rows in metrics file before cleaning:", nrow(val_metrics), "\n")

# -----------------------------
# Clean metrics just in case
# 1) keep valid model_id only
# 2) if duplicates exist, keep the one with lowest val_loss
# -----------------------------
val_metrics_clean <- val_metrics %>%
  mutate(model_id = as.integer(model_id)) %>%
  filter(model_id >= 1, model_id <= n_models) %>%
  arrange(model_id, val_loss, desc(val_auc)) %>%
  group_by(model_id) %>%
  slice(1) %>%
  ungroup()

cat("Rows in metrics file after cleaning:", nrow(val_metrics_clean), "\n")

# -----------------------------
# Pick top 20 configurations
# Tutor-style rule:
#   lowest validation loss first
# Tie-breaker:
#   higher validation AUC
# -----------------------------
top20_configs <- val_metrics_clean %>%
  arrange(val_loss, desc(val_auc)) %>%
  slice_head(n = 20) %>%
  left_join(grid.search.full, by = "model_id")

# -----------------------------
# Print result
# -----------------------------
cat("\nTop 20 configurations:\n")
print(top20_configs)

# -----------------------------
# Save result
# -----------------------------
saveRDS(top20_configs, out_rds)
write_csv(top20_configs, out_csv)

cat("\nSaved to:\n")
cat(out_rds, "\n")
cat(out_csv, "\n")