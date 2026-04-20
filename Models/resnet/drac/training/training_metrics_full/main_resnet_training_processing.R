#!/usr/bin/env Rscript

library(optparse)
library(dplyr)
library(tibble)

option_list <- list(
  make_option(
    c("-b", "--batch"),
    type = "integer",
    default = 0,
    help = "Batch ID [default = %default]",
    metavar = "integer"
  )
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)

nbatch <- opt$batch

setwd("../../../../../")

# -----------------------------
# Read the exact grid used in tuning
# -----------------------------
grid_path <- "Models/resnet/grid.search.full.rds"
grid.search.full <- readRDS(grid_path)
n_models <- nrow(grid.search.full)

cat("Grid file:", grid_path, "\n")
cat("Number of valid configurations in grid:", n_models, "\n")

# -----------------------------
# Read all training outputs for this batch
# -----------------------------
metrics_dir <- paste0("Models/resnet/drac/training/training_metrics_b", nbatch)

if (!dir.exists(metrics_dir)) {
  stop("Metrics directory does not exist: ", metrics_dir)
}

metric_files <- list.files(
  path = metrics_dir,
  pattern = "^training_output_[0-9]+\\.rds$",
  full.names = TRUE
)

cat("Number of metric files found in folder:", length(metric_files), "\n")

if (length(metric_files) == 0) {
  stop("No training_output_*.rds files found in: ", metrics_dir)
}

# -----------------------------
# Read and combine safely
# -----------------------------
read_one_metric <- function(f) {
  x <- readRDS(f)
  
  # ensure vector shape
  x <- as.numeric(x)
  
  if (length(x) != 7) {
    warning("Skipping file with unexpected length: ", f)
    return(NULL)
  }
  
  tibble(
    val_loss = x[1],
    best_epoch_loss = x[2],
    best_epoch_auc = x[3],
    val_auc = x[4],
    val_loss_at_best_auc = x[5],
    val_accuracy_at_best_auc = x[6],
    model_id = as.integer(x[7]),
    source_file = basename(f)
  )
}

all_metrics_list <- lapply(metric_files, read_one_metric)
all_metrics_list <- all_metrics_list[!vapply(all_metrics_list, is.null, logical(1))]
raw_metrics <- bind_rows(all_metrics_list)

cat("Rows read before cleaning:", nrow(raw_metrics), "\n")

# -----------------------------
# Diagnose bad rows
# -----------------------------
bad_id_rows <- raw_metrics %>%
  filter(model_id < 1 | model_id > n_models)

dup_id_rows <- raw_metrics %>%
  count(model_id, name = "n_rows") %>%
  filter(n_rows > 1)

cat("Rows with invalid model_id:", nrow(bad_id_rows), "\n")
cat("Duplicated model_id values:", nrow(dup_id_rows), "\n")

if (nrow(bad_id_rows) > 0) {
  cat("Invalid model_id rows:\n")
  print(bad_id_rows %>% select(model_id, source_file))
}

if (nrow(dup_id_rows) > 0) {
  cat("Duplicated model_id values:\n")
  print(dup_id_rows)
}

# -----------------------------
# Clean:
# 1) keep only valid model IDs from current grid
# 2) if duplicated, keep the one with smallest val_loss
# -----------------------------
final_data <- raw_metrics %>%
  filter(model_id >= 1, model_id <= n_models) %>%
  arrange(model_id, val_loss) %>%
  group_by(model_id) %>%
  slice(1) %>%
  ungroup()

cat("Rows after cleaning:", nrow(final_data), "\n")

# -----------------------------
# Save cleaned metrics
# -----------------------------
out_path <- paste0("Models/resnet/drac/training/val_metrics_b", nbatch, "_clean.rds")
saveRDS(final_data, file = out_path)

cat("Cleaned validation metrics saved to:\n", out_path, "\n")
cat("Processing complete for batch ", nbatch, "\n", sep = "")