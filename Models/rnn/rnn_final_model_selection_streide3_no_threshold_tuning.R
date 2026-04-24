library(dplyr)

# -------------------------
# 1. Load and merge all RDS files
# -------------------------
base_dir <- "Models/rnn/drac/training/training_metrics_all/training_metrics_b"

all_results <- list()

for (b in 1:4) {
  folder <- paste0(base_dir, b)
  files <- list.files(folder, pattern = "\\.rds$", full.names = TRUE)
  
  for (f in files) {
    result <- readRDS(f)
    all_results[[length(all_results) + 1]] <- result
  }
}

results_df <- do.call(rbind, lapply(all_results, function(x) as.data.frame(t(x)))) %>%
  arrange(val_loss)

print(paste("Total models loaded:", nrow(results_df)))

grid.search.full <- readRDS("Models/rnn/grid.search.full.no.threshold.tune.rds") %>%
  mutate(model_id = row_number())

results_df <- results_df %>%
  left_join(grid.search.full, by = "model_id")

# -------------------------
# 2. Top 20 by validation loss
# -------------------------
top20 <- results_df %>%
  arrange(val_loss) %>%
  slice(1:20)

print("Top 20 models by validation loss:")
print(top20)

# -------------------------
# 3. Apply filtering rules
# -------------------------
# filtered <- top20 %>%
#   filter(
#     val_specificity >= 0.9,
#     val_sensitivity >= 0.8,
#     val_bal_acc     >= 0.85
#   )
# 
# print(paste("Models passing all filters:", nrow(filtered)))

# -------------------------
# 4. Final pick
# -------------------------
final_model <- results_df %>%
  arrange(desc(val_bal_acc)) %>%
  slice(1)

print("Final selected model:")
print(final_model)