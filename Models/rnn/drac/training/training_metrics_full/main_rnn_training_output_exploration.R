
grid.search.full = readRDS(file = "Models/rnn/grid.search.full.rds")
grid.search.full = as_tibble(grid.search.full) %>% mutate(model_id = row_number())

final_data_b1 = readRDS("Models/rnn/drac/training/training_metrics_full/val_metrics_b1.rds")
final_data_b2 = readRDS("Models/rnn/drac/training/training_metrics_full/val_metrics_b2.rds")
final_data_b3 = readRDS("Models/rnn/drac/training/training_metrics_full/val_metrics_b3.rds")
final_data_b4 = readRDS("Models/rnn/drac/training/training_metrics_full/val_metrics_b4.rds")

final_data = rbind(final_data_b1,
                   final_data_b2,
                   final_data_b3,
                   final_data_b4)

final_data %>% 
  left_join(grid.search.full, by = "model_id") %>% 
  arrange(val_loss) %>% 
  slice(1:10)

# Store best 20 configurations
top20_configs = final_data %>% 
  left_join(grid.search.full, by = "model_id") %>% 
  arrange(val_loss) %>% 
  slice(1:20)

# Stire best 20 models
top20_models = top20_configs %>% 
  select(model_id)

saveRDS(top20_configs, "Models/rnn/drac/training/training_metrics_full/top20_configs.rds")
saveRDS(top20_models, "Models/rnn/drac/training/training_metrics_full/top20_models.rds")

