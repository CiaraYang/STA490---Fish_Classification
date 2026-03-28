
grid.search.full = readRDS(file = "Models/cnn/grid.search.full.rds")
grid.search.full = as_tibble(grid.search.full) %>% mutate(model_id = row_number())

final_data_b1 = readRDS("Models/cnn/drac/training/training_metrics_full/val_metrics_b1.rds")
final_data_b2 = readRDS("Models/cnn/drac/training/training_metrics_full/val_metrics_b2.rds")
final_data_b3 = readRDS("Models/cnn/drac/training/training_metrics_full/val_metrics_b3.rds")
final_data_b4 = readRDS("Models/cnn/drac/training/training_metrics_full/val_metrics_b4.rds")
final_data_b5 = readRDS("Models/cnn/drac/training/training_metrics_full/val_metrics_b5.rds")
final_data_b6 = readRDS("Models/cnn/drac/training/training_metrics_full/val_metrics_b6.rds")

final_data = rbind(final_data_b1,
                   final_data_b2,
                   final_data_b3,
                   final_data_b4,
                   final_data_b5,
                   final_data_b6)

final_data %>% left_join(grid.search.full) %>% 
  arrange(val_loss) %>% 
  filter(row_number() <= 10)
