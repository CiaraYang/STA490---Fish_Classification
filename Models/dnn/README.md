## DNN Model

### Step 2: Hyperparameter Tuning
- [`2.hyperparameter_tuning_local.R`](Models/dnn/2.hyperparameter_tuning_local.R) — 20 random combos  
- [`dnn_hyperparameter_tuning_server.R`](Models/dnn/dnn_hyperparameter_tuning_server.R) — all 216 combos  

### Step 3: Model Training
- [`3.model_training.R`](Models/dnn/3.model_training.R) — top 20 (from 216 by val loss) → test  
- [`3.model_training_best_configuration.R`](Models/dnn/3.model_training_best_configuration.R) — best model  
- [`3.model_training_local.R`](Models/dnn/3.model_training_local.R) — local test  

### Server
- [`single_job_dnn.sh`](Models/dnn/single_job_dnn.sh)  
- [`dnn_array_single_cpu.sh`](Models/dnn/dnn_array_single_cpu.sh)  
- [`drac/`](Models/dnn/drac/)  

### Other
- [`grid.search.full.rds`](Models/dnn/grid.search.full.rds)
