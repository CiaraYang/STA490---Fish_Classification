## RNN Model (strict split)

### Step 1: Data Processing
- [`1. RNN data process.R`](Models/rnn/1.%20RNN%20data%20process.R)  

### Step 2: Hyperparameter Tuning
- [`2. RNN Hyperparameter Tuning.R`](Models/rnn/2.%20RNN%20Hyperparameter%20Tuning.R) — local  
- [`rnn_hyperparameter_tuning_server.R`](Models/rnn/rnn_hyperparameter_tuning_server.R) — server  

### Step 3: Model Training
- [`3. RNN model training.R`](Models/rnn/3.%20RNN%20model%20training.R)  

### Server
- [`single_job_rnn.sh`](Models/rnn/single_job_rnn.sh)  
- [`rnn_array_single_cpu.sh`](Models/rnn/rnn_array_single_cpu.sh)  
- [`drac/`](Models/rnn/drac/)  

### Other
- [`grid.search.full.rds`](Models/rnn/grid.search.full.rds)

## RNN Model (sliding window)
### Step 1: Data Processing
- [`RNN data process_stride 3 window.R`](Models/rnn/RNN%20data%20process_stride%203%20window.R)  

### Step 2: Hyperparameter Tuning
- [`rnn_hyperparameter_tuning_stride3_no_threshold_tuning.R`](Models/rnn/rnn_hyperparameter_tuning_stride3_no_threshold_tuning.R) 

### Step 3: Model Training
- [`rnn_final_model_selection_streide3_no_threshold_tuning.R`](Models/rnn/rnn_final_model_selection_streide3_no_threshold_tuning.R)

### Step 4: Testing
- - [`rnn_final_model_testing_stride3_no_threshold_turning.R`](Models/rnn/rnn_final_model_testing_stride3_no_threshold_turning.R)  
