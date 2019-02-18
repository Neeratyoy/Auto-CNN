# --------------
# | BASIC BOHB |
# --------------

# python3 bohb_main.py --dataset KMNIST --eta 2 --min_budget 1 --max_budget 16 --n_iterations 10 --out_dir experiments/bohb/KMNIST/2_1_16

# python3 bohb_main.py --dataset KMNIST --eta 3 --min_budget 1 --max_budget 9 --n_iterations 20 --out_dir experiments/bohb/KMNIST/3_1_9

# python3 bohb_main.py --dataset KMNIST --eta 4 --min_budget 1 --max_budget 16 --n_iterations 20 --out_dir experiments/bohb/KMNIST/4_1_16

# taskset -c 1 python3 bohb_main.py --dataset KMNIST --eta 3 --min_budget 2 --max_budget 20 --n_iterations 20 --out_dir experiments/bohb/KMNIST/3_2_20

# taskset -c 0 python3 bohb_main.py --dataset K49 --eta 3 --min_budget 1 --max_budget 9 --n_iterations 10 --out_dir experiments/bohb/K49/3_1_9

# # ----------------------------
# # | GENERATE RESULT FOR BOHB |
# # ----------------------------

# taskset -c 0 python3 generate_result.py --dataset KMNIST --epochs 20 --config_dir experiments/bohb/KMNIST/2_1_16/ 

# taskset -c 1 python3 generate_result.py --dataset KMNIST --epochs 20 --config_dir experiments/bohb/KMNIST/3_1_9

# taskset -c 0 python3 generate_result.py --dataset KMNIST --epochs 20 --config_dir experiments/bohb/KMNIST/4_1_16

# taskset -c 0 python3 generate_result.py --dataset KMNIST --epochs 20 --config_dir experiments/bohb/KMNIST/3_2_20/

# taskset -c 1 python3 generate_result.py --dataset K49 --epochs 20 --config_dir experiments/bohb/K49/3_1_9

# # ------------------
# # | TRANSFER LEARN |
# # ------------------

# taskset -c 0 python3 transfer_learning.py --dataset_source KMNIST --dataset_dest K49 --config_dir experiments/bohb/KMNIST/2_1_16/ --epochs_source 20 --epochs_dest 20 > experiments/bohb/KMNIST/2_1_16/results.txt

# taskset -c 1 python3 transfer_learning.py --dataset_source KMNIST --dataset_dest K49 --config_dir experiments/bohb/KMNIST/3_1_9/ --epochs_source 10 --epochs_dest 20  > experiments/bohb/KMNIST/3_1_9/results.txt

# taskset -c 0 python3 transfer_learning.py --dataset_source KMNIST --dataset_dest K49 --config_dir experiments/bohb/KMNIST/4_1_16/ --epochs_source 11 --epochs_dest 20

# taskset -c 0 python3 transfer_learning.py --dataset_source KMNIST --dataset_dest K49 --config_dir experiments/bohb/KMNIST/3_2_20 --epochs_source 20 --epochs_dest 20

# taskset -c 1 python3 transfer_learning.py --dataset_source K49 --dataset_dest KMNIST --config_dir experiments/bohb/K49/3_1_9 --epochs_source 8 --epochs_dest 20


# # -------------------
# # | TRANSFER CONFIG |
# # -------------------

# taskset -c 0 python3 transfer_config.py --dataset K49 --config_dir experiments/bohb/KMNIST/3_2_20 --eta 3 --min_budget 1 --max_budget 9 --n_iterations 10 --out_dir experiments/transfer_config/KMNIST_3_2_20_K49_3_1_9/

# taskset -c 0 python3 transfer_config.py --dataset K49 --config_dir experiments/bohb/KMNIST/4_1_16 --eta 3 --min_budget 1 --max_budget 9 --n_iterations 3 --out_dir experiments/transfer_config/KMNIST_4_1_16_K49_3_1_9/

# taskset -c 1 python3 transfer_config.py --dataset K49 --config_dir experiments/bohb/KMNIST/3_2_20 --eta 4 --min_budget 1 --max_budget 16 --n_iterations 7 --out_dir experiments/transfer_config/KMNIST_3_2_20_K49_4_1_16/

taskset -c 0 python3 transfer_config.py --dataset K49 --config_dir experiments/bohb/KMNIST/4_1_16 --eta 4 --min_budget 1 --max_budget 16 --n_iterations 5 --out_dir experiments/transfer_config/KMNIST_4_1_16_K49_4_1_16/


# # ---------------------------------------
# # | GENERATE RESULT FOR TRANSFER CONFIG |
# # ---------------------------------------

# taskset -c 0 python3 generate_result.py --transfer True --dataset K49 --epochs 20 --config_dir experiments/transfer_config/KMNIST_3_2_20_K49_3_1_9/

# taskset -c 0 python3 generate_result.py --transfer True --dataset K49 --epochs 20 --config_dir experiments/transfer_config/KMNIST_4_1_16_K49_3_1_9/

taskset -c 0 python3 generate_result.py --transfer True --dataset K49 --epochs 20 --config_dir experiments/transfer_config/KMNIST_4_1_16_K49_4_1_16/