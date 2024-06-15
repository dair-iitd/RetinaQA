# data paths

BASE_DIR = "~/RETINAQA/data/grailqa"


data_train_path = f"{BASE_DIR}/dataset/grailqa_v1.0_train.json"
data_dev_path = f"{BASE_DIR}/dataset/grailqa_v1.0_dev.json"
data_test_path = f"{BASE_DIR}/dataset/grailqa_v1.0_test.json"

lfi_output_train_path = f"{BASE_DIR}/lf_integrator/lfs_cache_train_none.json"
lfi_output_dev_path = f"{BASE_DIR}/lf_integrator/lfs_cache_dev_none.json"
lfi_output_test_path = f"{BASE_DIR}/lf_integrator/lfs_cache_test_none.json"

lfr_output_train_path = f"{BASE_DIR}/lf_retriever/ranking_topk_results.json"
lfr_output_dev_path = f"{BASE_DIR}/lf_retriever/ranking_topk_results.json"
lfr_output_test_path = f"{BASE_DIR}/lf_retrieverranking_topk_results.json"

el_output_dev_el_path = f"{BASE_DIR}/entity_linker/dev_el_results.json"
el_output_test_el_path = f"{BASE_DIR}/entity_linker/test_el_results.json"
el_output_train_nk_el_path = ""


# flag variables
IS_AU_TRAINING = False
PATIENCE = 2
ablation_type = "none" #minus_SP_SR, minus_LFI, minus_LFR,none

# log paths
save_dir = f"{BASE_DIR}/../../saved_models/grailqa/discriminator/run_1_{ablation_type}"
log_dir = f"{BASE_DIR}/../../logs/grailqa/discriminator/run_1_{ablation_type}"

# default training config 

# gradient_accumulation_steps = 1
# num_train_epochs = 10
# train_batch_size = 4
# learning_rate = 1e-4
# num_neg_samples = 64
warmup_ratio = 0.01
warmup_steps = 0
save_every_step = 1000