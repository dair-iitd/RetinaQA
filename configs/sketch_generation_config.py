# data paths


BASE_DIR = "~/RETINAQA/data/grailqa"
ablation_type = "none" # Other ablations not implemented

sketch_data_train_path = f"{BASE_DIR}/dataset/grailqa_v1.0_train_sketch.json"
sketch_data_dev_path = f"{BASE_DIR}/dataset/grailqa_v1.0_dev_sketch.json"
sketch_data_test_path = f"{BASE_DIR}/dataset/grailqa_v1.0_test_sketch.json"

model_name = "t5-base"


# log paths
model_save_dir = f"{BASE_DIR}/../../saved_models/grailqa/sketch_generation/run_{model_name}_{ablation_type}/"
output_log_dir = f"{BASE_DIR}/../../logs/grailqa/sketch_generation/run_{model_name}_{ablation_type}/"
