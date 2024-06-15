# data paths

BASE_DIR = "~/RETINAQA/data/grailqa"

data_train_path = f"{BASE_DIR}/dataset/grailqa_v1.0_train.json"
data_dev_path = f"{BASE_DIR}/dataset/grailqa_v1.0_dev.json"
data_test_path = f"{BASE_DIR}/dataset/grailqa_v1.0_test.json"

sr_output_train_path = f"{BASE_DIR}/schema_retriever/dense_retrieval_grailqa_train.jsonl"
sr_output_dev_path = f"{BASE_DIR}/schema_retriever/dense_retrieval_grailqa_dev.jsonl"
sr_output_test_path = f"{BASE_DIR}/schema_retriever/dense_retrieval_grailqa_test.jsonl"

sketch_output_train_nk_path = f"{BASE_DIR}/sketch_generation/run_final_none/sketch_preds_train.jsonl"
sketch_output_dev_path = f"{BASE_DIR}/sketch_generation/run_final_none/sketch_preds_dev.jsonl"
sketch_output_test_path = f"{BASE_DIR}/sketch_generation/run_final_none/sketch_preds_test.jsonl"

el_output_dev_el_path = f"{BASE_DIR}/entity_linker/tiara_dev_el_results.json"
el_output_test_el_path = f"{BASE_DIR}/entity_linker/tiara_test_el_results.json"
el_output_train_nk_el_path = ""

lfi_output_train_path = f"{BASE_DIR}/lf_integrator/run_final_none/lfs_cache_train_none.json"
lfi_output_dev_path = f"{BASE_DIR}/lf_integrator/run_final_none/lfs_cache_dev_none.json"
lfi_output_test_path = f"{BASE_DIR}/lf_integrator/run_final_none/lfs_cache_test_none.json"

