import json
import sys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sys.path.append('.')
sys.path.append('..')
from configs.sketch_generation_config import sketch_data_train_path, sketch_data_dev_path, sketch_data_test_path, output_log_dir, model_save_dir


with open(f"{output_log_dir}/sketch_preds_test.jsonl","r") as fread:
    data = fread.readlines()

em = 0
count = 0
for d in data:
    d = json.loads(d)
    gs = d["golden_sketch"]
    if gs in ["NK", "none"]:
        continue
    ps = d["predicted_sketch"]
    gs = gs.replace("(R relation)","relation")
    ps = ps.replace("(R relation)","relation")
    em += gs == ps
    count += 1

print(count, em/count)
