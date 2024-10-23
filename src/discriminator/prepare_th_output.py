import copy
import json


pred_dev_file_read_path = "../../logs/grailqability/au/run_1/test_wdout_th.json"
pred_dev_file_write_path = "../../logs/grailqability/au/run_1/test_wd_lf_th.json"
threshold = -0.7

with open(pred_dev_file_read_path,"r") as fread:
    data = json.load(fread)

updated_data = {}


for d in data:

        temp_d = copy.deepcopy(data[d])

        pred_sexp = data[d]["logical_form"]
        score = float(data[d].get("score","0.0"))
        
        # score = 0.0 if data[d].get("scores",["0.0"]) == [] else float(data[d].get("scores",["0.0"])[0])
        
        if score < threshold:
             temp_d["logical_form"] = "NK"
             temp_d["answer"] = []
        
        updated_data.update({d:temp_d})

with open(pred_dev_file_write_path,"w") as fwrite:
    json.dump(updated_data, fwrite)
