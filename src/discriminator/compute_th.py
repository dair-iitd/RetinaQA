import sys
sys.path.append('.')
sys.path.append('..')

import json
import numpy as np
from tqdm import tqdm
from common.semantic_matcher import matcher


th_range_start = -10
th_range_end = 10
interval = 1
gt_dev_file_path = "~/RETINAQA/data/grailqability/dataset/grailqability_v1.0_dev_a_na_reference.json"
pred_dev_file_path = "~/RETINAQA/logs/grailqability/a/discriminator/run_1_none/dev_best_model.json"

with open(gt_dev_file_path,"r") as fread:
    data = json.load(fread)
gt_dev_data = {str(d["qid"]) :  d for d in data}

with open(pred_dev_file_path,"r") as fread:
    data = json.load(fread)

for th in np.arange(th_range_start,th_range_end,interval):
    print("-"*20)
    print(f"Computing metrics for threshold : {th}")
    na_em = 0

    total_em = 0
    a_em = 0
    a_count = 0
    na_count = 0
    total_count = 0
    one_len_count = 0

    for d in data:

        qType = gt_dev_data[d]["qType"]
        pred_sexp = data[d]["logical_form"]
        gold_sexp = gt_dev_data[d]["s_expression"]
        score = float(data[d]["score"])
        if gold_sexp in ["no logical form","", "NK"]:
            gold_sexp = "none"
        
        if pred_sexp in ["no logical form","", "NK"] or score < th:
            pred_sexp = "none"

        if qType == "U":
            if gold_sexp in ["no logical form","none"]:
                em = pred_sexp==gold_sexp
            else:
                em = matcher.same_logical_form(gold_sexp, pred_sexp)
            na_em += em
            total_em += em
            na_count += 1
        else:
            em = matcher.same_logical_form(gold_sexp, pred_sexp)
            a_em += em
            total_em += em
            a_count += 1

    print(f"a_em : {a_em/a_count}, na_em : {na_em/na_count}, total_em : {total_em/(a_count+na_count)}")

                                                                                    