import sys
import os
from tqdm import tqdm
import json
import numpy as np

sys.path.append('.')
sys.path.append('..')

from configs.common_config import freebase_addr, freebase_port, reverse_properties_file_path
from common.freebase_retriever import FreebaseRetriever
from common.logic_form_util import lisp_to_sparql
from utils.s_expr_util import execute_s_expr

retriever = FreebaseRetriever(freebase_addr=freebase_addr, freebase_port=freebase_port)
EXEC_CHECK = True
dataset = "webqsp" #["grailqa", "grailqability"]

base_path = "~/RETINAQA/logs/webqsp/discriminator/run_best_none/"
with open(f"{base_path}/test_best_epoch.txt","r") as fread:
    data = fread.readlines()

log_dict = {}
for d in tqdm(data[:]):
    d = json.loads(d)

    ranked_predictions = d["top_predictions"]
    # ranked_predictions = [d["predicted_s_expr"]]
    # gold_sexp = d["gold_sexpr"]
    scores = list(np.array(d["scores"]))

    predicted_sparql = ""
    predicted_ans = []
    for i,pred_sexpr in enumerate(ranked_predictions):
        try:
            if pred_sexpr in ["NK","no logical form","","none"]:
                log_dict.update({str(d["qid"]) : {"logical_form" : "NK", "answer" : [],"score" : 0}})
                break

            if dataset == "webqsp":
                predicted_ans = execute_s_expr(pred_sexpr)[1]
            else:
                predicted_sparql = lisp_to_sparql(pred_sexpr)
                predicted_ans = retriever.query_var(predicted_sparql, 'x')

            # print(predicted_sparql)
            # print(predicted_ans)
            if EXEC_CHECK:
                if len(predicted_ans) == 0 or predicted_ans[0] in [0, '0']:
                    continue
                else:
                    log_dict.update({str(d["qid"]) : {"logical_form" : pred_sexpr, "answer" : predicted_ans,"score" : scores[i]}})
                    break
            else:
                log_dict.update({str(d["qid"]) : {"logical_form" : pred_sexpr, "answer" : predicted_ans,"score" : scores[i]}})
                break

        except Exception as e:
            print("execption : ", d["qid"], pred_sexpr)
            continue

    if predicted_sparql == "": #predicted_ans == [] or predicted_ans[0] in [0, '0']:
        print("empty answer")
        log_dict.update({str(d["qid"]) : {"logical_form" : "NK", "answer" : [],"score" : 0}})
    
print("output len : ",len(log_dict))

with open(f"{base_path}/test_best_epoch.json","w") as fwrite:
    json.dump(log_dict, fwrite)
