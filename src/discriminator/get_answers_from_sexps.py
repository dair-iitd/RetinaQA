import argparse
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--exec_check', type=str, default=False, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--discriminator_output_file_path', type=str, required=True)


    args = parser.parse_args()

    if not ".txt" in args.discriminator_output_file_path:
        print("discriminator_output_file_path must be a txt")
        exit()
    
    if not args.dataset in ["webqsp", "grailqa", "grailqability"]:
        print("dataset should be one of the following: webqsp, grailqa, grailqability")
        exit()
    
    if not args.exec_check.lower() in ["true", "false"]:
        print("exec_check should be either True or False")
        exit()
    
    dataset = args.dataset
    discriminator_output_file_path = args.discriminator_output_file_path
    final_output_file_path = discriminator_output_file_path.replace(".txt",".json")

    print("Executing for: ")
    print(f"Dataset : {dataset}")
    print(f"Exec Check : {args.exec_check}")
    print(f"Output file path : {final_output_file_path}")

    with open(discriminator_output_file_path,"r") as fread:
        data = fread.readlines()

    log_dict = {}
    for d in tqdm(data[:]):
        d = json.loads(d)

        ranked_predictions = d["top_predictions"]
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

                if args.exec_check.lower() == "true":
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

    with open(final_output_file_path,"w") as fwrite:
        json.dump(log_dict, fwrite)
