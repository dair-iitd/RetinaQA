import sys
import os


sys.path.append('.')
sys.path.append('..')

import re
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm


from configs.common_config import freebase_addr, freebase_port
from configs.discriminator_config import *
from common.semantic_matcher import matcher




class Data:
    def __init__(self, data_path, ranked_lf_path, lf_candidates_path, split="train", data_type="A"):
        self.data = {}
        # self.linked_entities = {}
        self.ranked_lf = {}
        self.lf_candidates = {}
        self.lf_candidates_path = lf_candidates_path

        self.split = split

        # prepare data dict
        with open(data_path, 'r', encoding='UTF-8') as fread:
            f = json.load(fread)
            self.data = {}
            for d in f:
                # gt_s_exp = self.get_gold_sexpression_by_qid(d["QuestionId"])
                # if gt_s_exp[0] == "null":
                #     continue
                self.data.update({str(d["QuestionId"]) : d})

                
            self.len = len(self.data)
            # print(self.data.keys())
            print("length of data : ",self.len)


        # prepare ranked lf data
        with open(ranked_lf_path, 'r', encoding='UTF-8') as fread:
            f = json.load(fread)

            for d in f:
                candidate_lfs = [lf["logical_form"] for lf in f[d]["candidates"]]
                self.ranked_lf.update({str(d) : candidate_lfs})
        
        # prepare entity linking data
        # if os.path.exists(entity_linking_path):
        #     with open(entity_linking_path, 'r', encoding='UTF-8') as fread:
        #         f = json.load(fread)
        #         for d in f:
        #             el = [k for k,v in f[d]["entities"].items()][:]
        #             self.linked_entities.update({str(d) : el})
        # else:
        #     print("Path does not exist : ",entity_linking_path)

        # get generated_sexps data:
        if os.path.exists(lf_candidates_path):
            with open(lf_candidates_path, "r") as fread:
                self.lf_candidates = json.load(fread)
        else:
            print("Path does not exist : ",lf_candidates_path)
            print("Please prepare data first !!!")
            exit()


    # def get_gold_entities_by_qid(self, qid):
    #     gold_ents = []
    #     for n in self.data[str(qid)]["graph_query"]["nodes"]:
    #         if n["node_type"] == "entity":
    #             gold_ents.append(n["id"])
    #     return gold_ents

    def get_question_by_qid(self, qid):
        return self.data[str(qid)]["ProcessedQuestion"]

    def get_gold_sexpression_by_qid(self, qid):
        gold_sexps = []
        for parse in self.data[str(qid)]["Parses"]:
            gold_sexps.append(parse["SExpr"])            
        return gold_sexps[:]

    # def get_entities_by_qid(self, qid):
    #     return self.linked_entities[str(qid)]

    def get_ranked_lf_by_qid(self, qid):
        return self.ranked_lf.get(str(qid),[])


class DiscriminatorDataset:
    def __init__(self, split="train", data_type="AU"):
        self.split = split

        # print(self.tokenizer.all_special_tokens) # --> ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        # print(self.tokenizer.all_special_ids)
        # print(self.tokenizer.eos_token_id)

        if split=="train":
            print("preparing ranker data for train")
            self.data_class = Data(data_path=data_train_path, \
                                            ranked_lf_path=lfr_output_train_path, \
                                            lf_candidates_path=lfi_output_train_path,\
                                            split="train",data_type=data_type)

        elif split=="dev":                
            print("preparing ranker data for dev")         
            self.data_class = Data(data_path=data_dev_path,\
                                            ranked_lf_path=lfr_output_dev_path, \
                                            lf_candidates_path=lfi_output_dev_path,\
                                            split="dev",data_type=data_type)

        elif split=="test":
            self.data_class = Data(data_path=data_test_path, \
                                            ranked_lf_path=lfr_output_test_path, \
                                            lf_candidates_path=lfi_output_test_path,\
                                            split="test",data_type=data_type)
        
        self.compute_metrics(self.data_class, split)


    def compute_metrics(self, data_class, split):

        valid_sexps_len = {0:0, 5:0, 10:0,20:0, 50:0, 100:0, 200:0, 300:0, 400:0, 500:0, 700:0, 1000:0, 2000:0, 5000:0, 10000:0, 50000:0, 100000:0}
        total_hits = 0
        total_count = 0
        
        for qid_idx, qid in tqdm(enumerate(data_class.data)):
            
            question = data_class.get_question_by_qid(qid)

            gold_sexp= ""
            # if split != "test":
            gold_sexp = data_class.get_gold_sexpression_by_qid(qid)[0]
            
            #AU training
            if gold_sexp in ["no logical form","NK", "none","null"]:
                continue 

            
            if data_class.lf_candidates.get(str(qid), {}) != {}:
                valid_sexps = data_class.lf_candidates[str(qid)]["valid_sexps"][:]
                invalid_sexps = data_class.lf_candidates[str(qid)]["invalid_sexps"][:]
                correct_sexps = []#data_class.lf_candidates[str(qid)]["correct_sexps"]
            else:
                print(qid, "not in cache")
                continue

            if ablation_type == "minus_SP_SR":
                valid_sexps = []

            ranked_lf = []
            if ablation_type != "minus_LFR":
                ranked_lf = data_class.get_ranked_lf_by_qid(qid)
            valid_sexps.extend(ranked_lf)
            
            for key in valid_sexps_len:
                if len(valid_sexps) <= key:
                    valid_sexps_len[key] += 1
                    break
            
            valid_sexps = list(set(valid_sexps))

            
            total_count += 1
            for lf in valid_sexps:
                em = matcher.same_logical_form(gold_sexp, lf)
                if em:
                    total_hits += 1
                    break
        
        print("Size: ",total_count)
        print("Avg Hits : ", (total_hits/total_count))
        print(valid_sexps_len)

  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, required=True)
    # if data contains both A and U questions then mark it AU else A.
    parser.add_argument('--data_type', type=str, default="A") 

    args = parser.parse_args()
    train_dataset = DiscriminatorDataset(split=args.split, data_type=args.data_type)


       
