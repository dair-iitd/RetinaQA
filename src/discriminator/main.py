import logging
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

from transformers import AutoTokenizer, AdamW, AutoTokenizer, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader

from common.freebase_retriever import FreebaseRetriever
from t5_ranker_enc_dec import T5ForCandidateRanking
from configs.common_config import freebase_addr, freebase_port, reverse_properties_file_path
from configs.discriminator_config import *
from common.semantic_matcher import matcher
from utils.logging import Logger


import gc
torch.cuda.empty_cache()
gc.collect()

freebase_retriever = FreebaseRetriever(freebase_addr=freebase_addr, freebase_port=freebase_port)
entity_cache = {}

if os.path.exists(save_dir) and os.path.exists(log_dir):
    print(save_dir)
    print(log_dir)

# if os.path.exists(entity_to_name_cache):
#     with open(entity_to_name_cache,"r") as fread:
#         entity_cache = json.load(fread)
# print("len of entity cache : ", len(entity_cache))

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

set_seed(42)


def resolve_entity_ids(s_exp):
    global entity_cache
    m_ids = re.findall(r"\b((m|g)\.[0-9a-zA-Z\_]+)\b",s_exp)
    m_ids = [i[0] for i in m_ids][:]
    m_ids.sort(key=len, reverse=True)
    for m_id in m_ids:
        if not m_id in entity_cache:
            name = freebase_retriever.entity_name_by_mid(m_id)
            if isinstance(name, list):
                name = name[0]
            entity_cache.update({m_id:name})
            # name = ""
            # print(f"{m_id} not found")
            # continue
        else:
            name = entity_cache[m_id]
        if name != "":
            s_exp = s_exp.replace(m_id, name)
    return s_exp

class Data:
    def __init__(self, data_path, ranked_lf_path, lf_candidates_path, use_au_data, dataset, split="train",do_test=False):
        self.data = {}
        # self.linked_entities = {}
        self.ranked_lf = {}
        self.lf_candidates = {}
        self.lf_candidates_path = lf_candidates_path

        self.split = split
        self.dataset = dataset

        # prepare data dict
        with open(data_path, 'r', encoding='UTF-8') as fread:
            f = json.load(fread)
            # print(dataset)
            if dataset == "webqsp":
                for d in f[:]:
                    self.data.update({str(d["QuestionId"]) : d})

            elif self.dataset in ["grailqa", "grailqability"]:
                for d in f[:]:
                    #AU training
                    if do_test:
                        self.data.update({str(d["qid"]) : d})
                    
                    else:
                        if use_au_data:
                            if split == "train":
                                self.data.update({str(d["qid"]) : d})
                            elif split == "dev":
                                if d["s_expression"] != "NK":
                                    self.data.update({str(d["qid"]) : d})

                        else:
                            if d["qType"] == "A":# and len(self.data) <=10:# and d["qid"] in [2100065014000]:
                                self.data.update({str(d["qid"]) : d})
                
            self.len = len(self.data)
            # print(self.data.keys())
            print("length of data : ",self.len)


        # prepare ranked lf data
        with open(ranked_lf_path, 'r', encoding='UTF-8') as fread:
            f = json.load(fread)

            for d in f:
                candidate_lfs = [lf["logical_form"] for lf in f[d]["candidates"]]
                self.ranked_lf.update({str(d) : candidate_lfs})
        
        # get generated_sexps data:
        if os.path.exists(lf_candidates_path):
            with open(lf_candidates_path, "r") as fread:
                self.lf_candidates = json.load(fread)
        else:
            print("Path does not exist : ",lf_candidates_path)
            print("Please prepare data first !!!")
            exit()

    def get_question_by_qid(self, qid):
        if self.dataset == "webqsp":
            return self.data[str(qid)]["RawQuestion"]
        elif self.dataset in ["grailqa", "grailqability"]:
            return self.data[str(qid)]["question"]

    def get_gold_sexpression_by_qid(self, qid):
        if self.dataset == "webqsp":
            sexps_list = []
            for parse in self.data[str(qid)]["Parses"]:
                sexps_list.append(parse["SExpr"])
            return sexps_list
        elif self.dataset in ["grailqa", "grailqability"]:
            return self.data[str(qid)]["s_expression"]

 
    def get_ranked_lf_by_qid(self, qid):
        return self.ranked_lf.get(str(qid),[])


class DiscriminatorDataset:
    def __init__(self, tokenizer, use_au_data, num_neg_samples=None, split="train",do_test=False,dataset="grailqa"):
        self.split = split
        self.num_neg_samples = num_neg_samples
        self.tokenizer = tokenizer
        self.max_len = 0
        self.dataset = dataset

        # print(self.tokenizer.all_special_tokens) # --> ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        # print(self.tokenizer.all_special_ids)
        # print(self.tokenizer.eos_token_id)

        if split=="train":
            print("preparing ranker data for train")
            self.data_class = Data(data_path=data_train_path, \
                                            ranked_lf_path=lfr_output_train_path, \
                                            lf_candidates_path=lfi_output_train_path,\
                                            use_au_data=use_au_data,\
                                            split="train", do_test=do_test,dataset=dataset)

        elif split=="dev":                
            print("preparing ranker data for dev")         
            self.data_class = Data(data_path=data_dev_path,\
                                            ranked_lf_path=lfr_output_dev_path, \
                                            lf_candidates_path=lfi_output_dev_path,\
                                            use_au_data=use_au_data,\
                                            split="dev", do_test=do_test,dataset=dataset)

        elif split=="test":
            self.data_class = Data(data_path=data_test_path, \
                                            ranked_lf_path=lfr_output_test_path, \
                                            lf_candidates_path=lfi_output_test_path,\
                                            use_au_data=use_au_data,\
                                            split="test", do_test=do_test,dataset=dataset)
        
        self.rev_rels_cache = self.get_reverse_relation_cache()
        self.pos_neg_samples_dict = {}
        # split="dev"
        self.prepare_data(self.data_class, split, use_au_data, do_test)

        self.question_ids = [key for key in self.model_input_data]
        self.meta_data = {}


    def get_reverse_relation_cache(self):
        with open(reverse_properties_file_path,"r") as fread:
            rev_rels = fread.readlines()
        
        rev_rels_cache = {}
        for rel in rev_rels:
            rel_split = rel.replace("\n","").split("\t")
            rev_rels_cache.update({rel_split[0] : rel_split[1]})
            rev_rels_cache.update({rel_split[1] : rel_split[0]})
        
        return rev_rels_cache

    def get_pos_neg_samples_webqsp(self, qid, gold_sexp_list, all_sexps):

        pos_samples = set()
        neg_samples = set()

        for gold_sexp in gold_sexp_list:
            pos_samples.add(gold_sexp)

        for s_exp in set(all_sexps):
            is_pos = 0
            for gold_sexp in gold_sexp_list:
                if "http" in gold_sexp:
                    em = gold_sexp==s_exp
                else:
                    em = matcher.same_logical_form(gold_sexp, s_exp)
                if em:
                    is_pos = 1
                    break
            if is_pos==1:
                pos_samples.add(s_exp)
            else:
                neg_samples.add(s_exp)

            # if len(neg_samples) >= 20000:
            #     break

        self.pos_neg_samples_dict.update({qid : {"pos_samples" : list(pos_samples)[:], "neg_samples" : list(neg_samples)[:]}})
        return list(pos_samples), list(neg_samples)


    def get_pos_neg_samples(self, qid, gold_sexp, all_sexps):
        
        correct_sexps = set()
        pos_samples = set()
        neg_samples = set()

        pos_samples.add(gold_sexp)

        for s_exp in set(all_sexps):
            em = matcher.same_logical_form(gold_sexp, s_exp)
            if em:
                pos_samples.add(s_exp)
            else:
                neg_samples.add(s_exp)
            

        self.pos_neg_samples_dict.update({qid : {"pos_samples" : list(pos_samples)[:], "neg_samples" : list(neg_samples)[:]}})
        return list(pos_samples), list(neg_samples)


    def prepare_data(self, data_class, split, use_au_data, do_test=False):

        # self.model_input_data = {}
        valid_sexps_len = {0:0, 5:0, 10:0,20:0, 50:0, 100:0, 200:0, 300:0, 400:0, 500:0, 700:0, 1000:0, 2000:0, 5000:0, 10000:0, 50000:0, 100000:0}
        self.model_input_data = []
        questions_skipped = 0
        for qid_idx, qid in tqdm(enumerate(data_class.data)):
            
            # if qid_idx>= 64:
            #     break
            question = data_class.get_question_by_qid(qid)

            gold_sexp= ""
            if split != "test":
                gold_sexp = data_class.get_gold_sexpression_by_qid(qid)
            
            if self.dataset == "webqsp":
                if split == "test":
                    gold_sexp = [""]
                elif gold_sexp[0] in ["null",""] and do_test==False:
                    questions_skipped += 1
                    continue

            #AU training
            if use_au_data:
                if gold_sexp in ["no logical form","NK"]:
                    gold_sexp = "none"

            
            if data_class.lf_candidates.get(str(qid), {}) != {}:
                valid_sexps = data_class.lf_candidates[str(qid)]["valid_sexps"][:]
                invalid_sexps = data_class.lf_candidates[str(qid)]["invalid_sexps"][:]
                # correct_sexps = []#data_class.lf_candidates[str(qid)]["correct_sexps"]
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
            
            if use_au_data and (not do_test):
                valid_sexps.append("none")

            valid_sexps = list(set(valid_sexps))

            if split=="train":            
                
                if not qid in self.pos_neg_samples_dict:
                    if self.dataset in ["grailqa","grailqability"]:
                        pos_samples, neg_samples = self.get_pos_neg_samples(qid, gold_sexp, valid_sexps)
                    elif self.dataset in ["webqsp"]:
                        pos_samples, neg_samples = self.get_pos_neg_samples_webqsp(qid, gold_sexp_list=gold_sexp, all_sexps=valid_sexps)
                else:
                    pos_samples = self.pos_neg_samples_dict[qid]["pos_samples"]
                    neg_samples = self.pos_neg_samples_dict[qid]["neg_samples"]
                

                if ablation_type == "minus_LFI":
                    neg_samples.extend(invalid_sexps)

                if self.dataset in ["grailqa","grailqability"]:
                    pos_samples = [gold_sexp]
                elif self.dataset == "webqsp":
                    pos_samples = [gold_sexp[0]]
                
                for ps in pos_samples[:]:
                    sexps_list = [ps]
                    sexps_list.append("dummy")
                    mask = [0,1]

                    # if len(neg_samples) == 0:
                    #     sexps_list.extend(random.sample(invalid_sexps, self.num_neg_samples))
                    #     self.model_input_data.append({"question" : question, "logical_forms" : sexps_list[:], "correct_lf" : ps,\
                    #                           "correct_label" : 0, "qid": qid})

                    if len(neg_samples) == self.num_neg_samples:
                        sexps_list.extend(neg_samples)
                        mask.extend([0]*len(neg_samples))
                        self.model_input_data.append({"question" : question, "logical_forms" : sexps_list[:], "correct_lf" : ps,\
                                            "correct_label" : 0, "qid": qid, "mask" :  mask})

                    elif len(neg_samples) > self.num_neg_samples:
                        # print(qid, len(neg_samples), self.num_neg_samples)
                        # sexps_list.extend(random.sample(neg_samples, self.num_neg_samples))
                        # print("more num of neg samples :", qid, len(neg_samples), len(pos_samples))
                        for i in range(1 + (len(neg_samples) // (1*self.num_neg_samples))):
                            sexps_list = [ps]
                            sexps_list.append("dummy")
                            mask = [0,1]
                            sexps_list.extend(random.sample(neg_samples, self.num_neg_samples))
                            mask.extend([0]*self.num_neg_samples)
                            self.model_input_data.append({"question" : question, "logical_forms" : sexps_list[:], "correct_lf" : ps,\
                                              "correct_label" : 0, "qid": qid, "mask":mask})
                            if i >= 1:
                                break

                            # sexps_list.extend(random.sample(neg_samples, self.num_neg_samples))
                            # self.model_input_data.append({"question" : question, "logical_forms" : sexps_list[:], "correct_lf" : ps,\
                            #                     "correct_label" : 0, "qid": qid})
                        
                    elif len(neg_samples) < self.num_neg_samples:
                        sexps_list.extend(neg_samples)
                        mask.extend([0]*len(neg_samples))

                        for j in range(self.num_neg_samples-len(neg_samples)):
                            sexps_list.append("dummy")
                            mask.append(1)

                        # if (self.num_neg_samples-len(neg_samples)) > len(invalid_sexps):
                        #     sexps_list.extend(invalid_sexps)
                        #     mask.extend([0]*len(invalid_sexps))

                        #     for j in range(self.num_neg_samples-len(neg_samples)):
                        #         sexps_list.append("dummy")
                        #         mask.append(1)
                        # else:
                        #     sexps_list.extend(random.sample(invalid_sexps,self.num_neg_samples-len(neg_samples)))
                        #     mask.extend([0]*(self.num_neg_samples-len(neg_samples)))
  
                        self.model_input_data.append({"question" : question, "logical_forms" : sexps_list[:], "correct_lf" : ps,\
                                            "correct_label" : 0, "qid": qid, "mask": mask})
                    

                    assert len(sexps_list)==self.num_neg_samples+2, f"{len(sexps_list)}"
                    # self.model_input_data.append({"question" : question, "logical_forms" : sexps_list, "correct_lf" : ps,\
                    #                           "correct_label" : 0, "qid": qid})
                    
            else:
                if ablation_type == "minus_LFI": 
                    valid_sexps.extend(invalid_sexps)
                mask = [0]*len(valid_sexps)
                if self.dataset == "webqsp":
                    self.model_input_data.append({"question" : question, "logical_forms" : valid_sexps[:], "correct_lf" : gold_sexp[0],\
                                              "correct_label" : 0, "qid" : qid, "mask":mask})
                else:
                    self.model_input_data.append({"question" : question, "logical_forms" : valid_sexps[:], "correct_lf" : gold_sexp,\
                                                "correct_label" : 0, "qid" : qid, "mask":mask})


        random.shuffle(self.model_input_data)
        # print(self.model_input_data[0])
        print("len of model input data : ", len(self.model_input_data))
        print("valid_sexps_len dict : ", valid_sexps_len)
        print("questions_skipped : ",questions_skipped)

    def __len__(self):
        return len(self.model_input_data)

    def __getitem__(self, idx):
        question = self.model_input_data[idx]["question"]
        logical_forms = self.model_input_data[idx]["logical_forms"]
        label = self.model_input_data[idx]["correct_label"]  
        correct_lf =  self.model_input_data[idx]["correct_lf"]  
        qid = self.model_input_data[idx]["qid"]
        mask = self.model_input_data[idx]["mask"]

        encoded_tokenizer_input = []
        attention_mask = []
        decoded_logical_forms = ""
        for lf in logical_forms:
            resolved_lf = resolve_entity_ids(lf)
            # print("resolved_lf : ", resolved_lf)
            inp = "[CLS] " + question + " [SEP] " + resolved_lf
            encoded_inp = self.tokenizer(inp, return_tensors="pt", padding='max_length', max_length = 256, truncation=True) #512
            if len(encoded_inp["input_ids"][0]) > self.max_len:
                self.max_len = len(encoded_inp["input_ids"][0])
                print(f"len greater than than max len : {self.max_len}")
            encoded_tokenizer_input.append(encoded_inp["input_ids"][0])
            attention_mask.append(encoded_inp["attention_mask"][0])
            decoded_logical_forms = decoded_logical_forms + " [SEP] " + lf
        
        if len(encoded_tokenizer_input) == 0:
            encoded_tokenizer_input = torch.tensor(encoded_tokenizer_input)
            attention_mask = torch.tensor(attention_mask)
        else:
            encoded_tokenizer_input = torch.stack(encoded_tokenizer_input)
            attention_mask = torch.stack(attention_mask)

        mask = torch.tensor(mask)

        sample = {"model_input": encoded_tokenizer_input, "attention_mask": attention_mask,\
                  "label": torch.tensor(label),"correct_lf":correct_lf,\
                  "decoded_logical_forms":decoded_logical_forms, "qid" : qid, "question" : question, "mask" :  mask}

        return sample


def train(train_dataset, dev_dataset, model, gpus, use_au_data, target_id, args={}):

    dev_dataloader = DataLoader(dev_dataset, batch_size=1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

    best_em = 0

    t_total = args.num_train_epochs * (len(train_dataloader) // args.gradient_accumulation_steps)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(max(args.warmup_steps, t_total * args.warmup_ratio)),
        num_training_steps=t_total
    )

    training_logs.logger_obj.info(f"length of data loader : {len(train_dataloader)}")
    training_logs.logger_obj.info(f"optimisation steps : {t_total}")
    
    num_steps = 0
    best_epoch = -1
    for epoch in range(args.num_train_epochs):
        training_logs.logger_obj.info(f"Training for epoch {epoch}")

        train_dataset.prepare_data(train_dataset.data_class, "train", use_au_data)
        train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size)

        model.train()
        epoch_loss = 0
        for idx, batch in tqdm(enumerate(train_dataloader)):
            
            # print("training : ",idx)
            model_input = batch["model_input"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            mask = batch["mask"].to(device)

            loss, outputs = model({"labels" : labels, "attention_mask":attention_mask, "model_input" : model_input, "mask" : mask})
            # print("loss: " ,loss)

            if len(gpus) > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            loss.backward()
            epoch_loss += loss.item()

            if (num_steps + 1) % 100 == 0:
                training_logs.logger_obj.info(f"num steps {num_steps}: {epoch_loss/(idx+1)}")

            if (num_steps + 1) % args.gradient_accumulation_steps == 0:
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            num_steps += 1
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        if num_steps%save_every_step == 0:
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.t5_enc_dec.save_pretrained(f"{save_dir}/{str(epoch)}_steps")

        training_logs.logger_obj.info(f"epoch loss : {epoch_loss/len(train_dataloader)}")
        training_logs.logger_obj.info(f"{optimizer.param_groups[0]['lr']}")

        em = eval(dev_dataloader, model, device, log_file=f"{log_dir}/epoch_{epoch}.txt", target_id=target_id)
        if em > best_em:
            best_em = em
            best_epoch = epoch
            training_logs.logger_obj.info(f"best_em : {best_em}")
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.t5_enc_dec.save_pretrained(f"{save_dir}/best_model")
            # tokenizer.save_pretrained(f"{save_dir}/best_model")
            torch.save(model.state_dict(), f"{save_dir}/best_model.pt")

        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.t5_enc_dec.save_pretrained(f"{save_dir}/{str(epoch)}_epochs")

        training_logs.logger_obj.info(f"EM : {em}, Best EM : {best_em}")
        if (epoch - best_epoch) >= args.patience:
            training_logs.logger_obj.info(f"Best model not changed since {args.patience} epochs")
            exit()

    
def get_metrics(gold_sexp, ranked_predictions):
    # returns em_1, em_3, em_5

    for i,predicted_sexp in enumerate(ranked_predictions[:5]):
        em = matcher.same_logical_form(gold_sexp, predicted_sexp)
        if em and i==0:
            return 1,1,1
        elif em and i<=2:
            return 0,1,1
        elif em and i<=4:
            return 0,0,1
    return 0,0,0


def eval(eval_dataloader, model, device, log_file, target_id):
    
    model.eval()
    print("Evaluating...")

    fwrite = open(log_file,"w")
    logs = []
    em_1 = 0
    em_3 = 0
    em_5 = 0
    total_samples = 0
    print(len(eval_dataloader))

    for batch_idx, batch in tqdm(enumerate(eval_dataloader)):


        model_input = batch["model_input"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        correct_lf = batch["correct_lf"]
        decoded_logical_forms = batch["decoded_logical_forms"]
        qid = batch["qid"]
        question = batch["question"]
        mask = batch["mask"].to(device)

        if model_input.numel() == 0:# or model_input.shape[1] > 5000:
            # print("Writing empty logical forn : ",print(model_input.shape[1]))
            logs.append({"qid":qid[0], "question":question[0], "gold_sexpr": correct_lf, "top_predictions":[""],"scores":[]})
            json.dump({"qid":qid[0], "question":question[0], "gold_sexpr": correct_lf, "top_predictions":[""],"scores":[]}, fwrite)
            fwrite.write("\n")
            total_samples += 1
            em_1 += 0
            em_3 += 0
            em_5 += 0
            continue

        all_outputs = []
        sub_batch_eval=256
        with torch.no_grad():
            for i in range(0, model_input.shape[1], sub_batch_eval):
                # print(f"batch-index : {batch_idx}-{i}")
                input_ids = model_input[:,i:i+sub_batch_eval,:]#.squeeze(0)
                att_mask = attention_mask[:,i:i+sub_batch_eval,:]#.squeeze(0)
                batch_mask = mask[:,i:i+sub_batch_eval]
                decoder_input_ids = torch.zeros(len(input_ids), 1).to(device).int()
                loss, outputs = model({"model_input":input_ids, "attention_mask":att_mask, "mask":batch_mask})
                all_outputs.extend(outputs["logits"][:, 0, target_id].cpu().tolist())

        # all_outputs = torch.cat(all_outputs)
        # all_outputs = all_outputs.unsqueeze(dim=0)
        # print(len(all_outputs))
        # output_scores = np.array(outputs[0].detach().tolist())
        # print(len(all_outputs))
        output_scores = np.array([all_outputs])
        num_samples = model_input.shape[1]

        for idx,sample in enumerate(output_scores):
            total_samples += 1

            scores = np.sort(sample)[::-1]
            ranked_lf_indexes = np.argsort(sample)[::-1]
            
            decoded_lfs = []
            for lf in decoded_logical_forms[idx].split(" [SEP] "):
                if lf.strip() != "":
                    decoded_lfs.append(lf.strip())

            if len(decoded_lfs) == 0:
                logs.append({"qid":qid[idx], "question":question[idx], "gold_sexpr": correct_lf[idx], "top_predictions":[""],"scores":[]})
                json.dump({"qid":qid[idx], "question":question[idx], "gold_sexpr": correct_lf[idx], "top_predictions":[""],"scores":[]}, fwrite)
                fwrite.write("\n")
                total_samples += 1
                em_1 += 0
                em_3 += 0
                em_5 += 0
                continue

            # print(decoded_logical_forms)
            # print(ranked_lf_indexes)
            # print(output_scores)
            # print(qid[idx])
            # print("-"*10)
            # print(self.tokenizer.decode(model_input[0]))
            
            # print("-"*10)
            # print("decoded lf : ", decoded_lfs[0])
            # print("correct_lf : ",correct_lf[idx])
            
            ranked_lfs = np.array(decoded_lfs)[ranked_lf_indexes]
            scores = [float(a) for a in scores][:]
            logs.append({"qid":qid[idx], "question":question[idx], "gold_sexpr": correct_lf[idx], "top_predictions":ranked_lfs[:20],"scores":scores[:20]})
            json.dump({"qid":qid[idx], "question":question[idx], "gold_sexpr": correct_lf[idx], "top_predictions":list(ranked_lfs[:20]),"scores":list(scores[:20])}, fwrite)
            fwrite.write("\n")
            # print("ranked_lfs : ", ranked_lfs[0])
            # print(list(ranked_lfs[:5]),"\nscores",list(scores[:5]))
            em_k = get_metrics(correct_lf[idx], ranked_lfs[:5])
                
            em_1 += em_k[0]
            em_3 += em_k[1]
            em_5 += em_k[2]
        
    print("total_samples : ",total_samples)
    print(f"em_1 : {round(em_1/total_samples,4)}, em_3 : {round(em_3/total_samples,4)},\
            em_5 : {round(em_5/total_samples,4)}")
    fwrite.close()
    return round(em_1/total_samples,4)
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--use_au_data', action='store_true', default=False)
    parser.add_argument('--train_batch_size', type=int, default=4, required=False)
    parser.add_argument('--learning_rate', type=float, default=1e-4, required=False)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, required=False)
    parser.add_argument('--num_neg_samples', type=int, default=64, required=False)
    parser.add_argument('--num_train_epochs', type=int, required=False, default=10)
    parser.add_argument('--patience', type=int, required=False, default=3)
    parser.add_argument('--warmup_steps', type=int, required=False, default=0)
    parser.add_argument('--warmup_ratio', type=float, required=False, default=0.01)
    parser.add_argument('--gpu_ids', type=int, nargs="+", required=False)
    parser.add_argument('--dataset', type=str, default="grailqa", required=True)
    parser.add_argument('--do_test', action='store_true', default=False)
    parser.add_argument('--test_split', type=str, default="test")
    parser.add_argument('--saved_model_path', type=str, required=False)
    parser.add_argument('--test_logs_file', type=str, required=False)


    args = parser.parse_args()

    if args.gpu_ids:
        gpus = sorted(list(args.gpu_ids))
    else:
        gpus = [0]

    assert log_dir is not None and os.path.isdir(log_dir), f"Please create log directory : {log_dir}"
    assert save_dir is not None and os.path.isdir(save_dir), f"Please create model directory : {save_dir}"

    if args.do_test:
        print("Testing...")
        print(f'{args.saved_model_path}/../tokenizer')    
        tokenizer = AutoTokenizer.from_pretrained(f'{args.saved_model_path}/../tokenizer')
        target = "<extra_id_6>"
        target_id = tokenizer(target).input_ids
        assert len(target_id) == 2
        target_id = target_id[0]
        # config = AutoConfig.from_pretrained("t5-base")
        device = "cuda:0"# if torch.cuda.is_available() else "cpu"

        print("Loading model from : ", args.saved_model_path)
        ranker_model = T5ForCandidateRanking(args.saved_model_path, target_id=target_id)
        ranker_model = ranker_model.to(device)
        test_dataset = DiscriminatorDataset(tokenizer, use_au_data=True, split=args.test_split,do_test=True, dataset=args.dataset)
        # during inference only batch size = 1 is supported....
        test_dataloader = DataLoader(test_dataset, batch_size=1)

        print("Performing Evaluation...")
        em = eval(test_dataloader, ranker_model, device, log_file=args.test_logs_file, target_id = target_id)
        print("Testing EM : ", em)
        exit()


    if args.do_train:
        if os.path.exists(f"{save_dir}/best_model"):
            print("Model already exist")
            exit()
        
        tokenizer = AutoTokenizer.from_pretrained(f'{args.saved_model_path}',\
                                               local_files_only=False) #cache_dir = "~/.cache/huggingface/transformers/",
        special_tokens_dict = {'additional_special_tokens': ['[CLS]','[SEP]']}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        tokenizer.save_pretrained(f"{save_dir}/tokenizer")

        target = "<extra_id_6>"
        target_id = tokenizer(target).input_ids
        assert len(target_id) == 2
        target_id = target_id[0]

        training_logs = Logger(save_dir, "training")
        training_logs.logger_obj.setLevel(logging.INFO)

        training_logs.logger_obj.info("Training with following config")
        training_logs.logger_obj.info(f"batch size  : {args.train_batch_size}")
        training_logs.logger_obj.info(f"num neg samples : {args.num_neg_samples}")
        training_logs.logger_obj.info(f"num epochs : {args.num_train_epochs}")
        training_logs.logger_obj.info(f"gradient_accumulation_steps : {args.gradient_accumulation_steps}")
        training_logs.logger_obj.info(f"save_dir : {save_dir}")
        training_logs.logger_obj.info(f"log_dir : {log_dir}")
        training_logs.logger_obj.info(f"learning rate : {args.learning_rate}")
        training_logs.logger_obj.info(f"Use AU data : {args.use_au_data}")
        training_logs.logger_obj.info(f"GPU ids : {gpus}")

        dev_dataset = DiscriminatorDataset(tokenizer, use_au_data=args.use_au_data, split="dev", dataset=args.dataset)
        train_dataset = DiscriminatorDataset(tokenizer, use_au_data=args.use_au_data, split="train",num_neg_samples=args.num_neg_samples, dataset=args.dataset)

        ranker_model = T5ForCandidateRanking(args.saved_model_path, target_id=target_id)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # print(device)

        if len(gpus)>1:
            ranker_model = torch.nn.DataParallel(ranker_model, device_ids=gpus)
            # device = f"cuda:{gpus[1]}" if torch.cuda.is_available() else "cpu"

        ranker_model.to(device)
        
        train(train_dataset, dev_dataset, ranker_model, gpus, args.use_au_data, target_id, args)
