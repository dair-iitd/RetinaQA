
import sys
import os
sys.path.append('.')
sys.path.append('..')

from common.freebase_retriever import FreebaseRetriever
from configs.common_config import freebase_addr, freebase_port, reverse_properties_file_path
from configs.discriminator_config import *
from common.semantic_matcher import matcher
import json
from tqdm import tqdm
import re
from joblib import Parallel, delayed

dataset = "webqsp" #["grailqa","grailqability"]
PATH = "/Users/prayushifaldu/IITD/git/RETINAQA/data/webqsp/entity_to_name_cache.json"

freebase_retriever = FreebaseRetriever(freebase_addr=freebase_addr, freebase_port=freebase_port)

sexp_mid_cache = {}

global entity_cache

entity_cache = {}
with open(PATH,"r") as fread:
    entity_cache = json.load(fread)

def update_cache():
    with open(PATH,"r") as fread:
        curr_entity_cache = json.load(fread)
    
    entity_cache.update(curr_entity_cache)
    print("Updating cache... ", len(curr_entity_cache), len(entity_cache))
    
    if len(curr_entity_cache) == len(entity_cache):
        return

    with open(PATH,"w") as fwrite:
        json.dump(entity_cache, fwrite)

def resolve_entity_ids(s_exp):
    if not s_exp in sexp_mid_cache:
        m_ids = re.findall(r"\b((m|g)\.[0-9a-zA-Z\_]+)\b",s_exp)
        m_ids = [i[0] for i in m_ids][:]
        m_ids.sort(key=len, reverse=True)
        sexp_mid_cache.update({s_exp : m_ids[:]})
    else:
        m_ids = sexp_mid_cache[s_exp]
    results = []
    for m_id in m_ids:
        if not m_id in entity_cache:
            name = freebase_retriever.entity_name_by_mid(m_id)
            if isinstance(name, list):
                name = name[0]
            entity_cache.update({m_id:name})
            # print(entity_cache)
        else:
            continue
        results.append((m_id, name))
    return results

def get_gold_sexpression(d):
    gold_sexps = []
    for parse in d["Parses"]:
        gold_sexps.append(parse["SExpr"])            
    return gold_sexps[:]

def prepare_cache(data_path, ranked_lf_path, lf_candidates_path):

    with open(data_path, 'r', encoding='UTF-8') as fread:
        f = json.load(fread)
        
        # results = Parallel(n_jobs=512)(delayed(resolve_entity_ids)(d["s_expression"]) for d in tqdm(f[:]))
        # for r in results:
        #     if r != []:
        #         entity_cache.update({r[0][0]:r[0][1]})

        for i,d in enumerate(tqdm(f[:])):
            if dataset == "webqsp":
                for s_exp in get_gold_sexpression(d):
                    results = resolve_entity_ids(s_exp)
            else:
                results = resolve_entity_ids(d["s_expression"])
            if i%1000==0:
                update_cache()

    update_cache()
    with open(ranked_lf_path, 'r', encoding='UTF-8') as fread:
        f = json.load(fread)

        for i,d in enumerate(tqdm(f)):
            candidate_lfs = [lf["logical_form"] for lf in f[d]["candidates"]]
            # results = Parallel(n_jobs=128)(delayed(resolve_entity_ids)(lf) for lf in candidate_lfs)

            # for r in results:
            #     if r != []:
            #         entity_cache.update({r[0][0]:r[0][1]})

            for lf in candidate_lfs:
                resolved_lf = resolve_entity_ids(lf)
            
            if i%1000==0:
                update_cache()
    
    update_cache()
    print("Loading lf candidates")
    with open(lf_candidates_path, "r") as fread:
        lf_candidates = json.load(fread)
    
        for i,qid in enumerate(tqdm(lf_candidates)):
            valid_sexps = lf_candidates[qid]["valid_sexps"]
            invalid_sexps = lf_candidates[qid]["invalid_sexps"]

            # results = Parallel(n_jobs=128)(delayed(resolve_entity_ids)(lf) for lf in valid_sexps)
            # for r in results:
            #     if r != []:
            #         entity_cache.update({r[0][0]:r[0][1]})

            # results = Parallel(n_jobs=128)(delayed(resolve_entity_ids)(lf) for lf in invalid_sexps)
            # for r in results:
            #     if r != []:
            #         entity_cache.update({r[0][0]:r[0][1]})
            
            for lf in valid_sexps:
                resolved_lf = resolve_entity_ids(lf)
            for lf in invalid_sexps:
                resolved_lf = resolve_entity_ids(lf)

            if i%1000==0:
                update_cache()


print("Len of cache : ",len(entity_cache))
print("preparing ranker data for train")
data_class = prepare_cache(data_path=data_train_path, \
                        ranked_lf_path=lfr_output_train_path, \
                        lf_candidates_path=lfi_output_train_path)

update_cache()

print("Len of cache : ",len(entity_cache))
print("preparing ranker data for dev")         
data_class = prepare_cache(data_path=data_dev_path,\
                        ranked_lf_path=lfr_output_dev_path, \
                        lf_candidates_path=lfi_output_dev_path,)

update_cache()

print("Len of cache : ",len(entity_cache))
print("preparing ranker data for test")    
data_class = prepare_cache(data_path=data_test_path, \
                        ranked_lf_path=lfr_output_test_path, \
                        lf_candidates_path=lfi_output_test_path)
print("Len of cache : ",len(entity_cache))

update_cache()