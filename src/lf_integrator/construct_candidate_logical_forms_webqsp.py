import sys
sys.path.append('.')
sys.path.append('..')

import re
import os
import json
import argparse
from tqdm import tqdm
from common.semantic_matcher import matcher, relation_dr, upper_types
from logical_form_validator import LogicalFormValidator
from configs.lf_integrator_config import *
from configs.common_config import *
import itertools
import copy

from pathlib import Path
current_file_dir = Path(__file__).parent

VALIDITY_CACHE = {}

class Data:
    def __init__(self, data_path, schema_path, entity_linking_path, sketch_path, lf_candidates_path, split="train"):
        self.data = {}
        self.linked_schema = {}
        self.linked_entities = {}
        self.sketch = {}
        self.lf_candidates = {} # either empty json file or contains partially generated candidates (used as checkpoint)
        self.lf_candidates_path = lf_candidates_path

        self.all_rels = []
        self.all_types = []

        self.split = split

        # prepare data dict
        with open(data_path, 'r', encoding='UTF-8') as fread:
            f = json.load(fread)
            for d in f:
                self.data.update({str(d["QuestionId"]) : d})
                
            self.len = len(self.data)
            print("length of data : ",self.len)

        # prepare linked schema dict
        with open(schema_path, 'r') as fread:
            f = fread.readlines()

            for d in f:
                d = json.loads(d)
                self.linked_schema.update({str(d["qid"]) : {"classes" : d["classes"], "relations" : d["relations"]}})

        
        # prepare sketch parser data
        if os.path.exists(sketch_path):
            with open(sketch_path,"r") as fread:
                f = fread.readlines()
                for i,d in enumerate(f):
                    # print(i,type(d))
                    d = json.loads(d)
                    # self.sketch.update({str(d["qid"]) : d["predicted_s_expr"]})
                    self.sketch.update({str(d["qid"]) : d["top_predictions"]})
        else:
            print("Path does not exist : ", sketch_path)
        
        print(len(self.sketch))

        # prepare entity linking data
        if os.path.exists(entity_linking_path):
            with open(entity_linking_path, 'r', encoding='UTF-8') as fread:
                f = json.load(fread)
                for d in f:
                    el = [k for k,v in f[d]["entities"].items()][:]
                    self.linked_entities.update({str(d) : el})
        else:
            print("Path does not exist : ", entity_linking_path)

        # get domain dict
        with open(fb_roles_file_path,"r") as fread:
            fb_roles_data = fread.readlines()
            for d in fb_roles_data:
                d_split = d.split(" ")
                d_0 = d_split[0].replace("\n","").strip()
                d_1 = d_split[1].replace("\n","").strip()
                d_2 = d_split[2].replace("\n","").strip()

                self.all_rels.append(d_1)
                self.all_types.append(d_0)
                self.all_types.append(d_2)
                self.all_rels.append(f'{d_0}.time_macro')
                self.all_rels.append(f'{d_2}.time_macro')

        with open(domain_dict_file_path,"r") as fread:
            schema_items_dict = json.load(fread)
            for k,v in schema_items_dict.items():
                domain_len = len(k.split("."))
                for si in v:
                    if len(si.split("."))-domain_len == 1:
                        if not si in self.all_types:
                            self.all_types.append(si)
                            self.all_rels.append(f"{si}.time_macro")
                    elif len(si.split("."))-domain_len == 2:
                        if not si in self.all_rels:
                            self.all_rels.append(si)

        self.all_types.sort(key = len, reverse=True)
        self.all_rels.sort(key = len, reverse=True)

        # if len(si.split(".")) == 2:
        #     etypes.append(si)
        # elif len(si.split(".")) == 3:
        #     rels.append(si)



        # get generated_sexps data if any:
        if os.path.exists(lf_candidates_path):
            with open(lf_candidates_path, "r") as fread:
                self.lf_candidates = json.load(fread)

    def update_lf_candidates_cache(self, data, write_path=""):
        if write_path == "":
            write_path = self.lf_candidates_path
        with open(write_path, "w") as fwrite:
            json.dump(data,fwrite)
        fwrite.close()

    def get_gold_entities_by_qid(self, qid):
        gold_ents = []
        for parse in self.data[str(qid)]["Parses"]:
            gold_ents.append(parse["TopicEntityMid"])
        return list(set(gold_ents))

    def get_question_by_qid(self, qid):
        return self.data[str(qid)]["RawQuestion"]

    def get_gold_sexpression_by_qid(self, qid):
        gold_sexps = []
        for parse in self.data[str(qid)]["Parses"]:
            gold_sexps.append(parse["SExpr"])            
        return gold_sexps[:]

    def get_gold_sketch_by_qid(self, qid):

        s_exp = self.get_gold_sexpression_by_qid(qid)[0] #self.data[str(qid)]["s_expression"]
        s_exp = s_exp.replace("("," ( ").replace(")"," ) ")
        s_exp = re.sub(r"\b(m|g)\.[0-9a-zA-Z\_]+\b","entity",s_exp)
        s_exp_split = s_exp.split(" ")

        sketch_split = []
        for token in s_exp_split:
            if token in self.all_rels:
                sketch_split.append("relation")
            elif token in self.all_types:
                sketch_split.append("type")
            else:
                sketch_split.append(token)

        sketch = " ".join(sketch_split)
        sketch = sketch.replace(" ( ","(").replace(" ) ",")")
        # sketch = sketch.replace("(R relation)","relation")
        return sketch

    def get_entities_by_qid(self, qid):
        return self.linked_entities.get(str(qid),[])

    def get_linked_schema_by_qid(self, qid):
        return (self.linked_schema[str(qid)]["classes"][:10], self.linked_schema[str(qid)]["relations"][:10])

    def get_predicted_sketch_by_qid(self, qid, top_k=1):
        # assert top_k==1, "top_k = 1 is only supported."
        top_sketches = self.sketch[str(qid)][:top_k]
        return top_sketches

def flatten(test_list):
    if isinstance(test_list, list):
        temp = []
        for ele in test_list:
            temp.extend(flatten(ele))
        return temp
    else:
        return [test_list]

def lisp_to_nested_expression(lisp_string: str):

    stack = []
    current_expression = []
    tokens = lisp_string.split()

    assert lisp_string.count('(') == lisp_string.count(')'), "unbalanced sexp"

    # print(tokens, len(tokens))
    for token in tokens:
        while token[0] == '(':
            nested_expression = []
            current_expression.append(nested_expression)
            stack.append(current_expression)
            current_expression = nested_expression
            token = token[1:]
        current_expression.append(token.replace(')', ''))
        while token[-1] == ')':
            current_expression = stack.pop()
            token = token[:-1]

    # print(tokens, current_expression[0])
    # print(len(tokens), len(flatten(current_expression[0])))
    assert len(tokens) == len(flatten(current_expression[0])), "graph_utils.py : invalid s_exp (syntax)"
    return current_expression[0]


def get_all_sexps(sketches, rels, types, ents, logical_form_validator, gold_sexp_list):
    # print("entered 1")
    rels = list(set(rels))
    types = list(set(types))
    ents = list(set(ents))

    
    all_perms = []
    valid_perms = []
    correct_lfs = []
    for sketch in sketches:
        num_ents = sketch.count("entity")
        num_rels = sketch.count("relation")
        num_types = sketch.count("type")

        # ents_perm = set(itertools.permutations(ents, num_ents))
        # rels_perm = list(itertools.permutations(rels, num_rels))
        # types_perm = list(itertools.permutations(types, num_types))
        ents_perm = set(itertools.product(ents, repeat=num_ents))
        rels_perm = set(itertools.product(rels, repeat=num_rels))
        types_perm = set(itertools.product(types, repeat=num_types))
        
        if len(types_perm) == 0:
            types_perm.add("type")
        if len(rels_perm) == 0:
            rels_perm.add("relation")
        if len(ents_perm) == 0:
            ents_perm.add("entity")

        # print(len(types_perm),len(rels_perm),len(ents_perm))
        # print("total loops : ",len(types_perm)*len(rels_perm)*len(ents_perm))

        counter = 0
        for type_p in types_perm:
            for rel_p in rels_perm:
                for ent_p in ents_perm:
                    new_sketch = sketch
                    new_sketch = new_sketch.replace("("," ( ").replace(")"," ) ")

                    for t in type_p:
                        new_sketch = new_sketch.replace(" type ",f" {t} ",1)
                    for r in rel_p:
                        new_sketch = new_sketch.replace(" relation ",f" {r} ",1)
                    for e in ent_p:
                        new_sketch = new_sketch.replace(" entity ",f" {e} ",1)
                    
                    new_sketch = new_sketch.replace(" ( ","(").replace(" ) ",")")
                    all_perms.append(new_sketch)
                    counter += 1

                    is_valid=True
                    try :
                        if "time_macro" in new_sketch:
                            if "time_macro)" in new_sketch:
                                is_valid = False
                            elif not re.search(r"\.time_macro [0-9]+", new_sketch):
                                is_valid = False
                            elif re.search(r"\.time_macro [mg]\.", new_sketch):
                                is_valid = False

                        if is_valid:
                            form_1 = lisp_to_nested_expression(new_sketch)
                            G = matcher.logical_form_to_graph(form_1)
                            edges = list(G.edges(data = True))
                            nodes = list(G.nodes(data=True))
                            is_valid = logical_form_validator.check_validity(edges=edges, nodes=nodes, relation_dr=relation_dr, upper_types=upper_types)
                            VALIDITY_CACHE.update({new_sketch : is_valid})

                    except Exception as e:
                        print("Error in perm utils")
                        print(sketch)
                        print(new_sketch)
                        print(str(e))
                        print("-"*10)
                        VALIDITY_CACHE.update({new_sketch : False})
                        continue
                    
                    if is_valid:
                        valid_perms.append(new_sketch)
                        # em = matcher.same_logical_form(gold_sexp, new_sketch)
                        # if em:
                        #     correct_lfs.append(new_sketch)

                        # print(new_sketch)
                    if " entity " in new_sketch or " relation " in new_sketch or " type " in new_sketch:
                        print("Error")
                        print(new_sketch)
                        print(rel_p, type_p, ent_p)
                        print("-"*20)
                        continue

                if counter >= 50000:
                    break


    print("final : ", len(set(all_perms)), len(set(valid_perms)))
    return list(set(valid_perms)), list(set(all_perms)-set(valid_perms)), list(set(correct_lfs))


def get_reverse_relation_cache():
        with open(reverse_properties_file_path,"r") as fread:
            rev_rels = fread.readlines()
        
        rev_rels_cache = {}
        for rel in rev_rels:
            rel_split = rel.replace("\n","").split("\t")
            rev_rels_cache.update({rel_split[0] : rel_split[1]})
            rev_rels_cache.update({rel_split[1] : rel_split[0]})
        
        return rev_rels_cache

def prepare_data(data_class, lf_candidates_cache_path, split, entity_to_type_cache_path, data_subset=()):

    rev_rels_cache = get_reverse_relation_cache()
    logical_form_validator = LogicalFormValidator(entity_to_type_cache_path=entity_to_type_cache_path)
    
    if os.path.exists(lf_candidates_cache_path):
        with open(lf_candidates_cache_path, "r") as fread:
            lf_cache_current = json.load(fread)
        print("Lf cache already exist at given path of size ", len(lf_cache_current))
    else:
        lf_cache_current = {}
    
    data_class.lf_candidates = copy.deepcopy(lf_cache_current)

    if data_subset != ():
        print("Preparing data for subset : ", data_subset)
    else:
        print("Preparing data for entire data")
    
    for qid_idx, qid in tqdm(enumerate(data_class.data)):

        # if qid_idx <= 1454:
        #     continue
        # print(qid_idx)
        if data_subset != ():
            if qid_idx < data_subset[0] or qid_idx >= data_subset[1]:
                continue
        
        question = data_class.get_question_by_qid(qid)

        linked_schema_cls, linked_schema_rels = data_class.get_linked_schema_by_qid(qid)

        gold_sexp_list = []
        gold_sketch = ""
        if split=="train":
            # use gold sketch and entities for train.
            linked_entities =  data_class.get_gold_entities_by_qid(qid)
            gold_sexp_list = data_class.get_gold_sexpression_by_qid(qid)
            gold_sketch = data_class.get_gold_sketch_by_qid(qid)
            predicted_sketches = [gold_sketch]
        else:
            predicted_sketches = data_class.get_predicted_sketch_by_qid(qid, top_k=1)
            linked_entities = data_class.get_entities_by_qid(qid)
        
        if "http" in predicted_sketches[0]:
            with_time_macro = []
            for rel in linked_schema_rels:
                typ = ".".join(rel.split(".")[:-1])
                if not ("common." in typ):
                    with_time_macro.append(f"{typ}.time_macro")
                # with_time_macro.append(f"{rel}.time_macro")
            with_time_macro = list(set(with_time_macro))
            linked_schema_rels.extend(with_time_macro)
        
        linked_schema_rev_rels = []
        for rel in linked_schema_rels:
            if rev_rels_cache.get(rel, "") != "":
                linked_schema_rev_rels.append(rev_rels_cache.get(rel))
        linked_schema_rels.extend(linked_schema_rev_rels)


        # linked_schema_rev_rels = []
        # for rel in linked_schema_rels:
        #     linked_schema_rev_rels.append(f"(R {rel})")
        # linked_schema_rels.extend(linked_schema_rev_rels)

        # linked_schema_rev_rels = []
        # for rel in linked_schema_rels:
        #     if not "time_macro" in rel:
        #         linked_schema_rev_rels.append(f"(R {rel})")
        #         if rev_rels_cache.get(rel, "") != "":
        #             linked_schema_rev_rels.append(rev_rels_cache.get(rel))
        # linked_schema_rels.extend(linked_schema_rev_rels)

        if data_class.lf_candidates.get(str(qid), {}) != {}:
            # check if candidates already exist in the file.
            valid_sexps = data_class.lf_candidates[str(qid)]["valid_sexps"]
            invalid_sexps = data_class.lf_candidates[str(qid)]["invalid_sexps"]
            correct_sexps = data_class.lf_candidates[str(qid)]["correct_sexps"]
        else:
            
            valid_sexps, invalid_sexps, correct_sexps = get_all_sexps(sketches=predicted_sketches, rels=linked_schema_rels,\
                                                                        types=linked_schema_cls, ents=linked_entities,\
                                                                        gold_sexp_list=gold_sexp_list, logical_form_validator=logical_form_validator)
            data_class.lf_candidates.update({str(qid) : {"valid_sexps": valid_sexps[:], "invalid_sexps": invalid_sexps[:],\
                                                            "correct_sexps":correct_sexps[:]}})
            
            if qid_idx%250==0:
                # update cache file
                data_class.update_lf_candidates_cache(data_class.lf_candidates,write_path=lf_candidates_cache_path)


        if qid_idx%250==0:
            logical_form_validator.update_entity_type_cache()
            logical_form_validator = LogicalFormValidator(entity_to_type_cache_path=entity_to_type_cache_path)

    data_class.update_lf_candidates_cache(data_class.lf_candidates,write_path=lf_candidates_cache_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lf_candidates_cache_path', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)
    parser.add_argument('--start_index', default=-1, type=int, required=False)
    parser.add_argument('--end_index', default=-1, type=int, required=False)
    parser.add_argument('--entity_to_type_cache_path', default="",type=str, required=False)

    args = parser.parse_args()

    print("Preparing data...")
    data_subset = ()
    if args.start_index != -1 and args.end_index != -1:
        data_subset = (args.start_index, args.end_index)

    if args.split=="train":
        print("preparing ranker data for train")
        data_class = Data(data_path=data_train_path, \
                                schema_path=sr_output_train_path,\
                                sketch_path=sketch_output_train_nk_path,\
                                entity_linking_path=el_output_train_nk_el_path,\
                                lf_candidates_path=lfi_output_train_path,\
                                split="train")


    elif args.split=="dev":                
        print("preparing ranker data for dev")         
        data_class = Data(data_path=data_dev_path,\
                                schema_path=sr_output_dev_path,\
                                sketch_path=sketch_output_dev_path,\
                                entity_linking_path=el_output_dev_el_path,\
                                lf_candidates_path=lfi_output_dev_path,\
                                split="dev")

    elif args.split=="test":
        data_class = Data(data_path=data_test_path, \
                                schema_path=sr_output_test_path,\
                                sketch_path=sketch_output_test_path,\
                                entity_linking_path=el_output_test_el_path,\
                                lf_candidates_path=lfi_output_test_path,\
                                split="test")
        


    lf_candidates_cache_path = f"{current_file_dir}/../../{args.lf_candidates_cache_path}"
    entity_to_type_cache_path = f"{current_file_dir}/../../{args.entity_to_type_cache_path}"

    prepare_data(data_class=data_class, lf_candidates_cache_path=lf_candidates_cache_path, \
                 split=args.split, entity_to_type_cache_path=entity_to_type_cache_path,\
                 data_subset=data_subset)