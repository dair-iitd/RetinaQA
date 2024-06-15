import re
import json
from tqdm import tqdm
import copy
import argparse

import sys
sys.path.append('.')
sys.path.append('..')

from configs.common_config import domain_dict_file_path, fb_roles_file_path

with open(fb_roles_file_path, "r") as fread:
    fb_roles_data = fread.readlines()

with open(domain_dict_file_path,"r") as fread:
    schema_items_dict = json.load(fread)

etypes = []
rels = []
for d in fb_roles_data:
    d_split = d.split(" ")
    d_0 = d_split[0].replace("\n","").strip()
    d_1 = d_split[1].replace("\n","").strip()
    d_2 = d_split[2].replace("\n","").strip()

    rels.append(d_1)
    etypes.append(d_0)
    etypes.append(d_2)
    rels.append(f'{d_0}.time_macro')
    rels.append(f'{d_2}.time_macro')

for k,v in schema_items_dict.items():
    domain_len = len(k.split("."))
    for si in v:
        if len(si.split("."))-domain_len == 1:
            if not si in etypes:
                etypes.append(si)
                rels.append(f"{si}.time_macro")
        elif len(si.split("."))-domain_len == 2:
            if not si in rels:
                rels.append(si)
            
        # if len(si.split(".")) == 2:
        #     etypes.append(si)
        # elif len(si.split(".")) == 3:
        #     rels.append(si)

print(f"Len of Etypes : {len(etypes)}, Rels: {len(rels)}")
etypes.sort(key = len, reverse=True)
rels.sort(key = len, reverse=True)



def get_sketch(s_exp):
    s_exp = s_exp.replace("("," ( ").replace(")"," ) ")
    s_exp = re.sub(r"\b(m|g)\.[0-9a-zA-Z\_]+\b","entity",s_exp)

    s_exp_split = s_exp.split(" ")
    new_s_exp_split = []
    for token in s_exp_split:
        if token in rels:
            new_s_exp_split.append("relation")
        elif token in etypes:
            new_s_exp_split.append("type")
        else:
            new_s_exp_split.append(token)


    s_exp = " ".join(new_s_exp_split)
    s_exp = s_exp.replace(" ( ","(").replace(" ) ",")")
    # s_exp = s_exp.replace("(R relation)","relation")
    return s_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', type=str, required=True, choices=["train", "dev", "test"])
    # parser.add_argument('--train_type', type=str, required=True, choices=["au", "a"])
    parser.add_argument('--data_file', type=str, required=True)

    args = parser.parse_args()


    fread = open(args.data_file)
    data = json.load(fread)
    write_path = args.data_file.replace(".json","_sketch.json")

    # if args.split == "test":
    #     new_data = []
    #     for d in tqdm(data):
    #         temp_d = copy.deepcopy(d)
    #         temp_d.update({"sketch" : ""})
    #         new_data.append(temp_d)

    # else:

    new_data = []
    for d in tqdm(data):
        
        temp_d = copy.deepcopy(d) 

        possible_sexps = []
        possible_sketches = []
        
        for parse in d["Parses"]:
            s_exp = parse["SExpr"]
            if s_exp == "null":
                continue
            possible_sexps.append(s_exp)
            possible_sketches.append(get_sketch(s_exp))


        if len(possible_sexps) == 0:
            possible_sketches = ["null"]
            possible_sexps = ["null"]


        gt_sketch = possible_sketches[0] 
        temp_d["sketch"] = gt_sketch
        temp_d["qid"] = d["QuestionId"]
        temp_d["question"] = d["ProcessedQuestion"]
        temp_d["s_expression"] = possible_sexps[0]
        temp_d["possible_sexps"] = possible_sexps
        new_data.append(temp_d)

    print(len(new_data))
    with open(f"{write_path}","w") as fwrite:
        json.dump(new_data, fwrite)

        