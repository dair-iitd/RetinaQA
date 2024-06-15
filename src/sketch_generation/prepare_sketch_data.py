import re
import json
from tqdm import tqdm
import copy
import argparse

import sys
sys.path.append('.')
sys.path.append('..')

from configs.common_config import domain_dict_file_path

with open(domain_dict_file_path,"r") as fread:
    schema_items_dict = json.load(fread)

etypes = []
rels = []
for k,v in schema_items_dict.items():
    domain_len = len(k.split("."))
    for si in v:
        if len(si.split("."))-domain_len == 1:
            etypes.append(si)
        elif len(si.split("."))-domain_len == 2:
            rels.append(si)

print(len(etypes), len(rels))
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
    return s_exp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--split', type=str, required=True, choices=["train", "dev","test"])
    # parser.add_argument('--train_type', type=str, required=True, choices=["au", "a"])
    parser.add_argument('--data_file', type=str, required=True)

    args = parser.parse_args()


    fread = open(args.data_file)
    data = json.load(fread)
    write_path = args.data_file.replace(".json","_sketch.json")

    if args.split == "test":
        new_data = []
        for d in tqdm(data):
            temp_d = copy.deepcopy(d)
            temp_d.update({"sketch" : ""})
            new_data.append(temp_d)

    else:

        new_data = []
        for d in tqdm(data):
            
            temp_d = copy.deepcopy(d) 
            s_exp = d["s_expression"]

            if s_exp in ["NK"]:
                temp_d["sketch"] = "NK"
            else:
                gt_sketch = get_sketch(s_exp)
                temp_d["sketch"] = gt_sketch
            new_data.append(temp_d)

    print(len(new_data))
    with open(f"{write_path}","w") as fwrite:
        json.dump(new_data, fwrite)