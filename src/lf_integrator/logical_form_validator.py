import json
import argparse
from SPARQLWrapper import SPARQLWrapper, JSON
import urllib
import os

from configs.common_config import freebase_addr, freebase_port

sparql = SPARQLWrapper(f"http://{freebase_addr}:{freebase_port}/sparql")
sparql.setReturnFormat(JSON)


def load_json(fname):
    with open(fname) as f:
        return json.load(f)
    f.close()


class LogicalFormValidator:

    def __init__(self,entity_to_type_cache_path=None):
        self.entity_to_type_cache_path = entity_to_type_cache_path
        if entity_to_type_cache_path and os.path.exists(entity_to_type_cache_path):
            self.entity_to_type_cache = load_json(entity_to_type_cache_path)
        else:
            self.entity_to_type_cache = {}

    def update_entity_type_cache(self):
        if self.entity_to_type_cache_path and os.path.exists(self.entity_to_type_cache_path):
            entity_to_type_cache_latest = load_json(self.entity_to_type_cache_path)
        else:
            entity_to_type_cache_latest = {}
        entity_to_type_cache_latest.update(self.entity_to_type_cache)
        with open(self.entity_to_type_cache_path,"w") as fwrite:
            json.dump(entity_to_type_cache_latest, fwrite)
        fwrite.close()

    def get_types(self, entity: str):

        if entity in self.entity_to_type_cache:
            return self.entity_to_type_cache[entity]

        query = ("""
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX : <http://rdf.freebase.com/ns/> 
        SELECT (?x0 AS ?value) WHERE {
        SELECT DISTINCT ?x0  WHERE {
        """
                ':' + entity + ' :type.object.type ?x0 . '
                                """
        }
        }
        """)
        # print(query)
        sparql.setQuery(query)
        try:
            results = sparql.query().convert()
        except urllib.error.URLError as e:
            print("Exception : " , str(e))
            return []

        rtn = []
        for result in results['results']['bindings']:
            rtn.append(result['value']['value'].replace('http://rdf.freebase.com/ns/', ''))

        self.entity_to_type_cache.update({entity : rtn[:]})
        return rtn

    def check_validity(self, edges, nodes, relation_dr, upper_types):

        nodes = {n[0]: n for n in nodes}
        # print(edges)
        # print(nodes)

        is_valid = True

        for edge in edges:
            # print("edge : ", edge)
            head_node = edge[0]
            tail_node = edge[1]
            rel = edge[2]['relation']

            head_literal = False
            tail_literal = False

            try:
                gt_head_type = relation_dr[rel][0]
                gt_tail_type = relation_dr[rel][1]

                # print(gt_head_type)
                # print(gt_tail_type)

                current_head_type = []
                current_tail_type = []

                if nodes[head_node][1]["type"] == "literal":
                    head_literal = True
                
                if nodes[tail_node][1]["type"] == "literal":
                    tail_literal = True

                # if nodes[head_node][1]["type"] == "literal" or nodes[tail_node][1]["type"] == "literal":
                #     return True

                if nodes[head_node][1]["type"] == "entity":
                    current_head_type = self.get_types(nodes[head_node][1]["id"])
                else:
                    current_head_type = [nodes[head_node][1]["id"]]
                    current_head_type.extend(upper_types[current_head_type[0]])
                
                if nodes[tail_node][1]["type"] == "entity":
                    current_tail_type = self.get_types(nodes[tail_node][1]["id"])
                else:
                    current_tail_type = [nodes[tail_node][1]["id"]]
                    current_tail_type.extend(upper_types[current_tail_type[0]])
                    
            except Exception as e:
                print("check_validity exception : ", str(e))
                return False


            # print("current_head_type : ",current_head_type, gt_head_type)
            # print(current_tail_type, gt_tail_type)
            if (not gt_head_type in current_head_type) and (not head_literal):
                # print(edge)
                # print("head check failed")
                # print(gt_head_type)
                # print(current_head_type)
                return False
            if (not gt_tail_type in current_tail_type) and (not tail_literal):
                # print(edge)
                # print("tail check failed")
                # print(gt_tail_type)
                # print(current_tail_type)
                return False

        return True


if __name__ == '__main__':
    
    from common.semantic_matcher import matcher
    logical_form_validator = LogicalFormValidator()

    s_exp = "(AND travel.hotel (JOIN travel.hotel.drinking_establishments m.09v5rgg))"

    form_1 = lisp_to_nested_expression(s_exp)
    G = matcher.logical_form_to_graph(form_1)
    edges = list(G.edges(data = True))
    nodes = list(G.nodes(data=True))

    is_valid = logical_form_validator.check_validity(edges=edges, nodes=nodes, relation_dr=relation_dr, upper_types=upper_types)
    print(is_valid)
    exit()


    

    