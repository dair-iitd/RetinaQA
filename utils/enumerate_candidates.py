"""
 Copyright (c) 2021, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""


from collections import Counter, OrderedDict
from utils.expr_parser import extract_entities, parse_s_expr, tokenize_s_expr


def ordered_set_as_list(xs):
    ys = []
    for x in xs:
        if x not in ys:
            ys.append(x)
    return ys


def approx_time_macro_ast(node):
    # print('NODE', node.construction, node.logical_form())
    if (node.construction == 'AND' and
        node.fields[0].construction == 'JOIN' and
        node.fields[0].fields[0].construction == 'SCHEMA' and 
        'time_macro' in node.fields[0].fields[0].val):
        return node.fields[1]
    else:
        new_fileds = [approx_time_macro_ast(x) for x in node.fields]
        node.fields = new_fileds
        return node

def get_approx_s_expr(x):
    if not ('time_macro' in x):
        return x

    ast = parse_s_expr(x)
    approx_ast = approx_time_macro_ast(ast)
    approx_x = approx_ast.compact_logical_form()
    return approx_x

