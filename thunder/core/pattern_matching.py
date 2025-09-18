import itertools

import torch
import thunder
from thunder.core.trace import TraceCtx, tracectx
from thunder.core import prims
from thunder.core.proxies import proxy
import thunder.torch as ltorch
from thunder.core import utils
from thunder.executors.data_dependent_partition import horizontal_merge, dataflow_merge
from thunder.executors.utils import Region
from thunder.core.proxies import variableify, unvariableify


from thunder.executors.data_dependent_partition import Graph

x = torch.ones(2, 2)

y = torch.ones(2, 2)
z = torch.ones(2, 2)


def f(x):
    y = torch.sin(x)
    t = torch.mul(y, x)
    s = torch.add(t, x)
    return s.sum()


# def pattern(a, b, c):
#     t = torch.mul(a, b)
#     torch.add(t, c)

# def replacement(a, b, c):
#     t = torch.addcmul(a, b, c)
#     return t

jf = thunder.jit(f)
jf(x)
trc = thunder.last_traces(jf)[-1]
graph = Graph(trc)

for bsym in trc.bound_symbols:
    print(bsym)
print()


pattern_trc = TraceCtx()
for name in trc.names:
    pattern_trc.add_name(name)
with tracectx(pattern_trc):
    px = proxy(x)
    py = proxy(y)
    pz = proxy(z)
    t = prims.mul(py, pz)
    s = prims.add(t, px)
    # return statement distinguishes the output
    prims.python_return(s)

pattern_outputs = pattern_trc.output
if not isinstance(pattern_outputs, tuple):
    pattern_outputs = (pattern_outputs,)
pattern_trc.bound_symbols = pattern_trc.bound_symbols[:-1]

pattern_graph = Graph(pattern_trc)

replacement_trc = TraceCtx()
for name in pattern_trc.names:
    replacement_trc.add_name(name)

with tracectx(replacement_trc):
    px = proxy(x)
    py = proxy(y)
    pz = proxy(z)
    t = ltorch.addcmul(px, py, pz)
    # return statement distinguishes the output
    prims.python_return(t)


def match_symbol(node_1, node_2):
    return node_1.group_bsyms[0].sym.name == node_2.group_bsyms[0].sym.name


def construct_swap_map(node_1, node_2):
    swap_map = {}
    bsym_1 = node_1.group_bsyms[0]
    bsym_2 = node_2.group_bsyms[0]
    for input_1, input_2 in zip(bsym_1.flat_proxy_args, bsym_2.flat_proxy_args):
        swap_map[variableify(input_2)] = unvariableify(input_1)
    for output_1, output_2 in zip(bsym_1.flat_proxy_outs, bsym_2.flat_proxy_outs):
        swap_map[variableify(output_2)] = unvariableify(output_1)
    return swap_map


def match_subtree(node, pattern_node):
    if not match_symbol(node, pattern_node):
        return False, {}
    if not pattern_node.children:
        swap_map = construct_swap_map(node, pattern_node)
        return [node], swap_map
    unmatched = node.children.copy()
    for pattern_child in pattern_node.children:
        pattern_child_matched = False
        nonmatch = utils.OrderedSet()
        while unmatched:
            child = unmatched.pop()
            subtree_match, swap_map = match_subtree(child, pattern_child)
            if subtree_match is False:
                nonmatch.add(child)
            else:
                pattern_child_matched = True
                break
        unmatched = unmatched.union(nonmatch)
        if not pattern_child_matched:
            return False, {}
    subtree_match.append(node)
    swap_map.update(construct_swap_map(node, pattern_node))
    return subtree_match, swap_map


queue = graph.roots[:]


def merge_func(node_1, node_2):
    return hasattr(node_1, "stupid_flag") and hasattr(node_2, "stupid_flag")


subtree_match = None
while queue:
    node = queue.pop(0)
    subtree_match, pattern_to_orig_swap_map = match_subtree(node, pattern_graph.roots[0])
    if subtree_match is not False:
        print("matched")
        break

    queue.extend(node.children)

# mark the nodes that match the pattern
for n in subtree_match:
    n.stupid_flag = True

# fuse the nodes that match the pattern into a single node
dataflow_merge(graph, merge_func)
ret = horizontal_merge(graph, merge_func)
# locate the fused region (the one with more than one bsym)
for i, r in enumerate(ret):
    if len(r) > 1:
        break


producers, consumers = utils.producers_and_consumers(trc)
region = Region(producers, consumers, r)


replacement_outputs = replacement_trc.output
if not isinstance(replacement_outputs, tuple):
    replacement_outputs = (replacement_outputs,)
# remove return statement
replacement_trc.bound_symbols = replacement_trc.bound_symbols[:-1]
producers, consumers = utils.producers_and_consumers(replacement_trc)
replacement_region = Region(producers, consumers, replacement_trc.bound_symbols)


swap_map = {}
# !!! This needs prevalidation
for input, matched_input in zip(region.inputs, replacement_region.inputs):
    swap_map[variableify(matched_input)] = unvariableify(input)

match_outputs = [pattern_to_orig_swap_map[variableify(output)] for output in pattern_outputs]
for output, matched_output in zip(match_outputs, replacement_outputs):
    swap_map[variableify(matched_output)] = unvariableify(output)


replacement_bsyms = [bsym.from_bsym_swap_proxies(swap_map) for bsym in replacement_region.bound_symbols]

# replace the matched region with the replacement region
ret[i] = replacement_bsyms
# update any output tensor names
for j in range(i + 1, len(ret)):
    for k in range(len(ret[j])):
        ret[j][k] = ret[j][k].from_bsym_swap_proxies(swap_map)

print()
print("new trace")
for r in itertools.chain(*ret):
    print(r)
