# This is a "TorchScript-like" graph representation of Python IR.
# The idea is that blocks are "simple blocks" in terms of the code flow graph,
# i.e. without branches
import collections
import copy
import inspect
from typing import Any, Dict, List, Optional, Tuple, Type, TYPE_CHECKING, Set, Union
from collections.abc import Iterable, Iterator, Sequence

from thunder.core.script.python_ir_data import stack_effect_detail
from thunder.core.utils import OrderedSet

if TYPE_CHECKING:
    import dis
    import graphviz

GraphObject = Union["Value", "Node", "Block"]


def assert_value(v: Union[GraphObject, None]) -> "Value":
    assert isinstance(v, Value)
    return v


def assert_node(n: Union[GraphObject, None]) -> "Node":
    assert isinstance(n, Node)
    return n


def assert_block(bl: Union[GraphObject, None]) -> "Block":
    assert isinstance(bl, Block)
    return bl


class GraphSummaryCallback:
    def node(self, n: "Node") -> Tuple[list[str], list[str]]:
        return [], []

    def finish(self) -> list[str]:
        return []


class NULL:
    """marker for non-existant object."""

    pass


class MROAwareObjectRef:  # or as they call it super
    def __init__(self, obj: Any, start_klass: Optional[type] = None):
        self.obj = obj
        self.start_klass = start_klass

    def __getattr__(self, name: str) -> Any:
        ## handle non-methods...
        i = 0
        mro = inspect.getmro(self.obj.value.__class__)
        if self.start_klass is not None:
            while i < len(mro) and not mro[i] == self.start_klass:
                i += 1
            i += 1
        while i < len(mro) and not hasattr(mro[i], name):
            i += 1
        if i >= len(mro):
            raise AttributeError(f"{name} not a member")
        return getattr(mro[i], name)


# Values are
# - function arguments as inputs to the graph (including self)
# - constants and globals
# - intermediate results / local variables
# - attributes of other values given in .parent
# they can be used
# - as inputs and outputs of nodes (but inplace is still tricky)
# - as block_outputs (note that block_outputs can be either outputs of nodes
#   or attribute lookups).
# block_outputs (and only these) typically have .phi_values recorded.
# PhiValues are the block_inputs.
# - they have (one or multiple) block_outputs as .values, these are set at the
#   .jump_sources (TODO: .jump_sources records None for non-node-generated).
# - There must be a 1-1 correspondence between <Value>.phi_values-><PhiValue> and <PhiValue>.values-><Value>.
# All block_inputs (at least before an optimization pass towards the un-ssa-ing)
# are expected to be PhiValues and all PhiValues are expected to show up as
# block_inputs.
class Value:
    def __init__(
        self,
        *,
        node: Optional["Node"] = None,
        nr: Optional[int] = None,
        typ: Optional[type] = None,
        value: Any = None,
        name: Optional[str] = None,
        parent: Optional["Value"] = None,
        is_global: bool = False,
        is_const: bool = False,
        is_function_arg: bool = False,
    ):
        self.node = node
        self.nr = nr
        self.typ = typ if typ is not None or value in (None, NULL) else type(value)
        self.value = value
        self.name = name
        self.parent = parent
        self.is_global = is_global
        self.is_const = is_const
        self.is_function_arg = is_function_arg
        self.phi_values: list["PhiValue"] = []

    def clone(self, translation_dict: Optional[dict[GraphObject, GraphObject]] = None) -> "Value":
        # clones a value, including (recursively) parent value
        # uses translation_dict to look up parent value
        # updates translation_dict
        # does not register phi_values on the clone
        # always clone parents?
        if translation_dict is None:
            translation_dict = {}
        if self in translation_dict:
            return assert_value(translation_dict[self])
        parent = self.parent
        if parent:
            if parent in translation_dict:
                parent = assert_value(translation_dict[parent])
            else:
                parent = parent.clone(translation_dict=translation_dict)
        v = Value(
            node=self.node,
            nr=self.nr,
            typ=self.typ,
            value=self.value,
            name=self.name,
            parent=parent,
            is_global=self.is_global,
            is_const=self.is_const,
            is_function_arg=self.is_function_arg,
        )
        if translation_dict is not None:
            translation_dict[self] = v
        return v

    def __str__(self) -> str:
        parts = []
        if self.is_function_arg:
            parts.append("funcarg")
        if self.name:
            parts.append(f"name={self.name}")
        if self.typ is not None:
            parts.append(f"typ={self.typ}")
        if self.value:
            parts.append(f"value of type {type(self.value)}")
        if self.is_const:
            parts.append("const")
        if self.is_global:
            parts.append("global")
        if self.parent is not None:
            parts.append(f"parent={self.parent}")
        return f"""{type(self).__name__} {hex(id(self))} ({' '.join(parts)})"""

    def __repr__(self) -> str:
        return f"{super().__repr__()[:-1]} {self}>"


class PhiValue(Value):
    # node?
    def __init__(
        self,
        values: list[Value],
        jump_sources: Sequence[Optional["Node"]],
        block: "Block",
        _unfinished_clone: bool = False,
    ):
        super().__init__()
        self._unfinished_clone = _unfinished_clone
        self.block: Block = block
        self._set_values_jump_sourcess(values, jump_sources)

    def _set_values_jump_sourcess(self, values: list[Value], jump_sources: Sequence[Optional["Node"]]) -> None:
        assert len(values) == len(jump_sources)
        self.values = list(values)
        if not self._unfinished_clone:
            for v in self.values:
                if v is not None:
                    v.phi_values.append(self)
        self.jump_sources = list(jump_sources)

    def clone(self, translation_dict: Optional[dict[GraphObject, GraphObject]] = None) -> "PhiValue":
        # due to loops in the Graph, this is complicated:
        # we do not translate values or jump_sources here, but do
        # translate blocks.
        if translation_dict is None:
            translation_dict = {}
        if self in translation_dict:
            v = translation_dict[self]
            assert isinstance(v, PhiValue)
            return v
        v = PhiValue(self.values, self.jump_sources, assert_block(translation_dict[self.block]), _unfinished_clone=True)
        translation_dict[self] = v
        return v

    def post_process_clone(self, *, translation_dict: dict[GraphObject, GraphObject]) -> None:
        assert self._unfinished_clone
        self._unfinished_clone = False
        self._set_values_jump_sourcess(
            [assert_value(translation_dict.get(v, v)) for v in self.values],
            [(assert_node(translation_dict.get(js, js)) if js is not None else None) for js in self.jump_sources],
        )

    def add_missing_value(
        self, v: Value, idx: Optional[int] = None, jump_source: Optional["Node"] = None
    ) -> None:  # None: append
        if idx is None:
            assert v not in self.values
            self.values.append(v)
            v.phi_values.append(self)
            self.jump_sources.append(jump_source)
        else:
            assert 0 <= idx < len(self.values)
            assert self.values[idx] is None
            assert jump_source is None
            self.values[idx] = v
            v.phi_values.append(self)

    def remove_value(self, v: Value) -> None:
        idx = self.values.index(v)
        v.phi_values.remove(self)
        del self.values[idx]
        del self.jump_sources[idx]

    def replace_value(self, v_old: Value, v_new: Value) -> None:
        if v_old is v_new:
            return

        assert v_new not in self.values
        idx = self.values.index(v_old)
        self.values[idx] = v_new
        self.jump_sources[idx] = None

        v_old.phi_values.remove(self)
        v_new.phi_values.append(self)


# A node corresponds to one Python bytecode instruction given in .i
# it has Values as .inputs and .outputs
class Node:
    def __init__(
        self,
        *,
        i: "dis.Instruction",
        inputs: Optional[list[Value]] = None,
        outputs: Optional[list[Value]] = None,
        line_no: Optional[int] = None,
    ):
        self.i = i
        self.inputs: list[Value] = inputs if inputs is not None else []
        self.outputs: list[Value] = outputs if outputs is not None else []
        self.jump_targets: list[tuple[tuple[int, int], Block]] = []
        self.line_no = line_no
        self.block: Optional[Block] = None

    def clone(self, translation_dict: Optional[dict[GraphObject, GraphObject]] = None) -> "Node":
        """.block of the clone will be None if block is not in translation dict."""
        if translation_dict is None:
            translation_dict = {}
        if self in translation_dict:
            return assert_node(translation_dict[self])
        inputs = [i.clone(translation_dict=translation_dict) for i in self.inputs]
        outputs = [o.clone(translation_dict=translation_dict) for o in self.outputs]
        i = copy.copy(self.i)
        n2 = Node(i=i, inputs=inputs, outputs=outputs, line_no=self.line_no)
        n2.jump_targets = [(se, assert_block(translation_dict.get(bl, bl))) for se, bl in self.jump_targets]
        if self.block is None:
            n2.block = None
        else:
            bl2 = translation_dict.get(self.block)
            assert bl2 is None or isinstance(bl2, Block)
            n2.block = bl2
        translation_dict[self] = n2
        return n2

    def set_jump_target(self, jt: "Block", idx: Optional[int] = None) -> None:
        # TODO: more validation?
        # is_jump = (self.i.opname not in unconditional_jump_names) or (idx == 1) or (idx is None and self.jump_targets)
        # assert is_jump

        jt_plus = (stack_effect_detail(self.i, jump=False), jt)
        if idx is None:
            assert len(self.jump_targets) <= 1
            self.jump_targets.append(jt_plus)
        else:
            old_jt = self.jump_targets[idx][1]
            old_jt.jump_sources.remove(self)
            self.jump_targets[idx] = jt_plus
        jt.jump_sources.append(self)

    def __str__(self) -> str:
        # i.i.offset // 2, i.i.opname, i.i.arg, "(", i.i.argval, ")"
        if self.i.opname in {"CALL_METHOD", "CALL_FUNCTION"}:
            return f"{self.i.opname}({self.inputs})"
        return f"{self.i.opname} {self.i.arg} ({self.i.argval})"  # str(self.i)

    def __repr__(self) -> str:
        return f"{super().__repr__()[:-1]} {self}>"


# Blocks have the first instruction (only) as the jump target
# (or the function entry point)
# Blocks always have a single final instruction that jumps (or RETURN)
# conditional jumps (including e.g. FOR_ITER) always have the non-jumping
# target first and then the jumping target.
# The jump targets are other blocks and are atributes of the jump instruction.
class Block:
    def __init__(self):
        self.jump_sources: list[Optional[Node]] = []
        self.nodes: list[Node] = []
        self.block_inputs: list[Value] = []
        self.block_outputs = OrderedSet([])

    def __str__(self) -> str:
        return "\n".join([f"  Block (reached from {self.jump_sources})"] + ["    " + str(n) for n in self.nodes])

    def __repr__(self) -> str:
        return f"{super().__repr__()[:-1]} {self}>"

    def insert_node(self, n: Node, insert_after: Optional[Node] = None, insert_before: Optional[Node] = None) -> None:
        assert n.block is None
        assert (insert_after is None) != (insert_before is None), f"{insert_after=} {insert_before=}"
        to_find = insert_after or insert_before
        for idx, n2 in enumerate(self.nodes):
            if n2 is to_find:
                break
        if n2 is not to_find:
            raise ValueError(f"could not find node {n}")

        # validity checks? (also above)
        n.block = self
        if insert_after:
            self.nodes.insert(idx + 1, n)
        else:
            self.nodes.insert(idx, n)


# A graph contains Blocks.
# The first block (.blocks[0]) is the entry point. Other blocks are connected
# through jump instructions.
class Graph:
    def __init__(self, blocks: Optional[list[Block]] = None):
        self.blocks = [] if blocks is None else blocks[:]

    def __str__(self) -> str:
        return "\n".join(["Graph of"] + [str(b) for b in self.blocks])

    def __repr__(self) -> str:
        return f"{super().__repr__()[:-1]} {self}>"

    def nodes(self) -> Iterator[Node]:
        for b in self.blocks:
            yield from b.nodes

    def ensure_links(self) -> None:
        for bl in self.blocks:
            bl.graph = self
            for n in bl.nodes:
                n.block = bl
                inps = set(n.inputs)
                for o in n.outputs:
                    if o not in inps:  # not for inplace
                        o.block = bl
                        o.node = n
            for i in bl.block_inputs:
                i.block = bl

    def clone(self) -> tuple["Graph", dict[GraphObject, GraphObject]]:
        bls2, translation_dict = clone_blocks(self.blocks)
        g2 = Graph(blocks=bls2)
        g2.local_variables_at_start = [v.clone() for v in self.local_variables_at_start]
        replace_values(g2, {k: v for k, v in zip(self.local_variables_at_start, g2.local_variables_at_start)})
        g2.ismethod = self.ismethod
        g2.co_argcount = self.co_argcount
        g2.co_flags = self.co_flags
        g2.co_posonlyargcount = self.co_posonlyargcount
        g2.co_kwonlyargcount = self.co_kwonlyargcount
        g2.func_defaults = self.func_defaults[:]
        g2.func_kwdefaults = self.func_kwdefaults.copy()
        g2.method = self.method
        g2.module = self.module
        g2.mro_klass = self.mro_klass
        g2.self_value = self.self_value
        g2.source_start_line = self.source_start_line
        g2.source_lines = self.source_lines[:]

        return g2, translation_dict

    def print(self) -> None:
        value_counter = 1
        print(self.local_variables_at_start)
        for bl in self.blocks:
            for n in bl.nodes:
                for o in n.outputs:
                    o.print_name = f"{o.name}:{value_counter}" if o.name is not None else f":{value_counter}"
                    value_counter += 1
                for i in n.inputs:
                    if not hasattr(i, "print_name"):
                        i.print_name = f"{i.name}:{value_counter}" if i.name is not None else f":{value_counter}"
                        value_counter += 1
                av = f"[{n.i.argval}]" if n.i.argval is not None else ""
                print(
                    ",".join(o.print_name for o in n.outputs),
                    "=",
                    n.i.opname,
                    f"{av}(",
                    ", ".join([i.print_name for i in n.inputs]) + ")",
                )

    def summary(self, print_lines: bool = False, callback=GraphSummaryCallback()) -> None:
        type_count = collections.Counter()
        results = {}

        def get_name(v):
            if v not in results:
                idx = type_count[type(v)]
                type_count[type(v)] += 1
                prefix = {PhiValue: "𝜙", Value: "V"}.get(type(v), type(v).__name__)
                results[v] = (prefix, idx)

                # Populate cache
                if isinstance(v, PhiValue):
                    _ = [get_name(vi) for vi in v.values]

            return "{}_{}".format(*results[v])

        graph_lines = []
        legend_lines = []

        for block in self.blocks:
            graph_lines.extend(
                (
                    f"Block inputs: {[get_name(i) for i in block.block_inputs]}",
                    f"Block outputs: {[get_name(i) for i in block.block_outputs]}",
                )
            )
            for i, node in enumerate(block.nodes):
                if (i == 0 or node.line_no != block.nodes[i - 1].line_no) and node.line_no is not None:
                    line = f"# l{node.line_no + self.source_start_line:3d} {self.source_lines[node.line_no].rstrip()}"
                else:
                    line = ""
                lines_before, lines_after = callback.node(node)
                graph_lines.extend(lines_before)
                graph_lines.append(
                    f"  {node.i.opname:<20} {f'{[get_name(v) for v in node.inputs]} -> {[get_name(v) for v in node.outputs]}':<80}   {line}"
                )
                graph_lines.extend(lines_after)
            graph_lines.append("")
        graph_lines.extend(callback.finish())

        for v, (prefix, idx) in sorted(results.items(), key=lambda x: x[1]):
            values = f"[{', '.join(get_name(vi) for vi in v.values)}]" if isinstance(v, PhiValue) else ""
            legend_lines.append(f"{prefix}_{idx}  {str(v):<16} {values}")

        if print_lines:
            print("\n".join(graph_lines) + "\n" + "\n".join(legend_lines))

        return tuple(graph_lines), tuple(legend_lines)


def unify_values(values: list[Value], jump_sources: list[Node], bl: Block, all_predecessors_done: bool = True) -> Value:
    if all_predecessors_done:
        if len(values) == 1:
            return values[0]
        val = values[0]
        if all(v is val for v in values[1:]):
            return val
        # different values
    return PhiValue(values, jump_sources, bl)


def insert_before(new_n: Node, n: Node) -> None:
    bl = assert_block(n.block)
    idx = bl.nodes.index(n)
    bl.nodes.insert(idx, new_n)
    new_n.block = n.block


def insert_after(new_n: Node, n: Node) -> None:
    bl = assert_block(n.block)
    idx = bl.nodes.index(n)
    bl.nodes.insert(idx + 1, new_n)
    new_n.block = n.block


def replace_values(
    gr_or_bl: Union[Graph, Block], value_map: dict[Value, Value], follow_phi_values: bool = False
) -> None:
    ### Replacing a value:
    # - as inputs/outputs of nodes
    # - value.parent for other values
    # - phi nodes
    # - graph input (?) / initial vars
    processed = set()

    def map_values(v: Value) -> Value:
        # do not call map_values without guarding for infinite recursion
        if v in processed:
            return value_map.get(v, v)
        processed.add(v)

        if v in value_map:
            if follow_phi_values:
                for pv in v.phi_values[:]:
                    pv.replace_value(v, value_map[v])
                    assert len(pv.values) == len(pv.jump_sources)
            return value_map[v]

        if isinstance(v.value, MROAwareObjectRef):
            v.value.obj = map_values(v.value.obj)
        if v.parent is not None:
            v.parent = map_values(v.parent)
        if isinstance(v, PhiValue):
            for ov in v.values:
                nv = map_values(ov)
                v.replace_value(ov, nv)
            assert len(v.values) == len(v.jump_sources)
        return v

    def process_block(bl: Block) -> None:
        bl.block_inputs = [map_values(vv) for vv in bl.block_inputs]
        for n in bl.nodes:
            n.inputs = [map_values(vv) for vv in n.inputs]
            n.outputs = [map_values(vv) for vv in n.outputs]
        bl.block_outputs = OrderedSet(map_values(vv) for vv in bl.block_outputs)

    if isinstance(gr_or_bl, Graph):
        for bl in gr_or_bl.blocks:
            process_block(bl)
    elif isinstance(gr_or_bl, Block):
        process_block(gr_or_bl)
    else:
        raise TypeError("replace_values works on Graph or Block objects")


## TODO: our should this be a method?
def make_dot(gr: Graph, format: str = "png", add_names: bool = False) -> "graphviz.Digraph":
    import graphviz

    dot = graphviz.Digraph(name="thunder_graph", format=format)

    block_idxes = {}

    value_idxes: dict[Value, int] = {}

    for i_bl, bl in enumerate(gr.blocks):
        block_idxes[bl] = i_bl
        with dot.subgraph(name=f"cluster_bl_{i_bl}") as sub_dot:
            for i_i, i in enumerate(bl.block_inputs):
                i_nr = len(value_idxes)
                value_idxes[i] = i_nr
                i_name = f"bi %{i_nr}"
                if add_names:
                    i.name = i_name
                v_color = "black" if i not in bl.block_outputs else "red"
                sub_dot.node(f"v {i_nr}", label=i_name, color=v_color)

            for i_n, n in enumerate(bl.nodes):
                label = n.i.opname
                if n.i.opname == "CALL_METHOD":
                    assert n.inputs[0].name is not None
                    label = "CM " + n.inputs[0].name
                elif n.i.opname == "CALL_FUNCTION" and n.inputs[0].name:
                    label = "CF " + n.inputs[0].name
                sub_dot.node(f"i {i_bl} {i_n}", label, shape="box")
                for o in n.outputs:
                    if o not in value_idxes:
                        o_nr = len(value_idxes)
                        value_idxes[o] = o_nr
                        o_name = o.name or f"%{o_nr}"
                        if add_names:
                            o.name = o_name
                        v_color = "black" if o not in bl.block_outputs else "red"
                        sub_dot.node(f"v {o_nr}", label=o_name, color=v_color)
                    else:
                        o_nr = value_idxes[o]
                    sub_dot.edge(f"i {i_bl} {i_n}", f"v {o_nr}", color="blue")
                if i_n > 0:
                    sub_dot.edge(f"i {i_bl} {i_n - 1}", f"i {i_bl} {i_n}")

    for i_bl, bl in enumerate(gr.blocks):
        for _, jt_bl in bl.nodes[-1].jump_targets:
            dot.edge(f"i {i_bl} {len(bl.nodes) - 1}", f"i {block_idxes[jt_bl]} {0}")
        for i in bl.block_inputs:
            i_idx = value_idxes[i]
            if isinstance(i, PhiValue):
                for v in i.values:
                    if v in value_idxes:
                        dot.edge(f"v {value_idxes[v]}", f"v {i_idx}", color="green")

        for i_n, n in enumerate(bl.nodes):
            for i in n.inputs:
                if i in value_idxes:
                    dot.edge(f"v {value_idxes[i]}", f"i {i_bl} {i_n}", color="blue")
                elif isinstance(i, PhiValue):
                    assert False, "This should be removed?"
                    for v in i.values:
                        if v in value_idxes:
                            dot.edge(f"v {value_idxes[v]}", f"i {i_bl} {i_n}", color="red")

    return dot


def clone_blocks(
    blocks_to_clone: list[Block], translation_dict: Optional[dict[GraphObject, GraphObject]] = None
) -> tuple[list[Block], dict[GraphObject, GraphObject]]:
    if translation_dict is None:
        translation_dict = {}

    blocks_todo = []
    for obl in blocks_to_clone:
        if obl not in translation_dict:
            bl = Block()
            translation_dict[obl] = bl
            blocks_todo.append(obl)

    for obl in blocks_todo:
        bl = assert_block(translation_dict[obl])
        bl.block_inputs = [i.clone(translation_dict=translation_dict) for i in obl.block_inputs]
        bl.block_outputs = OrderedSet(o.clone(translation_dict=translation_dict) for o in obl.block_outputs)
        bl.nodes = [n.clone(translation_dict=translation_dict) for n in obl.nodes]
    for obl in blocks_todo:
        bl = assert_block(translation_dict[obl])
        for js in obl.jump_sources:
            if js is None:
                bl.jump_sources.append(None)
            elif js in translation_dict:
                bl.jump_sources.append(assert_node(translation_dict[js]))

        for i in bl.block_inputs:
            i.post_process_clone(translation_dict=translation_dict)
    return [assert_block(translation_dict[bl]) for bl in blocks_to_clone], translation_dict


def _check_graph(gr: Graph) -> None:
    # some sanity checks for the values
    import collections

    phi_value_refs: dict[PhiValue, list[Union[Value, tuple[Value, Optional[Node]]]]] = collections.defaultdict(list)
    v: Value
    for bl in gr.blocks:
        known_values: set[Value] = set(bl.block_inputs)
        for i in bl.block_inputs:
            for v in i.phi_values:
                phi_value_refs[v].append(i)
        for n in bl.nodes:
            assert n.line_no is not None, f"{n}({n.inputs}) does not have a line number"
            n.block = bl
            for i in n.inputs:
                i_or_p = i
                while not (i_or_p in known_values or i_or_p.is_const or i_or_p.is_global):
                    if i_or_p.parent is not None:
                        i_or_p = i_or_p.parent
                    else:
                        raise RuntimeError(f"unknown value {repr(i_or_p)} needed in {n}")

            for o in n.outputs:
                known_values.add(o)
                # inplace modified values are not re-assigned. should they, likely: yes
                if o not in n.inputs:
                    for v in o.phi_values:
                        phi_value_refs[v].append((o, n))
                # else:
                #     print("inplace")
        for o in bl.block_outputs:
            is_attr = False
            o_or_parent = o
            while o_or_parent not in known_values and o_or_parent.parent is not None:
                o_or_parent = o_or_parent.parent
                is_attr = True
            if is_attr:
                for v in o.phi_values:
                    phi_value_refs[v].append((o, None))
            assert (
                o_or_parent in known_values or o_or_parent.is_const or o_or_parent.is_global
            ), f"{o_or_parent} (from {o}) unknown {known_values=}"

    for bl in gr.blocks:
        for i in bl.block_inputs:
            assert isinstance(i, PhiValue)
            assert len(i.jump_sources) == len(i.values)
            # assert i.block is bl
            pvr = phi_value_refs.get(i, [])
            assert len([v for v in i.values if not (v.is_function_arg or v.is_const or v.is_global)]) == len(
                pvr
            ), f"phi value {repr(i)} source count {len(i.values)} does not match sets {pvr}"
            if i in phi_value_refs:  # not for function args in first block
                del phi_value_refs[i]
            for v in i.values:
                assert i in v.phi_values, f"phi value {repr(i)} not in phi_values of {repr(v)}"

    assert not phi_value_refs, f"phi_values not found {phi_value_refs}"

    jump_targets: dict[Optional[Node], set[Block]] = {}
    jump_targets[None] = {gr.blocks[0]}  # function entry point

    for bl in gr.blocks:
        for n in bl.nodes[:-1]:
            assert not n.jump_targets
        n = bl.nodes[-1]
        if n.i.opname in {"RETURN_VALUE", "RAISE_VARARGS", "RERAISE"}:
            assert not n.jump_targets
        else:
            assert 1 <= len(n.jump_targets) <= 2, f"{n} should have one or two ump targets, but has {n.jump_targets}"
            jump_targets[n] = {jt for _, jt in n.jump_targets}
            assert len(n.jump_targets) == len(jump_targets[n])

    for bl in gr.blocks:
        for js in bl.jump_sources:
            js_jt = jump_targets[js]
            js_jt.remove(bl)

    assert not any(jump_targets.values()), f"{jump_targets} should be all empty"
    assert tuple(gr.blocks[0].jump_sources) == (None,), gr.blocks[0].jump_sources


def check_graph(gr: Graph) -> None:
    try:
        _check_graph(gr)
        cloned, _ = gr.clone()
        _check_graph(cloned)
    except BaseException:
        print()
        gr.summary(print_lines=True)
        raise
