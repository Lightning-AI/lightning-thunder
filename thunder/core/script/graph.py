# This is a "TorchScript-like" graph representation of Python IR.
# The idea is that blocks are "simple blocks" in terms of the code flow graph,
# i.e. without branches
import copy
import inspect

from .python_ir_data import jump_instructions, stack_effect_detail, unconditional_jump_names


class NULL:
    """marker for non-existant object."""

    pass


def _make_set(s):
    if isinstance(s, set):
        return s
    return set(s)


class MROAwareObjectRef:  # or as they call it super
    def __init__(self, obj, start_klass=None):
        self.obj = obj
        self.start_klass = start_klass

    def __getattr__(self, name):
        print("###", self.obj, self.start_klass, name)
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
        node=None,
        nr=None,
        typ=None,
        value=None,
        name=None,
        parent=None,
        is_global=False,
        is_const=False,
        is_function_arg=False,
    ):
        self.node = node
        self.nr = nr
        self.typ = typ if typ is not None or value is None else type(value)
        self.value = value
        self.name = name
        self.parent = parent
        self.is_global = is_global
        self.is_const = is_const
        self.is_function_arg = is_function_arg
        self.phi_values = []

    def clone(self, translation_dict=None):
        # clones a node, including (recursively) parent nodes
        # uses translation_dict to look up parent node
        # updates translation_dict
        # does not register phi_values on the clone
        # always clone parents?
        if translation_dict is None:
            translation_dict = {}
        if self in translation_dict:
            return translation_dict[self]
        parent = self.parent
        if parent:
            if parent in translation_dict:
                parent = translation_dict[parent]
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

    def __str__(self):
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
        return f"""{type(self).__name__}({' '.join(parts)})"""

    def __repr__(self):
        return f"{super().__repr__()[:-1]} {self}>"


class PhiValue(Value):
    # node?
    def __init__(self, values, jump_sources, block, _unfinished_clone=False):
        super().__init__()
        self._unfinished_clone = _unfinished_clone
        self.block = block
        self._set_values_jump_sourcess(values, jump_sources)

    def _set_values_jump_sourcess(self, values, jump_sources):
        assert len(values) == len(jump_sources)
        self.values = list(values)
        if not self._unfinished_clone:
            for v in self.values:
                if v is not None:
                    v.phi_values.append(self)
        self.jump_sources = jump_sources[:]

    def clone(self, translation_dict=None):
        # due to loops in the Graph, this is complicated:
        # we do not translate values or jump_sources here, but do
        # translate blocks.
        if translation_dict is None:
            translation_dict = {}
        if self in translation_dict:
            return translation_dict[self]
        v = PhiValue(self.values, self.jump_sources, translation_dict[self.block], _unfinished_clone=True)
        translation_dict[self] = v
        return v

    def post_process_clone(self, *, translation_dict):
        assert self._unfinished_clone
        self._unfinished_clone = False
        self._set_values_jump_sourcess(
            [translation_dict.get(v, v) for v in self.values],
            [translation_dict.get(js, js) for js in self.jump_sources],
        )

    def add_missing_value(self, v, idx=None, jump_source=None):  # None: append
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

    def remove_value(self, v):
        idx = self.values.index(v)
        v.phi_values.remove(self)
        del self.values[idx]
        del self.jump_sources[idx]


def unify_values(values, jump_sources, bl, all_predecessors_done=True):
    if all_predecessors_done:
        if len(values) == 1:
            return values[0]
        val = values[0]
        if all(v is val for v in values[1:]):
            return val
        # different values
    return PhiValue(values, jump_sources, bl)


# A node corresponds to one Python bytecode instruction given in .i
# it has Values as .inputs and .outputs
class Node:
    def __init__(self, *, i=None, inputs=None, outputs=None, line_no=None):
        self.i = i
        self.inputs = inputs if inputs is not None else []
        self.outputs = outputs if inputs is not None else []
        self.jump_targets = []
        self.line_no = line_no
        self.block = None

    def clone(self, translation_dict=None):
        """.block of the clone will be None if block is not in translation dict."""
        if translation_dict is None:
            translation_dict = {}
        if self in translation_dict:
            return translation_dict[self]
        inputs = [i.clone(translation_dict=translation_dict) for i in self.inputs]
        outputs = [o.clone(translation_dict=translation_dict) for o in self.outputs]
        i = copy.copy(self.i)
        n2 = Node(i=i, inputs=inputs, outputs=outputs, line_no=self.line_no)
        n2.jump_targets = [(se, translation_dict.get(bl, bl)) for se, bl in self.jump_targets]
        n2.block = translation_dict.get(self.block)
        translation_dict[self] = n2
        return n2

    def set_jump_target(self, jt, idx=None):
        is_jump = (self.i.opname not in unconditional_jump_names) or (idx == 1) or (idx is None and self.jump_targets)
        jt_plus = (stack_effect_detail(self.i.opname, self.i.arg, jump=False), jt)
        if idx is None:
            assert len(self.jump_targets) <= 1
            self.jump_targets.append(jt_plus)
        else:
            old_jt = self.jump_targets[idx][1]
            old_jt.jump_sources.remove(self)
            self.jump_targets[idx] = jt_plus
        jt.jump_sources.append(self)

    def __str__(self):
        # i.i.offset // 2, i.i.opname, i.i.arg, "(", i.i.argval, ")"
        if self.i.opname in {"CALL_METHOD", "CALL_FUNCTION"}:
            return f"{self.i.opname}({self.inputs})"
        return f"{self.i.opname} {self.i.arg} ({self.i.argval})"  # str(self.i)

    def __repr__(self):
        return f"{super().__repr__()[:-1]} {self}>"


# Blocks have the first instruction (only) as the jump target
# (or the function entry point)
# Blocks always have a single final instruction that jumps (or RETURN)
# conditional jumps (including e.g. FOR_ITER) always have the non-jumping
# target first and then the jumping target.
# The jump targets are other blocks and are atributes of the jump instruction.
class Block:
    def __init__(self, is_ssa=True):
        self.is_ssa = is_ssa
        self.jump_sources = []
        self.nodes = []
        self.block_inputs = []
        self.block_outputs = set()

    def __str__(self):
        return "\n".join([f"  Block (reached from {self.jump_sources})"] + ["    " + str(n) for n in self.nodes])

    def __repr__(self):
        return f"{super().__repr__()[:-1]} {self}>"

    def insert_node(self, n, insert_after=None, insert_before=None):
        assert n.block is None
        if insert_after is None and insert_before is None:
            if self.is_ssa:
                raise ValueError("need to supply insert_after or insert_before")
            else:
                self.nodes.append(n)
                # validity checks? (also below)
                n.block = self
                return
        elif insert_after is not None and insert_before is not None:
            raise ValueError("only one of insert_after or insert_before can be supplied")
            # this is the usual case.
            # in the pre-ssa graph, both None mean to insert at the end.
            assert insert_after is not None or insert_before is not None

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
    def __init__(self, blocks=None):
        self.blocks = [] if blocks is None else blocks

    def __str__(self):
        return "\n".join(["Graph of"] + [str(b) for b in self.blocks])

    def __repr__(self):
        return f"{super().__repr__()[:-1]} {self}>"

    def nodes(self):
        for b in self.blocks:
            yield from b.nodes

    def ensure_links(self):
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

    def print(self):
        value_counter = 1
        print(self.local_variables_at_start)
        for bl in self.blocks:
            for n in bl.i:
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


def insert_before(new_n, n):
    idx = n.block.nodes.index(n)
    n.block.nodes.insert(idx, new_n)
    new_n.block = n.block


def insert_after(new_n, n):
    idx = n.block.nodes.index(n)
    n.block.nodes.insert(idx + 1, new_n)
    new_n.block = n.block


def replace_values(gr_or_bl, value_map, follow_phi_values=False):
    ### Replacing a value:
    # - as inputs/outputs of nodes
    # - value.parent for other values
    # - phi nodes
    # - graph input (?) / initial vars

    def map_values(v):
        if v in value_map:
            if follow_phi_values:
                for pv in v.phi_values[:]:
                    assert len(pv.values) == len(pv.jump_sources)
                    pv.remove_value(v)
                    pv.add_missing_value(value_map[v])
                    assert len(pv.values) == len(pv.jump_sources)
            return value_map[v]
        if isinstance(v.value, MROAwareObjectRef):
            v.value.obj = map_values(v.value.obj)
        if v.parent is not None:
            v.parent = map_values(v.parent)
        if isinstance(v, PhiValue):
            assert len(v.values) == len(v.jump_sources)
            for ov in v.values:
                nv = map_values(ov)
                v.remove_value(ov)
                v.add_missing_value(nv)
            assert len(v.values) == len(v.jump_sources)
        return v

    def process_block(bl):
        bl.block_inputs = [map_values(vv) for vv in bl.block_inputs]
        for n in bl.nodes:
            n.inputs = [map_values(vv) for vv in n.inputs]
            n.outputs = [map_values(vv) for vv in n.outputs]
        bl.block_outputs = {map_values(vv) for vv in bl.block_outputs}

    if isinstance(gr_or_bl, Graph):
        for bl in gr_or_bl.blocks:
            process_block(bl)
    elif isinstance(gr_or_bl, Block):
        process_block(gr_or_bl)
    else:
        raise TypeError("replace_values works on Graph or Block objects")


## TODO: our should this be a method?
def make_dot(gr, format="png", add_names=False):
    import graphviz

    dot = graphviz.Digraph(name="thunder_graph", format=format)

    block_idxes = {}

    value_idxes = {}

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
                    print("oops", repr(i))
                    for v in i.values:
                        if v in value_idxes:
                            dot.edge(f"v {value_idxes[v]}", f"i {i_bl} {i_n}", color="red")

    return dot


def clone_blocks(blocks_to_clone: list[Block], translation_dict=None):
    assert all(bl.is_ssa for bl in blocks_to_clone)
    if translation_dict is None:
        translation_dict = {}

    blocks_todo = []
    for obl in blocks_to_clone:
        if obl not in translation_dict:
            bl = Block()
            translation_dict[obl] = bl
            blocks_todo.append(obl)

    for obl in blocks_todo:
        bl = translation_dict[obl]
        bl.block_inputs = [i.clone(translation_dict=translation_dict) for i in obl.block_inputs]
        bl.block_outputs = {o.clone(translation_dict=translation_dict) for o in obl.block_outputs}
        bl.nodes = [n.clone(translation_dict=translation_dict) for n in obl.nodes]
    for obl in blocks_todo:
        bl = translation_dict[obl]
        bl.jump_sources = [translation_dict[js] for js in obl.jump_sources if js in translation_dict]
        for i in bl.block_inputs:
            i.post_process_clone(translation_dict=translation_dict)
    return [translation_dict[bl] for bl in blocks_to_clone], translation_dict


def check_graph(gr):
    # some sanity checks for the values
    import collections

    phi_value_refs = collections.defaultdict(list)
    for bl in gr.blocks:
        known_values = set(bl.block_inputs)
        for i in bl.block_inputs:
            for v in i.phi_values:
                phi_value_refs[v].append(i)
        for n in bl.nodes:
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
                else:
                    print("inplace")
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

    jump_targets = {}
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

    assert not any(jump_targets.values())
