import dis
import inspect
import sys
import types
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

from thunder.core.script.graph import Graph, MROAwareObjectRef, Node, Value, insert_before, insert_after


def get_instruction(opname: str, arg: Optional[int]) -> dis.Instruction:
    i = dis.Instruction(
        opname=opname,
        opcode=dis.opmap.get(opname, -1),
        arg=arg,
        argval=None,
        argrepr="None",
        offset=-999,
        starts_line=None,
        is_jump_target=False,
    )
    return i


def undo_ssa(gr: "Graph") -> Tuple[List[Value], List[str], List[str], List[Any]]:
    consts: List[Any] = []
    names: List[str] = []

    def get_value(v: Value, n: Node, inpidx: Optional[int] = None) -> None:
        if n.i.opname == "CALL_METHOD" and inpidx == 1:
            return
        if v.is_const:
            idx = len(consts)
            consts.append(v.value)
            new_n = Node(i=get_instruction(opname="LOAD_CONST", arg=idx), outputs=[v], inputs=[])
            insert_before(new_n, n)
        elif isinstance(v.value, MROAwareObjectRef):
            # this works for attribs, but for methods? maybe have a pass eliminating/making explicit the super...
            get_value(v.value.obj, n)
        elif v.parent is not None:
            assert v.name is not None
            get_value(v.parent, n)
            if n.i.opname == "CALL_METHOD" and inpidx == 0:
                # print("###inputs", n.inputs, v, v in n.inputs)
                try:
                    idx = names.index(v.name)
                except ValueError:
                    idx = len(names)
                    names.append(v.name)
                new_n = Node(
                    i=get_instruction(opname="LOAD_METHOD", arg=idx),
                    outputs=[v, v.parent],
                    inputs=[v.parent],
                )
                insert_before(new_n, n)
            elif n.i.opname == "LOAD_ATTR":
                # print("###load attr", n.outputs, n.i.argval)
                pass
            else:
                assert v.name is not None
                try:
                    idx = names.index(v.name)
                except ValueError:
                    idx = len(names)
                    names.append(v.name)
                new_n = Node(
                    i=get_instruction(opname="LOAD_ATTR", arg=idx),
                    outputs=[v],
                    inputs=[v.parent],
                )
                insert_before(new_n, n)
        elif v.is_global:  # make binding the globals optional?
            if v.value not in consts:
                consts.append(v.value)
            idx = consts.index(v.value)
            new_n = Node(i=get_instruction(opname="LOAD_CONST", arg=idx), outputs=[v], inputs=[])
            insert_before(new_n, n)
        else:
            idx = local_vars.index(v)
            # assert idx >= 0
            new_n = Node(i=get_instruction(opname="LOAD_FAST", arg=idx), outputs=[v], inputs=[])
            insert_before(new_n, n)

    for bl in gr.blocks:
        for n in bl.nodes:
            n.block = bl

    local_vars: List[Value] = []
    lv_names = []

    def get_or_add_lv(v: Value, name: Optional[str] = None) -> int:
        try:
            idx = local_vars.index(v)
        except ValueError:
            idx = len(local_vars)
            local_vars.append(v)
            # handle name collisions...
            if name is None:
                name = v.name
            if name is None:
                name = f"_tmp_{idx}"
            else:
                name = name.replace(".", "_").replace("[", "").replace("]", "")
            if not name[:1].isalpha():
                name = "_" + name
            fullname = name
            suffix = 0
            while fullname in lv_names:
                suffix += 1
                fullname = f"{name}_{suffix}"
            lv_names.append(fullname)
            if v.name is None:  # TODO: or do this always?
                v.name = fullname
        return idx

    nodes_to_skip = set()

    def store_phi_values(o: Value, o_idx: int, last_n: Optional[Node]) -> Optional[Node]:
        phi_values_in_processing = set()

        def store_phi_values_inner(o: Value, o_idx: int, last_n: Optional[Node]) -> Optional[Node]:
            if o in phi_values_in_processing:
                # avoid loops
                return last_n
            phi_values_in_processing.add(o)
            for v in o.phi_values:
                idx2 = get_or_add_lv(v)
                # last_n = store_phi_values_inner(v, o_idx, last_n)
                new_n = Node(i=get_instruction(opname="LOAD_FAST", arg=o_idx), outputs=[o], inputs=[])
                nodes_to_skip.add(new_n)
                if last_n is None:
                    insert_before(new_n, gr.blocks[0].nodes[0])
                else:
                    insert_after(new_n, last_n)
                last_n = new_n
                new_n = Node(i=get_instruction(opname="STORE_FAST", arg=idx2), outputs=[], inputs=[o])
                nodes_to_skip.add(new_n)
                insert_after(new_n, last_n)
                last_n = new_n
            return last_n

        return store_phi_values_inner(o, o_idx, last_n)

    for v in gr.local_variables_at_start:
        if v is not None:
            get_or_add_lv(v)

    # inputs in phi values
    last_n = None
    # need to make a copy of the list because we're adding items to the list
    for idx, i in enumerate(local_vars[:]):
        last_n = store_phi_values(i, idx, last_n)

    names = []

    for bl in gr.blocks:
        jump_node = bl.nodes[-1]
        for n in bl.nodes[:]:
            processed_block_outputs = set()
            if n not in nodes_to_skip:
                for inpidx, i in enumerate(n.inputs):
                    get_value(i, n=n, inpidx=inpidx)
                last_n = n
                for o in n.outputs[::-1]:
                    idx = get_or_add_lv(o)
                    new_n = Node(
                        i=get_instruction(opname="STORE_FAST", arg=idx),
                        outputs=[],
                        inputs=[o],
                    )
                    assert last_n is not None
                    insert_after(new_n, last_n)
                    last_n = new_n
                    if o in bl.block_outputs:
                        processed_block_outputs.add(o)
                        last_n = store_phi_values(o, idx, last_n)
        if bl.nodes[-1].i.opname != "RETURN_VALUE":  # TODO Should the return block have outputs (probably not)
            for o in bl.block_outputs:
                if o not in processed_block_outputs:
                    get_value(o, n=jump_node)  # before the jump
                    idx = get_or_add_lv(o, name="bo")
                    new_n = Node(
                        i=get_instruction(opname="STORE_FAST", arg=idx),
                        outputs=[],
                        inputs=[o],
                    )
                    insert_before(new_n, n=jump_node)
                    store_phi_values(o, idx, new_n)

    return local_vars, lv_names, names, consts


# this function is taken from PyTorch Dynamo (c) 2022 by Facebook/Meta licensed
# as per https://github.com/pytorch/pytorch/blob/master/LICENSE
def linetable_writer(first_lineno: int) -> Tuple[List[int], Callable, Callable]:
    """Used to create typing.CodeType.co_linetable See
    https://github.com/python/cpython/blob/main/Objects/lnotab_notes.txt This is the internal format of the line number
    table if Python >= 3.10."""
    assert sys.version_info >= (3, 9)
    linetable: List[int] = []
    lineno = first_lineno
    lineno_delta = 0
    byteno = 0

    def _update(byteno_delta: int, lineno_delta: int) -> None:
        while byteno_delta != 0 or lineno_delta != 0:
            byte_offset = max(0, min(byteno_delta, 254))
            line_offset = max(-127, min(lineno_delta, 127))
            assert byte_offset != 0 or line_offset != 0
            byteno_delta -= byte_offset
            lineno_delta -= line_offset
            linetable.extend((byte_offset, line_offset & 0xFF))

    def update(lineno_new: int, byteno_new: int) -> None:
        nonlocal lineno, lineno_delta, byteno
        byteno_delta = byteno_new - byteno
        byteno = byteno_new
        _update(byteno_delta, lineno_delta)
        lineno_delta = lineno_new - lineno
        lineno = lineno_new

    def end(total_bytes: int) -> None:
        _update(total_bytes - byteno, lineno_delta)

    return linetable, update, end


def generate_function(gr: "Graph") -> Callable:
    orig_gr = gr
    gr, _ = gr.clone()

    local_vars, lv_names, names, consts = undo_ssa(gr)
    assert len(local_vars) == len(lv_names)

    NodeKey = Union[Node, Tuple[Node, bool]]
    instruction_sizes: Dict[NodeKey, int] = {}

    def build_address_map() -> Dict[NodeKey, int]:
        # Key either <Node> (for jump nodes and jump=True)
        #     or (<Node>, False) for non-jump in conditional jump
        address_map: Dict[NodeKey, int] = {}
        ctr = 0
        for bl in gr.blocks:
            # assumes first block is function start
            for n in bl.nodes:
                address_map[n] = ctr
                ctr += instruction_sizes.get(n, 1)
                if len(n.jump_targets) == 2:  # implicit unconditional jump
                    ctr += instruction_sizes.get((n, False), 1)
        return address_map

    def make_bc() -> Tuple[List[int], bool]:
        bc = []

        def write_extended_args(node_key: NodeKey, arg: int) -> bool:
            # returns if instruction size has changed
            instruction_size = instruction_sizes.get(node_key, 1)
            if arg > 0x_FF_FF_FF or instruction_size == 4:
                instruction_size = 4
                bc.append(dis.opmap["EXTENDED_ARG"])
                bc.append(arg >> 24)
            if arg > 0x_FF_FF or instruction_size >= 3:
                instruction_size = max(instruction_size, 3)
                bc.append(dis.opmap["EXTENDED_ARG"])
                bc.append((arg >> 16) & 0xFF)
            if arg > 0x_FF or instruction_size >= 2:
                instruction_size = max(instruction_size, 2)
                bc.append(dis.opmap["EXTENDED_ARG"])
                bc.append((arg >> 8) & 0xFF)
            else:
                instruction_size = 1

            if instruction_size != instruction_sizes.get(node_key, 1):
                instruction_sizes[node_key] = instruction_size
                return True
            return False

        changed_size = False
        line_no = None
        for bl in gr.blocks:
            jump_node = None
            for n in bl.nodes:
                opcode = n.i.opcode
                if opcode is None or opcode == -1:  # Todo: opcode is typed int in dis.Instruction, remove None here?
                    opcode = dis.opmap[n.i.opname]
                assert opcode is not None, f"{n} has invalid opcode"
                # source range instead for 3.11?
                if n.line_no is not None and n.line_no != line_no:
                    linetable_update(n.line_no, address_map[n] * 2)  # byte offset for Python 3.10, too...
                    line_no = n.line_no
                if opcode in dis.hasjabs:
                    arg = address_map[n.jump_targets[-1][1].nodes[0]]
                elif opcode in dis.hasjrel:
                    # TODO forward, backward
                    arg = address_map[n.jump_targets[-1][1].nodes[0]] - address_map[n] - 1
                else:
                    arg_ = n.i.arg
                    arg = 0 if arg_ is None else arg_

                changed_size |= write_extended_args(n, arg)

                bc.append(opcode)
                bc.append(arg & 0x_FF)
                if len(n.jump_targets) > 1:
                    jump_node = n
            if jump_node is not None:
                assert len(jump_node.jump_targets) == 2
                jarg = address_map[jump_node.jump_targets[0][1].nodes[0]]
                changed_size |= write_extended_args((jump_node, False), jarg)
                i = get_instruction(opname="JUMP_ABSOLUTE", arg=jarg & 0xFF)
                bc.append(i.opcode)
                assert i.arg is not None
                bc.append(i.arg)
        return bc, not changed_size

    done = False
    while not done:
        linetable, linetable_update, linetable_end = linetable_writer(gr.source_start_line)
        address_map = build_address_map()
        bc, done = make_bc()

    linetable_end(len(bc))
    linetable_bytes = bytes(linetable)
    bc_bytes = bytes(bc)

    lv_at_start = [v for v in gr.local_variables_at_start if v is not None]
    co_argcount = gr.co_argcount
    co_posonlyargcount = gr.co_posonlyargcount
    co_kwonlyargcount = gr.co_kwonlyargcount
    co_nlocals = len(local_vars)
    # TODO: actually track the stack size when doing codegen (for optimizations)
    co_stacksize = max(max(len(n.inputs), len(n.outputs)) for n in gr.nodes())
    co_flags = gr.co_flags
    co_codestring = bc_bytes
    co_consts = tuple(consts)
    co_names = tuple(names)
    co_varnames = tuple(lv_names)
    co_filename = f"<thunder-generated-{id(gr)}>"
    co_name = f"thunder_{id(gr)}"  # nicer name?
    co_firstlineno = gr.source_start_line
    co_linetable = linetable_bytes  # XXX
    co_freevars = ()
    co_cellvars = ()

    c = types.CodeType(
        co_argcount,  # int
        co_posonlyargcount,  # int
        co_kwonlyargcount,  # int
        co_nlocals,  # int
        co_stacksize,  # int
        co_flags,  # int
        co_codestring,  # bytes
        co_consts,  # tuple
        co_names,  # tuple
        co_varnames,  # tuple
        co_filename,  # string
        co_name,  # string
        co_firstlineno,  # integer
        co_linetable,  # bytes
        co_freevars,  # tuple
        co_cellvars,  # tuple
    )

    # types.FunctionType(code, globals, name=None, argdefs=None, closure=None)
    func = types.FunctionType(c, {}, argdefs=tuple(gr.func_defaults))
    func.__kwdefaults__ = gr.func_kwdefaults
    func._gr = orig_gr

    # simple cache hack
    mtime = None  # this signals that the cache should not be invalidated(!)
    lines = gr.source_lines
    size = len("".join(lines))
    inspect.linecache.cache[co_filename] = size, mtime, lines, co_filename

    return func
