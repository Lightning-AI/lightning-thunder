# CapabilityBasedParitioner returns a GraphModule where `fused_*` represent the subgraphs
# that should go to `thunder` and the forward of this graph module should be passed to `torch.compile` (after removing thunder bits)
# Example -
# class GraphModule(torch.nn.Module):
#     def forward(self, L_x_: "f32[2]"):
#         l_x_ = L_x_

#         # No stacktrace found for following nodes
#         _enter_autocast = torch.amp.autocast_mode._enter_autocast('cpu', None, True, None)

#          # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:181 in func, code: return torch.matmul(x, y)
#         fused_0: "bf16[]" = self.fused_0(l_x_);  l_x_ = None

#         # No stacktrace found for following nodes
#         _exit_autocast = torch.amp.autocast_mode._exit_autocast(_enter_autocast);  _enter_autocast = _exit_autocast = None
#         return (fused_0,)

#     class fused_0(torch.nn.Module):
#         def forward(self, l_x_: "f32[2]"):
#              # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:177 in func, code: x = x + 2
#             x: "f32[2]" = l_x_ + 2;  l_x_ = None

#              # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:179 in func, code: z = torch.ones(3, 3)
#             z: "f32[3, 3]" = torch.ones(3, 3);  z = None

#              # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:180 in func, code: y = torch.sin(x)
#             y: "f32[2]" = torch.sin(x)

#              # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:181 in func, code: return torch.matmul(x, y)
#             matmul: "bf16[]" = torch.matmul(x, y);  x = y = None
#             return matmul


def capability_partitioner_splitter(gm, sample_args):
    gm_copy = copy.deepcopy(gm)
    op_support = ThunderOperatorSupport(gm_copy)
    partitioner = CapabilityBasedPartitioner(gm_copy, op_support)
    fused_partition = partitioner.partition_and_fuse()
    gm_copy.print_readable()
    return gm_copy


# Splitter with _SplitterBase.

# class GraphModule(torch.nn.Module):
#     def forward(self, L_x_: "f32[2]"):
#         l_x_ = L_x_

#          # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:176 in func, code: x = x + 2
#         x: "f32[2]" = l_x_ + 2;  l_x_ = None

#         # No stacktrace found for following nodes
#         _enter_autocast = torch.amp.autocast_mode._enter_autocast('cpu', None, True, None)

#          # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:178 in func, code: y = torch.sin(x)
#         y: "f32[2]" = torch.sin(x)

#          # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:179 in func, code: return torch.matmul(x, y)
#         matmul: "bf16[]" = torch.matmul(x, y);  x = y = None

#         # No stacktrace found for following nodes
#         _exit_autocast = torch.amp.autocast_mode._exit_autocast(_enter_autocast);  _enter_autocast = _exit_autocast = None
#         return (matmul,)

# Got 1 acc subgraphs and 2 non-acc subgraphs
# class GraphModule(torch.nn.Module):
#     def forward(self, l_x_: "f32[2]"):
#         # No stacktrace found for following nodes
#         _run_on_cpu_0 = self._run_on_cpu_0();  _run_on_cpu_0 = None
#         _run_on_acc_1 = self._run_on_acc_1(l_x_);  l_x_ = None
#         _run_on_cpu_2 = self._run_on_cpu_2(_run_on_acc_1);  _run_on_acc_1 = None
#         return (_run_on_cpu_2,)

#     class _run_on_cpu_0(torch.nn.Module):
#         def forward(self):
#             # No stacktrace found for following nodes
#             _enter_autocast = torch.amp.autocast_mode._enter_autocast('cpu', None, True, None)
#             _exit_autocast = torch.amp.autocast_mode._exit_autocast(_enter_autocast);  _enter_autocast = _exit_autocast = None
#             return ()

#     class _run_on_acc_1(torch.nn.Module):
#         def forward(self, l_x_: "f32[2]"):
#              # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:176 in func, code: x = x + 2
#             x: "f32[2]" = l_x_ + 2;  l_x_ = None
#             return x

#     class _run_on_cpu_2(torch.nn.Module):
#         def forward(self, x: "f32[2]"):
#              # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:178 in func, code: y = torch.sin(x)
#             y: "f32[2]" = torch.sin(x)

#              # File: /home/kkalambarkar/lightning-thunder/scratchpad/test.py:179 in func, code: return torch.matmul(x, y)
#             matmul: "bf16[]" = torch.matmul(x, y);  x = y = None
#             return matmul


class GraphModuleSplitter(torch.fx.passes.splitter_base._SplitterBase):
    def starter_nodes(self):
        """
        Finds nodes that consume module inputs or get_attr nodes.
        """
        starter_cpu_nodes: NodeSet = set()
        starter_acc_nodes: NodeSet = set()

        for node in self.module.graph.nodes:
            if node.op not in {"placeholder", "get_attr"}:
                continue
            for user in node.users:
                if user in self.acc_nodes:
                    starter_acc_nodes.add(user)
                else:
                    starter_cpu_nodes.add(user)

        for node in self.module.graph.nodes:
            if node.op in {"output", "placeholder", "get_attr"}:
                continue

            if len(self.deps[node]) == 0:
                if node in self.acc_nodes:
                    starter_acc_nodes.add(node)
                else:
                    starter_cpu_nodes.add(node)

        return starter_cpu_nodes, starter_acc_nodes


def splitter(self, gm, sample_input):
    """
    This function splits the graph provided by Dynamo
    if it contains any operation or construct that is not supported by thunder.
    For the unsupported subgraph, it is passed to inductor.
    """
    from thunder import jit

    # Setup the splitter class
    settings = torch.fx.passes.splitter_base._SplitterSettingBase(allow_non_tensor=True)
    splitter = GraphModuleSplitter(gm, sample_input, operator_support=ThunderOperatorSupport(gm), settings=settings)
    gm.print_readable()
    # Call the splitter to split GraphModule.
    split_module = splitter()
    split_module.print_readable()
    compiled_funcs = []
    for node in split_module.graph.nodes:
        if node.name.startswith("_run_on_acc_"):
            graph_module = getattr(split_module, node.name)
            jit_fn = self.thunder_jit(graph_module)
            setattr(split_module, node.name, jit_fn)
            compiled_funcs.append(jit_fn)
        if node.name.startswith("_run_on_cpu_") or node.name.startswith("_run_on_gpu_"):
            graph_module = getattr(split_module, node.name)
            jit_fn = torch.compile(graph_module, backend="inductor")
            setattr(split_module, node.name, jit_fn)
            compiled_funcs.append(jit_fn)

    self.subgraph_infos.append(SubgraphInfo(gm, True, compiled_funcs, [], split_module))
    # split_module.print_readable()
    return split_module
