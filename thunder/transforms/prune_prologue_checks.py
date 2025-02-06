import thunder
from thunder.core import prims as prims


class PrunePrologueChecks(thunder.core.transform_common.Transform):
    """A Transform to prune Prologue checks

    This transform removes prologue checks and can be applied when the user
    controls the environment enough to ensure that these checks would always
    succeed. By default, we remove all checks that use the module.
    """

    def __init__(self, prune_all_checks=False):
        self.prune_all_checks = prune_all_checks

    def transform_traces_pre_prologue(self, prologue_trace, compute_trace, epilogue_trace, **kwargs):
        def is_check(bsym):
            return bsym.sym in {
                prims.check_tensor_shape_and_metadata,
                prims.check_string_value,
                prims.check_number_type_and_value,
                prims.check_len,
                prims.check_literal_like,
            }

        if not self.prune_all_checks:
            bsyms_to_skip = set()
            module_member_names = set()

            for bsym in prologue_trace.bound_symbols:
                if bsym.sym in {prims.unpack_trivial, prims.unpack_cache_info, prims.python_return}:
                    # These don't have inputs but need to be skipped to not trigger false positives
                    # python_return may have no inputs
                    continue
                if bsym.sym is prims.unpack_function_obj:
                    for o in bsym.flat_proxy_outs:
                        module_member_names.add(o.name)
                    continue

                input_names = {i.name in module_member_names for i in bsym.flat_proxy_args}
                if all(input_names):
                    # This has the special case of no proxy inputs, which is the case for unpack_function_obj,
                    # the root of module_member_names
                    assert input_names, f"unexpected symbol {bsym.sym.name} without inputs"
                    for o in bsym.flat_proxy_outs:
                        module_member_names.add(o.name)
                    if is_check(bsym):
                        bsyms_to_skip.add(bsym)

            def should_skip_bsym(bsym):
                return bsym in bsyms_to_skip

        else:
            should_skip_bsym = is_check

        new_prologue_trace = thunder.core.trace.from_trace(prologue_trace)
        for bsym in prologue_trace.bound_symbols:
            if not should_skip_bsym(bsym):
                new_prologue_trace.bound_symbols.append(bsym.from_bsym())

        new_prologue_trace = thunder.core.transform_common.dce(new_prologue_trace)
        new_prologue_trace.set_provenance(thunder.core.trace.TraceProvenance(f"{self.__class__.__name__}"))

        return new_prologue_trace, compute_trace, epilogue_trace
