import torch
from torch.testing import make_tensor, assert_close

import thunder as lc
from thunder.core.symbol import BoundSymbol


# NOTE This test modifies the global executor map, so it technically should not
#   be run in parallel with other tests
def test_custom_operator_extension():
    # Defines a silly sample operator executor extension for testing purposes
    # This operator executor executors torch.add, but only when alpha is None
    def torch_add_check(a, b, *, alpha=None) -> bool:
        if alpha is not None:
            return False

        return True

    def torch_add_impl(a, b, *, alpha=None):
        return torch.add(a, b)

    _op_map = {
        "torch.add": ("testex_torch_add", torch_add_check, torch_add_impl),
    }

    # NOTE This doesn't add itself to the default executors to minimize this test's
    #   impact on global state
    lc.add_operator_executor("testex", _op_map, add_to_default_executors=False)

    myexecutors_list = ("testex",) + lc.list_default_executors()

    device = "cpu"
    tdtype = torch.float32
    a = make_tensor((2, 1), device=device, dtype=tdtype)
    b = make_tensor((2, 2), device=device, dtype=tdtype)

    # Tests that the custom executor is called as expected when adding
    #   tensors without alpha
    def foo(a, b):
        return a + b

    cfoo = lc.compile(foo, executors_list=myexecutors_list)

    lc_result = cfoo(a, b)
    torch_result = foo(a, b)

    assert_close(lc_result, torch_result)

    traces = lc.last_traces(cfoo)
    extrace = traces[-1]

    has_custom_symbol = False
    for bsym in extrace.bound_symbols:
        if bsym.sym.name == "testex_torch_add":
            has_custom_symbol = True
            break

    assert has_custom_symbol

    # Tests that the custom executor is not called when alpha is specified
    def bar(a, b):
        return torch.add(a, b, alpha=2)

    cbar = lc.compile(bar, executors_list=myexecutors_list)

    lc_result = cbar(a, b)
    torch_result = bar(a, b)

    assert_close(lc_result, torch_result)

    traces = lc.last_traces(cbar)
    extrace = traces[-1]

    has_custom_symbol = False
    for bsym in extrace.bound_symbols:
        if bsym.sym.name == "testex_torch_add":
            has_custom_symbol = True
            break

    assert not has_custom_symbol

    # Tests that multiple operators in a contiguous region works as expected
    def caz(a, b):
        return a + b + b + a + torch.add(a, a)

    ccaz = lc.compile(caz, executors_list=myexecutors_list)

    lc_result = ccaz(a, b)
    torch_result = caz(a, b)

    assert_close(lc_result, torch_result)

    traces = lc.last_traces(ccaz)
    extrace = traces[-1]

    count_custom_symbols = 0
    for bsym in extrace.bound_symbols:
        if bsym.sym.name == "testex_torch_add":
            count_custom_symbols += 1

    assert count_custom_symbols == 5
