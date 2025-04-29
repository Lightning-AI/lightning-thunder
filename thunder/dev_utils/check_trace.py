def check_subsymbols(parent_bsym):
    if parent_bsym.sym.is_prim:
        # assert that there are no subsymbols?
        return
    known_proxies = {a.name: a for a in parent_bsym.flat_proxy_args}
    for bsym in parent_bsym.subsymbols:
        for a in bsym.flat_proxy_args:
            assert a.name in known_proxies, f"unknown proxy {a.name} is used in {bsym}"
            assert known_proxies[a.name] is a, f"proxy name collision {a.name} in {bsym}"
        for o in bsym.flat_proxy_outs:
            assert known_proxies.get(o.name, o) is o, f"known proxy or proxy name collision {a.name} in {bsym}"
            known_proxies[o.name] = o
        check_subsymbols(bsym)
    for o in parent_bsym.flat_proxy_outs:
        assert known_proxies.get(o.name, o) is o, f"known proxy or proxy name collision {a.name} in {parent_bsym}"


def check_trace(trace):
    """checks a trace for consistency"""
    # TODO:
    # - args vs. unpack trivial
    # - args vs. flat_args in return
    known_proxies = {}
    for bsym in trace.bound_symbols:
        for a in bsym.flat_proxy_args:
            assert a.name in known_proxies, f"unknown proxy {a.name} is used in {bsym}"
            assert known_proxies[a.name] is a, f"proxy name collision {a.name} in {bsym}"
        for o in bsym.flat_proxy_outs:
            assert known_proxies.get(o.name, o) is o, f"proxy name collision {a.name} in {bsym}"
            known_proxies[o.name] = o
        check_subsymbols(bsym)


class CheckedListOfTraces(list):
    def append(self, trace):
        check_trace(trace)
        super().append(trace)

    def extend(self, traces):
        for tr in traces:
            check_trace(tr)
        super().extend(traces)
