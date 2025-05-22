import thunder

CHECK_VERSION = 3


def check_subsymbols(parent_bsym):
    if parent_bsym.sym.is_prim:
        # assert that there are no subsymbols?
        return
    known_proxies = {a.name: a for a in parent_bsym.flat_proxy_args}
    for bsym in parent_bsym.subsymbols:
        for a in bsym.flat_proxy_args:
            assert a.name in known_proxies, f"unknown proxy {a.name} is used in {bsym} args"
            assert known_proxies[a.name] is a, f"proxy name collision {a.name} in {bsym} args"
        for o in bsym.flat_proxy_outs:
            assert known_proxies.get(o.name, o) is o, f"known proxy or proxy name collision {o.name} in {bsym} outputs"
            known_proxies[o.name] = o
        check_subsymbols(bsym)
    for o in parent_bsym.flat_proxy_outs:
        assert known_proxies.get(o.name, o) is o, f"known proxy or proxy name collision {o.name} in {parent_bsym}"


def check_trace(trace, *, version=CHECK_VERSION):
    """checks a trace for consistency

    The check is versioned for the benefit of CI and other automated testing.

    As a user, don't pass a version to get all implemented checks.

    If you add new checks and do not fix all newly detected inconsistencies,
    bump the CHECK_VERSION and make your tests only apply to this latest version.

    Please do file issues for things that fail with the latest versions so we can
    catch up.
    """
    # TODO:
    # - args vs. unpack trivial
    # - args vs. flat_args in return
    known_proxies = {}
    for bsym in trace.bound_symbols:
        if (version >= 3) and bsym.sym == thunder.core.prims.unpack_sequence:
            coll = bsym.args[0].collection()
            assert len(coll) == len(bsym.output), f"unpack collection length mismatch {bsym}"
            for c, o in zip(coll, bsym.output):
                if o is None:  # unused outputs
                    continue
                if isinstance(c, thunder.Proxy):
                    assert c is o, f"mismatch in unpack collection: {c} {o} {bsym}"

        for a in bsym.flat_proxy_args:
            assert a.name in known_proxies, f"unknown proxy {a.name} is used in {bsym} args"
            assert known_proxies[a.name] is a, f"proxy name collision {a.name} in {bsym} args"
        for o in bsym.flat_proxy_outs:
            assert known_proxies.get(o.name, o) is o, f"known proxy or proxy name collision {o.name} in {bsym} outputs"
            known_proxies[o.name] = o
        check_subsymbols(bsym)

    tr = thunder.core.trace.from_trace(trace)
    with thunder.core.trace.tracectx(tr):
        for bsym in trace.bound_symbols:
            if bsym.sym.name.startswith("unpack") or bsym.sym.name in {"pack_buffer"}:
                continue
            res = bsym.sym(*bsym.args, **bsym.kwargs)

            def check_shape(x, y):
                if isinstance(x, thunder.TensorProxy) and y is not None:  # formal output can be none if unused
                    assert x.shape == y.shape, f"shape of proxy {y.name} recomputes to {x.shape} incorrectly in {bsym}"
                return x

            thunder.core.utils.safe_map_flat(check_shape, res, bsym.output)


class CheckedListOfTraces(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        cd = thunder.core.compile_data.get_compile_data()
        if cd.debug_options.check_traces is True:
            self._check_version = CHECK_VERSION
        elif not cd.debug_options.check_traces:
            self._check_version = 0
        else:
            self._check_version = cd.debug_options.check_traces

    def append(self, trace):
        check_trace(trace, version=self._check_version)
        super().append(trace)

    def extend(self, traces):
        for tr in traces:
            check_trace(tr, version=self._check_version)
        super().extend(traces)
