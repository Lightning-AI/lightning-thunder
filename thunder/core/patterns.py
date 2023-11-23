from typing import Any
from collections.abc import Callable
from functools import partial

from thunder.core.trace import TraceCtx
from thunder.core.symbol import BoundSymbol
from thunder.core.codeutils import get_siginfo
from thunder.core.utils import ProxyDict, producers

#
# Classes and functions for matching patterns
#
# See dev_tutorials/patterns.ipynb for more information on patterns.


# TODO Consider promoting this for more general use
# A helper to easily access the inputs to a function by their names in the signature
def bind_names(bsym: BoundSymbol) -> Any:
    return get_siginfo(bsym.sym, bsym.args, bsym.kwargs, _make_named_inputs=True)


# Translates a trace into the line numbers of each boundsymbol's immediate ancestors
#   The ancestors are represented by their corresponding line number
def numbered_ancestors(trc: TraceCtx) -> list[set[int]]:
    producer_map: ProxyDict = producers(trc, _map_to_numbers=True)

    ancestors: list[set[int]] = []

    bsym: BoundSymbol
    for bsym in trc.bound_symbols:
        bsym_ancestors: set[int] = {producer_map[inp] for inp in bsym.flat_proxy_args}
        ancestors.append(bsym_ancestors)

    return ancestors


# TODO Consider support ctx_updates that aren't dicts
# Finds all BoundSymbols between start_idx and start_idx + window, inclusive, for which matcher returns True.
#   The matches are returned as a list of (idx, BoundSymbol) tuples.
def match_all(
    trc: TraceCtx,
    matcher: Callable,
    *,
    ancestors: None | list[set[int]] = None,
    restricted_indices: set[int],
    matched_indices: None | set[int] = None,
    start_idx: int,
    window: int,
) -> list[tuple[int, BoundSymbol, dict]]:
    idx: int
    bsym: BoundSymbol
    matches: list[tuple[int, BoundSymbol, dict]] = []

    for idx, bsym in enumerate(trc.bound_symbols[start_idx:]):
        if window >= 0 and idx > window:
            break

        actual_idx: int = start_idx + idx
        if actual_idx in restricted_indices:
            continue

        matched: bool
        ctx_update: dict
        matched, ctx_update = matcher(bsym)

        if matched:
            # Checks that this bsym can be reordered next to existing matched operations
            # NOTE This currently happens after matching as a performance heuristic,
            #   but it could also happen before querying the matcher
            # NOTE When idx is zero then the bsym doesn't need to be reordered
            if idx > 0 and len(matched_indices) > 0:
                immediate_ancestors: set[int] = ancestors[actual_idx]
                cur: set[int] = immediate_ancestors - matched_indices
                oldest_ancestor: int = min(matched_indices)

                can_reorder: bool = True
                while len(cur) > 0 and max(cur) >= oldest_ancestor:
                    # Detects a relationship where another op is between the current matched operations and
                    #   this potential match
                    if len(cur & matched_indices) > 0:
                        can_reorder = False
                        break

                    # Updates the nodes under evaluation to the next generation
                    cur = set().union(*(ancestors[x] for x in cur))

                if not can_reorder:
                    continue

            matches.append((actual_idx, bsym, ctx_update))

    return matches


# Defines a pattern, and supports matching that pattern in a trace
# To use this class, add one or more matchers by calling the match method,
#   then call the pattern on the trace, where it will return a list of lists of
#   BoundSymbols matching the specified the pattern.
class Pattern:
    def __init__(self):
        self.matchers: list[tuple[Callable, int, int]] = []

    # TODO Extend with the ability to define a window?
    def match(self, matcher: Callable, *, min_times: int = 1, max_times: int = 1):
        self.matchers.append((matcher, min_times, max_times))

    # TODO Should we consider supporting multiple matches with a window size != 1? That could miss some
    #   valid matches if it greedily acquired the multiple matches
    # TODO There's probably a more elegant way to model this than using a helper function and a special-case
    #   for matching the same pattern multiple times
    # Matches the same pattern multiple times -- assumes a window of 1 (no gaps between operations)
    def _match_multiple(
        self,
        trc: TraceCtx,
        *,
        previously_matched: list[BoundSymbol],
        match_ctx: dict,
        restricted_indices: set[int],
        depth: int,
        idx: int,
        already_matched: int = 0,
    ) -> None | tuple[list[tuple[int, BoundSymbol]], dict]:
        matcher, min_times, max_times = self.matchers[depth]

        assert min_times >= 0, "Setting the minimum number of matches to a negative value is invalid"
        assert max_times != 0, "Setting the maximum number of matches to zero is invalid"

        nmatches: int = 0
        matches: list[tuple[int, BoundSymbol]] = []
        while True:
            cidx: int = idx + nmatches

            # Stops greedily matching if the operation was already merged into another pattern
            if cidx in restricted_indices:
                break

            pmatcher: Callable = partial(matcher, previously_matched=previously_matched, match_ctx=match_ctx)
            match: list[tuple[int, BoundSymbol, dict]] = match_all(
                trc, pmatcher, start_idx=cidx, window=0, restricted_indices=restricted_indices
            )

            if len(match) == 0:
                break

            assert len(match) == 1
            nmatches += 1

            midx, bsym, ctx_update = match[0]
            previously_matched = previously_matched + [bsym]
            match_ctx = match_ctx | ctx_update
            matches.append((midx, bsym))

            if (nmatches + already_matched) == max_times:
                break

        if (len(matches) + already_matched) >= min_times:
            return matches, match_ctx

        return None

    # TODO Add already_matched set to avoid matching the same operation twice (possible with window > 1)
    def _helper(
        self,
        trc: TraceCtx,
        *,
        ancestors: list[set[int]],
        previously_matched: list[BoundSymbol],
        match_ctx: dict,
        restricted_indices: set[int],
        matched_indices: set[int],
        depth: int,
        start_idx: int,
        window: int,
    ) -> None | list[tuple[int, BoundSymbol]]:
        # Stops execution when there are no matchers to match, signaling the match is complete
        if depth == len(self.matchers):
            return []

        matcher, min_times, max_times = self.matchers[depth]
        cur_match: list[tuple[int, BoundSymbol]]
        continuation_matches: None | list[tuple[int, BoundSymbol]]

        # Handles multiple matches
        if max_times < 0 or max_times > 1:
            multiple_matches: None | list[tuple[int, BoundSymbol]] = self._match_multiple(
                trc,
                previously_matched=previously_matched,
                match_ctx=match_ctx,
                restricted_indices=restricted_indices,
                depth=depth,
                idx=start_idx,
            )

            if multiple_matches is None:
                return None

            multi_matches, match_ctx = multiple_matches
            if len(multi_matches) > 0:
                midx, mbsym = multi_matches[0]
                matched_indices = matched_indices | {midx}
                previously_matched = previously_matched + [mbsym]
                idx, _ = multi_matches[-1]
                cur_match = [(midx, mbsym)]
                for midx, mbsym in multi_matches[1:]:
                    matched_indices.add(midx)
                    previously_matched.append(mbsym)
                    cur_match.append((midx, mbsym))

            continuation_matches = self._helper(
                trc,
                ancestors=ancestors,
                previously_matched=previously_matched,
                match_ctx=match_ctx,
                restricted_indices=restricted_indices,
                matched_indices=matched_indices,
                depth=(depth + 1),
                start_idx=idx + 1,
                window=window,
            )

        else:
            # NOTE In this path there are not repeated matches being considered
            pmatcher: Callable = partial(matcher, previously_matched=previously_matched, match_ctx=match_ctx)

            cur_matches: list[tuple[int, BoundSymbol, dict]] = match_all(
                trc,
                pmatcher,
                ancestors=ancestors,
                matched_indices=matched_indices,
                restricted_indices=restricted_indices,
                start_idx=start_idx,
                window=window,
            )

            # Shortcircuits if there are no matches at this level
            if len(cur_matches) == 0:
                return None

            idx: int
            bsym: BoundSymbol
            ctx_update: dict
            for idx, bsym, ctx_update in cur_matches:
                continuation_matches = self._helper(
                    trc,
                    ancestors=ancestors,
                    previously_matched=previously_matched + [bsym],
                    match_ctx=match_ctx | ctx_update,
                    restricted_indices=restricted_indices,
                    matched_indices=matched_indices | {idx},
                    depth=(depth + 1),
                    start_idx=idx + 1,
                    window=window,
                )

                if continuation_matches is not None:
                    cur_match = [(idx, bsym)]
                    break

        if continuation_matches is not None:
            return cur_match + continuation_matches

        # NOTE In this patch there was no valid continuation of the pattern
        return None

    # TODO Call ancestors in the pattern matching transform, not for each pattern call
    # TODO Document return value
    # Finds sequences of operations in trc that match the pattern
    def __call__(
        self, trc: TraceCtx, *, restricted_indices: None | set[int] = None, window: int = 5
    ) -> list[list[tuple[int, BoundSymbol]]]:
        matches: list[list[tuple[int, BoundSymbol]]] = []

        if restricted_indices is None:
            restricted_indices = set()

        # Shortcircuits if there are no matchers (which means this matches nothing)
        if len(self.matchers) == 0:
            return matches

        ancestors: list[set[int]] = numbered_ancestors(trc)
        match_ctx: dict = {}
        previously_matched: list[BoundSymbol] = []
        first_matcher, min_times, max_times = self.matchers[0]
        pmatcher: Callable = partial(first_matcher, previously_matched=previously_matched, match_ctx=match_ctx)
        start_matches: list[tuple[int, BoundSymbol]] = match_all(
            trc,
            pmatcher,
            ancestors=ancestors,
            restricted_indices=restricted_indices,
            matched_indices=set(),
            start_idx=0,
            window=-1,
        )

        if min_times <= 0:
            raise NotImplementedError(
                "The first pattern must be matched at least 1 time; please file an issue if you'd like support for optional initial patterns"
            )

        idx: int
        bsym: BoundSymbol
        ctx_update: dict
        for idx, bsym, ctx_update in start_matches:
            previously_matched: list[BoundSymbol] = [bsym]
            matched_indices = {idx}
            ctx: dict = ctx_update

            # NOTE The following check is necessary, because the initial match doesn't consider any indices
            #   restricted, but previous pattern matching may have made some of the initial match invalid
            if idx in restricted_indices:
                continue

            cur_match: list[tuple[int, BoundSymbol]] = [(idx, bsym)]

            # Handles multiple matches
            if max_times < 0 or max_times > 1:
                multiple_matches: None | tuple[list[tuple[int, BoundSymbol]], dict] = self._match_multiple(
                    trc,
                    previously_matched=previously_matched,
                    match_ctx=ctx,
                    restricted_indices=restricted_indices,
                    depth=0,
                    idx=(idx + 1),
                    already_matched=1,
                )
                if multiple_matches is None:
                    continue

                multi_matches, ctx = multiple_matches
                if len(multi_matches) > 0:
                    idx, _ = multi_matches[-1]
                    for midx, mbsym in multi_matches:
                        previously_matched.append(mbsym)
                        matched_indices.add(midx)
                        cur_match.append((midx, mbsym))

            continuation_matches: None | list[tuple[int, BoundSymbol]] = self._helper(
                trc,
                ancestors=ancestors,
                previously_matched=previously_matched,
                match_ctx=ctx,
                restricted_indices=restricted_indices,
                matched_indices=matched_indices,
                depth=1,
                start_idx=idx + 1,
                window=window,
            )
            if continuation_matches is not None:
                match: list[tuple[int, BoundSymbol]] = cur_match + continuation_matches
                matches.append(match)

                # Updates which indices have been matched (and so are not eligible for other matches)
                for match_idx, _ in match:
                    restricted_indices.add(match_idx)

        return matches

        # Get all matches given current state
        # For each match, see if matcher returns true
        # If it does, recurse with next matcher (track depth)
        # If everything returns True -- pattern
        # Recursive call returns (idx, bysm) or None

        return matches
