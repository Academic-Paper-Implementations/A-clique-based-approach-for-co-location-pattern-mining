# cliquecoloc/utils.py
from __future__ import annotations
from typing import Iterable, FrozenSet, List
from itertools import combinations


def all_nonempty_subsets(p: FrozenSet[str]) -> List[FrozenSet[str]]:
    elems = list(p)
    subs: List[FrozenSet[str]] = []
    # CHỈ lấy subset size >= 2
    for r in range(2, len(elems)):
        for comb in combinations(elems, r):
            subs.append(frozenset(comb))
    return subs


def direct_subsets(p: FrozenSet[str]) -> List[FrozenSet[str]]:
    elems = list(p)
    k = len(elems) - 1
    # Nếu size sau khi bớt 1 < 2 thì bỏ luôn (không sinh subset size 1)
    if k < 2:
        return []
    return [frozenset(c) for c in combinations(elems, k)]
