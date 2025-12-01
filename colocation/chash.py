# colocation/chash.py - algor4: c-hash structure
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Set
from data.data import Instance


@dataclass
class CHash:
    """
    Definition 9:
    key:   tập feature F_c (frozenset[str])
    value: dict feature -> set(instance)
    """
    data: Dict[FrozenSet[str], Dict[str, Set[Instance]]] = field(
        default_factory=dict
    )

    def add_clique(self, clique: Iterable[Instance]) -> None:
        feats = frozenset(inst.feature for inst in clique)
        if len(feats) <= 1:
            return
        if feats not in self.data:
            self.data[feats] = {f: set() for f in feats}
        bucket = self.data[feats]
        for inst in clique:
            bucket[inst.feature].add(inst)

    # API cho Algorithm 5 & 6
    def keys(self) -> List[FrozenSet[str]]:
        return list(self.data.keys())

    def supersets_of(self, cp: FrozenSet[str]) -> List[FrozenSet[str]]:
        return [k for k in self.data if cp.issubset(k)]

    def instances_for(self, patt: FrozenSet[str], feature: str) -> Set[Instance]:
        return self.data[patt][feature]
