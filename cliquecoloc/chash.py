# cliquecoloc/chash.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Set

from .data import Instance


@dataclass
class CHash:
    table: Dict[FrozenSet[str], Dict[str, Set[Instance]]] = field(default_factory=dict)

    def add_clique(self, clique: Iterable[Instance]) -> None:
        cl_list = list(clique)
        # BỎ clique size 1
        if len(cl_list) < 2:
            return

        key: FrozenSet[str] = frozenset(s.feature for s in cl_list)
        # BỎ luôn nếu chỉ có 1 feature
        if len(key) < 2:
            return

        if key not in self.table:
            self.table[key] = {f: set() for f in key}

        bucket = self.table[key]
        for s in cl_list:
            bucket[s.feature].add(s)

    @property
    def candidates(self) -> List[FrozenSet[str]]:
        return list(self.table.keys())

    def instances_for(self, key: FrozenSet[str], feature: str) -> Set[Instance]:
        return self.table[key][feature]
