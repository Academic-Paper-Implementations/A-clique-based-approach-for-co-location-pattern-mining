# colocation/prevalent.py - algor5 & 6: prevalent co-location mining
from __future__ import annotations
from typing import Dict, FrozenSet, List, Set
from .chash import CHash
from data.data import Instance


class PrevalentMiner:
    def __init__(
        self,
        chash: CHash,
        feature_counts: Dict[str, int],
        min_prev: float,
    ):
        self.chash = chash
        self.feature_counts = feature_counts
        self.min_prev = min_prev

    # ---------- Algorithm 6: Calculate PI value ----------

    def calculate_pi(self, cp: FrozenSet[str]) -> float:
        supersets = self.chash.supersets_of(cp)
        if not supersets:
            return 0.0

        ins_union: Dict[str, Set[Instance]] = {f: set() for f in cp}
        for patt in supersets:
            for f in cp:
                ins_union[f].update(self.chash.instances_for(patt, f))

        PRs: Dict[str, float] = {}
        for f in cp:
            denom = self.feature_counts.get(f, 0)
            num = len(ins_union[f])
            PRs[f] = num / denom if denom > 0 else 0.0

        return min(PRs.values()) if PRs else 0.0

    # ---------- Algorithm 5: Prevalent co-location filtering ----------

    def mine(self) -> Dict[FrozenSet[str], float]:
        candidates: List[FrozenSet[str]] = sorted(
            self.chash.keys(),
            key=lambda s: (-len(s), sorted(s)),
        )
        result: Dict[FrozenSet[str], float] = {}

        while candidates:
            curr = candidates[0]
            pi = self.calculate_pi(curr)

            if pi >= self.min_prev:
                # curr prevalent -> mọi subset cũng prevalent
                subsets = self._all_nonempty_subsets(curr)
                for sub in subsets:
                    if sub not in result:
                        result[sub] = self.calculate_pi(sub)
                    if sub in candidates:
                        candidates.remove(sub)
            else:
                # curr không prevalent -> thử các (k-1)-subsets
                direct = self._direct_subsets(curr)
                candidates.remove(curr)
                for sub in direct:
                    if sub not in candidates and sub not in result:
                        candidates.append(sub)
                candidates.sort(key=lambda s: (-len(s), sorted(s)))

        return result

    # ---------- helper subset functions ----------

    def _all_nonempty_subsets(self, cp: FrozenSet[str]) -> List[FrozenSet[str]]:
        feats = list(cp)
        n = len(feats)
        subsets: List[FrozenSet[str]] = []
        for mask in range(1, 1 << n):
            subsets.append(
                frozenset(feats[i] for i in range(n) if mask & (1 << i))
            )
        return subsets

    def _direct_subsets(self, cp: FrozenSet[str]) -> List[FrozenSet[str]]:
        feats = list(cp)
        subs = []
        for i in range(len(feats)):
            subs.append(
                frozenset(feats[j] for j in range(len(feats)) if j != i)
            )
        return subs
