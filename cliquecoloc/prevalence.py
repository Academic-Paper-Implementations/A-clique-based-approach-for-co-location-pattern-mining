from __future__ import annotations
from typing import Dict, FrozenSet, List

from .data import SpatialDataset
from .chash import CHash
from .utils import all_nonempty_subsets, direct_subsets


# ---------------- Algorithm 6 – Calculate PI value -----------------


def calculate_pi(cp: FrozenSet[str], chash: CHash, feature_counts: Dict[str, int]) -> float:
    """
    Algorithm 6 + Lemma 10: PI(cp) = min_i |⋃ cp_j[f_i]| / |f_i|.
    """
    supersets = [key for key in chash.table.keys() if cp.issubset(key)]
    if not supersets:
        return 0.0

    ins_union: Dict[str, set] = {f: set() for f in cp}
    for key in supersets:
        bucket = chash.table[key]
        for f in cp:
            ins_union[f].update(bucket[f])

    prs: Dict[str, float] = {}
    for f in cp:
        denom = feature_counts.get(f, 0)
        prs[f] = 0.0 if denom == 0 else len(ins_union[f]) / float(denom)

    return min(prs.values()) if prs else 0.0


# ---------------- Algorithm 5 – Prevalent co-locations filtering ----


def mine_prevalent_patterns(
    dataset: SpatialDataset,
    chash: CHash,
    min_prev: float,
) -> Dict[FrozenSet[str], float]:
    """
    Algorithm 5 – Prevalent co-location filtering.
    Trả về map: co-location -> PI.
    """
    feature_counts = dataset.feature_counts()

    candidates: List[FrozenSet[str]] = chash.candidates
    candidates.sort(key=len, reverse=True)
    candidate_set = set(candidates)
    results: Dict[FrozenSet[str], float] = {}

    while candidates:
        curr = candidates.pop(0)
        if curr not in candidate_set:
            continue

        pi = calculate_pi(curr, chash, feature_counts)

        if pi >= min_prev:
            # currCandidate là prevalent (Steps 6–10)
            subsets = all_nonempty_subsets(curr)
            for sub in subsets:
                if sub in results:
                    continue
                sub_pi = calculate_pi(sub, chash, feature_counts)
                results[sub] = sub_pi

            results[curr] = pi

            for sub in subsets:
                candidate_set.discard(sub)
            candidate_set.discard(curr)
        else:
            # currCandidate không prevalent (Steps 11–15)
            dsubs = direct_subsets(curr)
            candidate_set.discard(curr)
            for sub in dsubs:
                if sub not in candidate_set and sub not in results:
                    candidates.append(sub)
                    candidate_set.add(sub)
            candidates.sort(key=len, reverse=True)

    return results
