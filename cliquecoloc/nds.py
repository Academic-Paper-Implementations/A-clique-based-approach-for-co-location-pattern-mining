from __future__ import annotations
from typing import List, Tuple, Set

from .data import Instance, SpatialDataset
from .data import Instance, SpatialDataset
from .neighborhood import NeighborhoodList
from concurrent.futures import ProcessPoolExecutor
from itertools import islice


def _neighbors_in_set(
    s: Instance,
    cand: Set[Instance],
    nbs: NeighborhoodList
) -> Set[Instance]:
    """
    Trả về các láng giềng của s nằm trong tập cand.
    Dùng Ns(s) đã materialize trong NeighborhoodList.
    """
    return nbs.ns(s) & cand


def _process_heads_nds(
    heads: List[Instance], nbs: NeighborhoodList
) -> List[Tuple[Instance, ...]]:
    """
    Worker function: Mine cliques using Bron-Kerbosch for a list of head-nodes.
    """
    local_cliques: List[Tuple[Instance, ...]] = []

    for head in heads:
        # Body candidates: big neighbors of head
        body_candidates: Set[Instance] = set(nbs.bns(head))

        # Bron-Kerbosch
        def expand(
            clique: Tuple[Instance, ...],
            candidates: Set[Instance],
            excluded: Set[Instance]
        ) -> None:
            if not candidates and not excluded:
                if len(clique) >= 2:
                    local_cliques.append(tuple(sorted(clique)))
                return

            for v in list(candidates):
                new_clique = clique + (v,)
                new_candidates = _neighbors_in_set(v, candidates, nbs)
                new_excluded = _neighbors_in_set(v, excluded, nbs)

                expand(new_clique, new_candidates, new_excluded)

                candidates.remove(v)
                excluded.add(v)

        expand((head,), body_candidates, set())
        
    return local_cliques


def mine_cliques_nds(
    dataset: SpatialDataset,
    nbs: NeighborhoodList,
    n_workers: int = 1
) -> List[Tuple[Instance, ...]]:
    """
    NDS – Mine N-cliques (maximal cliques).
    Supports parallel execution if n_workers > 1.
    """

    all_cliques: List[Tuple[Instance, ...]] = []
    instances = sorted(dataset.instances)

    # Serial path
    if n_workers <= 1:
        all_cliques = _process_heads_nds(instances, nbs)
    else:
        # Parallel path
        chunk_size = max(1, len(instances) // n_workers)
        chunks = [
            instances[i : i + chunk_size]
            for i in range(0, len(instances), chunk_size)
        ]
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(_process_heads_nds, chunk, nbs) 
                for chunk in chunks
            ]
            for fut in futures:
                all_cliques.extend(fut.result())

    # De-duplicate to be safe (though algorithmically should be unique if strict weak ordering)
    unique: List[Tuple[Instance, ...]] = []
    seen: Set[Tuple[Instance, ...]] = set()

    for c in all_cliques:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(key)

    return unique
