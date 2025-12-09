from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import math
from concurrent.futures import ProcessPoolExecutor
from itertools import islice

from .data import Instance, SpatialDataset


@dataclass
class NeighborhoodEntry:
    instance: Instance
    ns: Set[Instance] = field(default_factory=set)   # Ns(s)
    sns: Set[Instance] = field(default_factory=set)  # SNs(s)
    bns: Set[Instance] = field(default_factory=set)  # BNs(s)


class NeighborhoodList:
    def __init__(self, dataset: SpatialDataset) -> None:
        self.dataset = dataset
        self.entries: Dict[Instance, NeighborhoodEntry] = {
            s: NeighborhoodEntry(s) for s in dataset.instances
        }

    def get_entry(self, s: Instance) -> NeighborhoodEntry:
        return self.entries[s]

    def ns(self, s: Instance) -> Set[Instance]:
        return self.entries[s].ns

    def sns(self, s: Instance) -> Set[Instance]:
        return self.entries[s].sns

    def bns(self, s: Instance) -> Set[Instance]:
        return self.entries[s].bns

    @property
    def instances(self) -> List[Instance]:
        return list(self.dataset.instances)


def _divide_space(instances: List[Instance], min_dist: float):
    """
    DivideSpace(min_dist, S) – chia theo grid min_dist x min_dist.
    """
    if not instances:
        return {}, 0.0, 0.0

    min_x = min(s.x for s in instances)
    min_y = min(s.y for s in instances)

    grids: Dict[Tuple[int, int], List[Instance]] = {}
    for s in instances:
        gx = int(math.floor((s.x - min_x) / min_dist))
        gy = int(math.floor((s.y - min_y) / min_dist))
        grids.setdefault((gx, gy), []).append(s)
    return grids, min_x, min_y


def _neighbor_grid_coords(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
    gx, gy = cell
    coords = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            coords.append((gx + dx, gy + dy))
    return coords



def _is_neighbor(a: Instance, b: Instance, min_dist: float) -> bool:
    dx = a.x - b.x
    dy = a.y - b.y
    return dx * dx + dy * dy <= min_dist * min_dist


def _process_cells(
    cells: List[Tuple[int, int]],
    grids: Dict[Tuple[int, int], List[Instance]],
    min_dist: float
) -> List[Tuple[Instance, Instance]]:
    """
    Worker function to process a batch of grid cells.
    Returns a list of neighbor pairs (u, v) such that u < v.
    """
    edges: List[Tuple[Instance, Instance]] = []
    
    for cell in cells:
        if cell not in grids:
            continue
            
        g_instances = grids[cell]
        ncoords = _neighbor_grid_coords(cell)
        ngrid_instances: List[Instance] = []
        for nc in ncoords:
            if nc in grids:
                ngrid_instances.extend(grids[nc])
        
        # Check neighbors
        for s in g_instances:
            for s_prime in ngrid_instances:
                if s is s_prime:
                    continue
                
                # Only keep pairs where s < s_prime to avoid duplicates
                if s < s_prime:
                    if _is_neighbor(s, s_prime, min_dist):
                        edges.append((s, s_prime))
                        
    return edges


def materialize_neighborhoods(
    dataset: SpatialDataset, min_dist: float, n_workers: int = 1
) -> NeighborhoodList:
    """
    Algorithm 1 – Neighborhood materialization (Grid-based).
    Supports parallel execution if n_workers > 1.
    """
    nbs = NeighborhoodList(dataset)
    instances = dataset.instances

    grids, _, _ = _divide_space(instances, min_dist)
    all_cells = list(grids.keys())

    # Serial path
    if n_workers <= 1:
        # Use simple logic but optimized to reuse _process_cells logic or original?
        # Let's keep original logic structure or reuse _process_cells?
        # Reusing _process_cells ensures consistency.
        edges = _process_cells(all_cells, grids, min_dist)
        _apply_edges(nbs, edges)
        return nbs

    # Parallel path
    chunk_size = max(1, len(all_cells) // n_workers)
    chunks = [
        all_cells[i : i + chunk_size]
        for i in range(0, len(all_cells), chunk_size)
    ]

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_process_cells, chunk, grids, min_dist) 
            for chunk in chunks
        ]
        
        for fut in futures:
            edges = fut.result()
            _apply_edges(nbs, edges)

    return nbs


def _apply_edges(nbs: NeighborhoodList, edges: List[Tuple[Instance, Instance]]) -> None:
    """
    Helper to update NeighborhoodList from list of (u, v) edges where u < v.
    """
    for u, v in edges:
        # u < v verified by _process_cells
        entry_u = nbs.get_entry(u)
        entry_v = nbs.get_entry(v)

        # u < v  ⇒ v ∈ BNs(u), u ∈ SNs(v)
        entry_u.bns.add(v)
        entry_u.ns.add(v)
        
        entry_v.sns.add(u)
        entry_v.ns.add(u)
