from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
import math

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


def materialize_neighborhoods(dataset: SpatialDataset, min_dist: float) -> NeighborhoodList:
    """
    Algorithm 1 – Neighborhood materialization (Grid-based).
    """
    nbs = NeighborhoodList(dataset)
    instances = dataset.instances

    grids, _, _ = _divide_space(instances, min_dist)

    for cell, g_instances in grids.items():
        ncoords = _neighbor_grid_coords(cell)
        ngrid_instances: List[Instance] = []
        for nc in ncoords:
            ngrid_instances.extend(grids.get(nc, []))

        for s in g_instances:
            entry_s = nbs.get_entry(s)
            for s_prime in ngrid_instances:
                if s is s_prime:
                    continue
                if not _is_neighbor(s, s_prime, min_dist):
                    continue

                entry_sp = nbs.get_entry(s_prime)

                if s < s_prime:
                    # s < s'  ⇒ s' ∈ BNs(s), s ∈ SNs(s')
                    entry_s.bns.add(s_prime)
                    entry_s.ns.add(s_prime)
                    entry_sp.sns.add(s)
                    entry_sp.ns.add(s)
                else:
                    # s > s' ⇒ s ∈ BNs(s'), s' ∈ SNs(s)
                    entry_sp.bns.add(s)
                    entry_sp.ns.add(s)
                    entry_s.sns.add(s_prime)
                    entry_s.ns.add(s_prime)

    return nbs
