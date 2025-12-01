# colocation/neighbors.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set
from collections import defaultdict
from data.data import Instance, SpatialDataset


@dataclass
class Neighborhood:
    SNs: Dict[Instance, Set[Instance]]
    BNs: Dict[Instance, Set[Instance]]


class NeighborMaterializer:
    """
    Algorithm 1 (paper): chia không gian thành grid min_dist x min_dist
    rồi chỉ check 9 ô lân cận để sinh SNs(s), BNs(s).
    """

    def __init__(self, dataset: SpatialDataset, min_dist: float):
        self.instances = dataset.instances
        self.min_dist = min_dist

    def materialize(self) -> Neighborhood:
        grid_size = self.min_dist
        grids: Dict[Tuple[int, int], List[Instance]] = defaultdict(list)

        # chia lưới
        for inst in self.instances:
            gx = int(inst.x // grid_size)
            gy = int(inst.y // grid_size)
            grids[(gx, gy)].append(inst)

        sorted_instances = sorted(self.instances)
        order = {inst: i for i, inst in enumerate(sorted_instances)}
        SNs = {inst: set() for inst in sorted_instances}
        BNs = {inst: set() for inst in sorted_instances}

        def neighbor_cells(cell: Tuple[int, int]):
            x, y = cell
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    yield (x + dx, y + dy)

        for cell, insts in grids.items():
            candidates: List[Instance] = []
            for nc in neighbor_cells(cell):
                candidates.extend(grids.get(nc, []))

            for s in insts:
                for t in candidates:
                    if s is t:
                        continue
                    dx = s.x - t.x
                    dy = s.y - t.y
                    if dx * dx + dy * dy <= self.min_dist ** 2:
                        if order[s] < order[t]:
                            BNs[s].add(t)
                            SNs[t].add(s)
                        elif order[t] < order[s]:
                            BNs[t].add(s)
                            SNs[s].add(t)

        return Neighborhood(SNs=SNs, BNs=BNs)
