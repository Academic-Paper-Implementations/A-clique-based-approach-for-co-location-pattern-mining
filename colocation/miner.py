## colocation/miner.py – Facade chạy full pipeline
from typing import Dict, FrozenSet, List, Tuple
from data.data import SpatialDataset, Instance
from .neighbors import NeighborMaterializer
from .itree_ids import IDSGenerator
from .ntree_nds import NDSGenerator
from .chash import CHash
from .prevalent import PrevalentMiner


class CoLocationMiner:
    def __init__(
        self,
        dataset: SpatialDataset,
        min_dist: float,
        min_prev: float,
    ):
        self.dataset = dataset
        self.min_dist = min_dist
        self.min_prev = min_prev

    def _build_chash(self, cliques: List[Tuple[Instance, ...]]) -> CHash:
        ch = CHash()
        for cl in cliques:
            ch.add_clique(cl)
        return ch

    def run_ids(self):
        neigh = NeighborMaterializer(self.dataset, self.min_dist).materialize()
        ids = IDSGenerator(neigh)
        cliques = ids.generate_cliques()
        chash = self._build_chash(cliques)
        miner = PrevalentMiner(chash, self.dataset.feature_counts(), self.min_prev)
        prevalent = miner.mine()
        return cliques, prevalent

    def run_nds(self):
        neigh = NeighborMaterializer(self.dataset, self.min_dist).materialize()
        nds = NDSGenerator(neigh)
        cliques = nds.generate_cliques()
        chash = self._build_chash(cliques)
        miner = PrevalentMiner(chash, self.dataset.feature_counts(), self.min_prev)
        prevalent = miner.mine()
        return cliques, prevalent
