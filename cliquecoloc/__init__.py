from .data import Instance, SpatialDataset, load_csv, save_csv
from .neighborhood import materialize_neighborhoods, NeighborhoodList
from .ids import mine_cliques_ids
from .nds import mine_cliques_nds
from .chash import CHash
from .prevalence import mine_prevalent_patterns
from .generator import GeneratorParams, generate_synthetic

__all__ = [
    "Instance",
    "SpatialDataset",
    "load_csv",
    "save_csv",
    "materialize_neighborhoods",
    "NeighborhoodList",
    "mine_cliques_ids",
    "mine_cliques_nds",
    "CHash",
    "mine_prevalent_patterns",
    "GeneratorParams",
    "generate_synthetic",
]


def run_pipeline(
    dataset: SpatialDataset,
    min_dist: float,
    min_prev: float,
    schema: str = "nds",  # "ids" or "nds"
    n_workers: int = 1,
):
    """
    Hàm tiện dụng: chạy toàn bộ Fig.2(c) với IDS hoặc NDS.
    Supports parallel execution if n_workers > 1.
    """
    nbs = materialize_neighborhoods(dataset, min_dist, n_workers=n_workers)

    if schema.lower() == "ids":
        cliques = mine_cliques_ids(dataset, nbs, n_workers=n_workers)
    else:
        cliques = mine_cliques_nds(dataset, nbs, n_workers=n_workers)

    chash = CHash()
    for cl in cliques:
        chash.add_clique(cl)

    patterns = mine_prevalent_patterns(dataset, chash, min_prev)
    return cliques, chash, patterns
