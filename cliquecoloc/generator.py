from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random

from .data import Instance, SpatialDataset


@dataclass
class GeneratorParams:
    P: int          # number of core co-locations
    I: int          # avg row-instances per core
    D: int          # spatial area size (D x D)
    F: int          # number of features
    Q: int          # average size of core co-location
    m: int          # total number of instances
    min_dist: float
    clumpy: int = 1


def generate_synthetic(params: GeneratorParams, seed: int | None = None) -> SpatialDataset:
    """
    Spatial data generator giống mô tả Sec.4.1 + Table 2 (simplified nhưng đúng ý).
    """
    if seed is not None:
        random.seed(seed)

    # 1. Tạo feature names: A,B,C,...
    features = [chr(ord("A") + i) for i in range(params.F)]

    # 2. Sinh P core patterns, mỗi cái là tập feature size ≈ Q
    cores: List[List[str]] = []
    for _ in range(params.P):
        size = max(2, int(random.gauss(params.Q, 0.5)) )
        size = min(size, params.F)
        cores.append(random.sample(features, size))

    instances: List[Instance] = []
    # index counter cho từng feature
    idx_counter = {f: 1 for f in features}

    # 3. Generate I row-instances per core pattern
    grid_size = params.min_dist  # mỗi neighborhood ~1 grid cell
    num_grids_side = max(1, params.D // int(grid_size))
    for core in cores:
        for _ in range(params.I):
            # chọn 1 grid ngẫu nhiên
            gx = random.randrange(num_grids_side)
            gy = random.randrange(num_grids_side)
            base_x = gx * grid_size
            base_y = gy * grid_size
            for feat in core:
                x = base_x + random.random() * grid_size
                y = base_y + random.random() * grid_size
                instances.append(
                    Instance(feat, idx_counter[feat], x, y)
                )
                idx_counter[feat] += 1

    # 4. Bổ sung instance ngẫu nhiên tới đủ m
    while len(instances) < params.m:
        feat = random.choice(features)
        x = random.random() * params.D
        y = random.random() * params.D
        instances.append(
            Instance(feat, idx_counter[feat], x, y)
        )
        idx_counter[feat] += 1

    return SpatialDataset(instances)
