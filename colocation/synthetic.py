# colocation/synthetic.py
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from data.data import Instance, SpatialDataset


@dataclass
class GeneratorParams:
    P: int          # số core co-locations
    I: int          # average số row-instance mỗi core
    D: float        # kích thước vùng không gian (D x D)
    F: int          # số feature tổng
    Q: int          # average size của core co-location
    m: int          # tổng số instance mong muốn
    min_dist: float # ngưỡng neighbor distance
    clumpy: int = 1 # độ dồn cục (clumpy degree)

    def feature_names(self) -> List[str]:
        # F feature: A, B, C, ... Z, AA, AB, ...
        names: List[str] = []
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        n = self.F
        k = 0
        while len(names) < n:
            if k < 26:
                names.append(alphabet[k])
            else:
                # AA, AB, ...
                prefix = alphabet[(k // 26) - 1]
                suffix = alphabet[k % 26]
                names.append(prefix + suffix)
            k += 1
        return names[: self.F]


class SyntheticSpatialGenerator:
    """
    Spatial data generator bám sát Section 4.1 của paper.

    - D x D: vùng không gian
    - chia grid kích thước min_dist x min_dist
    - P core patterns, mỗi pattern size ~ Q
    - mỗi core pattern sinh khoảng I row-instance
    - clumpy: số row-instance được dồn vào cùng 1 grid
    """

    def __init__(self, params: GeneratorParams, seed: int | None = None):
        self.p = params
        if seed is not None:
            random.seed(seed)

        self.feature_names = self.p.feature_names()
        # đếm index instance theo feature
        self.feature_index: Dict[str, int] = {f: 0 for f in self.feature_names}

        # grid hệ toạ độ nguyên: (gx, gy)
        self.grid_size = self.p.min_dist
        self.num_cells_per_dim = max(1, int(math.floor(self.p.D / self.grid_size)))

    # ---------- public API ----------

    def generate(self) -> SpatialDataset:
        """
        Trả về SpatialDataset chứa:
        - instance sinh từ P core patterns với clumpy
        - noise instances (nếu cần) để đạt tổng m.
        """
        instances: List[Instance] = []

        # 1) sinh P core patterns (mỗi pattern là 1 tập feature)
        core_patterns = self._generate_core_patterns()

        # 2) sinh row-instance cho mỗi core pattern
        for pattern in core_patterns:
            insts_for_pattern = self._generate_row_instances_for_pattern(pattern)
            instances.extend(insts_for_pattern)

        # 3) nếu ít hơn m instance, sinh thêm noise
        if len(instances) < self.p.m:
            instances.extend(self._generate_noise_instances(self.p.m - len(instances)))

        # 4) nếu nhiều hơn m (hiếm) thì cắt bớt (hoặc bạn có thể giữ nguyên)
        if len(instances) > self.p.m:
            instances = instances[: self.p.m]

        return SpatialDataset(instances)

    # ---------- core pattern generation ----------

    def _generate_core_patterns(self) -> List[List[str]]:
        patterns: List[List[str]] = []
        F_names = self.feature_names
        for _ in range(self.p.P):
            # size ~ Q, nhưng có thể random xung quanh
            size = max(2, int(random.gauss(self.p.Q, 1)))
            size = min(size, self.p.F)
            patt_features = random.sample(F_names, size)
            patterns.append(sorted(patt_features))
        return patterns

    # ---------- row-instance generation ----------

    def _random_grid(self) -> Tuple[int, int]:
        gx = random.randint(0, self.num_cells_per_dim - 1)
        gy = random.randint(0, self.num_cells_per_dim - 1)
        return gx, gy

    def _random_point_in_grid(self, gx: int, gy: int) -> Tuple[float, float]:
        # mỗi grid: [gx*grid_size, (gx+1)*grid_size)
        base_x = gx * self.grid_size
        base_y = gy * self.grid_size
        x = base_x + random.random() * self.grid_size
        y = base_y + random.random() * self.grid_size
        # giới hạn trong [0, D)
        x = min(x, self.p.D - 1e-6)
        y = min(y, self.p.D - 1e-6)
        return x, y

    def _next_index(self, feature: str) -> int:
        self.feature_index[feature] += 1
        return self.feature_index[feature]

    def _generate_row_instances_for_pattern(
        self, pattern: List[str]
    ) -> List[Instance]:
        """
        pattern: danh sách feature (core co-location)
        Mỗi row-instance: mỗi feature trong pattern -> 1 instance trong cùng grid.
        clumpy: số row-instance được dồn vào cùng một grid mỗi lần random grid.
        """
        instances: List[Instance] = []
        remaining = self.p.I

        while remaining > 0:
            # chọn một grid
            gx, gy = self._random_grid()
            # trong grid này sinh 'clumpy' row-instances (hoặc ít hơn nếu near hết)
            batch = min(self.p.clumpy, remaining)

            for _ in range(batch):
                for f in pattern:
                    idx = self._next_index(f)
                    x, y = self._random_point_in_grid(gx, gy)
                    instances.append(Instance(feature=f, index=idx, x=x, y=y))

            remaining -= batch

        return instances

    # ---------- noise generation ----------

    def _generate_noise_instances(self, count: int) -> List[Instance]:
        """
        Sinh thêm instance noise (feature random, vị trí random trên toàn vùng).
        """
        ins: List[Instance] = []
        for _ in range(count):
            f = random.choice(self.feature_names)
            idx = self._next_index(f)
            x = random.random() * self.p.D
            y = random.random() * self.p.D
            ins.append(Instance(feature=f, index=idx, x=x, y=y))
        return ins
