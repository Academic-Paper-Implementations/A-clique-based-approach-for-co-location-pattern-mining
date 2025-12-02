# colocation/data.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict


@dataclass(order=True, frozen=True)
class Instance:
    """Một instance không gian: feature, index, tọa độ."""
    feature: str
    index: int
    x: float
    y: float

    @property
    def name(self) -> str:
        return f"{self.feature}.{self.index}"


class SpatialDataset:
    def __init__(self, instances: List[Instance]):
        # luôn sort theo (feature, index)
        self.instances = sorted(instances)

    @classmethod
    def toy_example(cls) -> "SpatialDataset":
        """
        Dataset nhỏ để test (bạn thay bằng data riêng).
        """
        insts = [
            Instance("A", 1, 0.0, 0.0),
            Instance("B", 1, 0.1, 0.0),
            Instance("C", 1, 0.05, 0.08),

            Instance("A", 2, 1.0, 1.0),
            Instance("B", 2, 1.1, 1.0),
            Instance("C", 2, 1.05, 1.07),

            Instance("D", 1, 0.5, 0.5),
        ]
        return cls(insts)

    def feature_counts(self) -> Dict[str, int]:
        """|f_i| cho mỗi feature (dùng khi tính PR, PI)."""
        cnt = defaultdict(int)
        for inst in self.instances:
            cnt[inst.feature] += 1
        return dict(cnt)

    @classmethod
    def from_csv(cls, csv_path: str) -> "SpatialDataset":
        """
        Load dataset from CSV file.
        Expected format: InstanceID,Feature,X,Y
        """
        import csv
        instances = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            feature_indices = defaultdict(int)
            
            for row in reader:
                feature = row['Feature']
                x = float(row['X'])
                y = float(row['Y'])
                
                # Auto-increment index for each feature
                feature_indices[feature] += 1
                index = feature_indices[feature]
                
                instances.append(Instance(feature, index, x, y))
        
        return cls(instances)
