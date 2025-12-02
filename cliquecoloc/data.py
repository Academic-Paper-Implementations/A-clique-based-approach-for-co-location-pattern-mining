from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set
import csv
from pathlib import Path


@dataclass(order=True, frozen=True)
class Instance:
    """
    Một instance s = (feature, idx, x, y)
    Thứ tự: feature name, sau đó index – đúng mô tả Sec.3.
    """
    feature: str
    idx: int
    x: float
    y: float

    def __str__(self) -> str:
        return f"{self.feature}.{self.idx}"


@dataclass
class SpatialDataset:
    """
    Tập dữ liệu không gian (F,S) như Sec.2.
    """
    instances: List[Instance] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Sort chuẩn theo paper: theo feature name rồi index
        self.instances.sort()
        self.feature_to_instances: Dict[str, List[Instance]] = {}
        for s in self.instances:
            self.feature_to_instances.setdefault(s.feature, []).append(s)

    @property
    def features(self) -> Set[str]:
        return set(self.feature_to_instances.keys())

    def feature_counts(self) -> Dict[str, int]:
        return {f: len(v) for f, v in self.feature_to_instances.items()}


# -------------------- CSV helpers --------------------


def load_csv(path: str | Path) -> SpatialDataset:
    """
    Đọc file CSV với cột: feature, idx, x, y (hoặc Feature, InstanceID, X, Y).
    """
    path = Path(path)
    instances: List[Instance] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Handle both lowercase and capitalized column names
            feature = row.get("feature") or row.get("Feature")
            idx = row.get("idx") or row.get("InstanceID")
            x = row.get("x") or row.get("X")
            y = row.get("y") or row.get("Y")
            
            instances.append(
                Instance(
                    feature=str(feature),
                    idx=int(idx),
                    x=float(x),
                    y=float(y),
                )
            )
    return SpatialDataset(instances)


def save_csv(dataset: SpatialDataset, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "idx", "x", "y"])
        for s in dataset.instances:
            writer.writerow([s.feature, s.idx, s.x, s.y])
