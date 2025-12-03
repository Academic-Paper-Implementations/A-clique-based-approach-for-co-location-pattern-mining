# cliquecoloc/chash.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Set

from .data import Instance


@dataclass
class CHash:
    """
    C-Hash structure.

    - Key  : frozenset of feature names (colocation type), ví dụ: frozenset({"A", "B", "C"})
    - Value: dict[feature_name] -> set[Instance] xuất hiện trong các clique thuộc type đó.
    """
    table: Dict[FrozenSet[str], Dict[str, Set[Instance]]] = field(default_factory=dict)

    def add_clique(self, clique: Iterable[Instance]) -> None:
        """
        Thêm một clique (I-clique hoặc N-clique) vào C-Hash.

        - Bỏ các clique size < 2.
        - Bỏ các clique mà sau khi gom feature chỉ còn 1 feature (không phải colocation).
        - Gom tất cả instance theo (type, feature) để về sau tính prevalence / participation.
        """
        cl_list = list(clique)

        # Bỏ clique size 1
        if len(cl_list) < 2:
            return

        # Type = tập feature của clique
        key: FrozenSet[str] = frozenset(s.feature for s in cl_list)

        # Nếu chỉ có 1 feature thì không phải colocation
        if len(key) < 2:
            return

        # (Optional) nếu muốn đảm bảo mỗi clique không có 2 instance cùng feature:
        # assert len(key) == len(cl_list), "Clique has duplicated features!"

        # Khởi tạo bucket cho type mới
        bucket = self.table.get(key)
        if bucket is None:
            bucket = {f: set() for f in key}
            self.table[key] = bucket

        # Gom instance theo feature
        for s in cl_list:
            bucket[s.feature].add(s)

    @property
    def candidates(self) -> List[FrozenSet[str]]:
        """
        Danh sách tất cả colocation types (các tập feature) đã thu được.
        """
        return list(self.table.keys())

    def instances_for(self, key: FrozenSet[str], feature: str) -> Set[Instance]:
        """
        Lấy tập instance của một feature trong một colocation type cụ thể.

        Ví dụ:
            key = frozenset({"A", "B"})
            feature = "A"
            -> trả về tất cả instance A thuộc các clique có type {A,B}.
        """
        return self.table[key][feature]
