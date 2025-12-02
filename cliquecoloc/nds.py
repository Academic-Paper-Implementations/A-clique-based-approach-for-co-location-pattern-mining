from __future__ import annotations
from typing import List, Tuple, Set

from .data import Instance, SpatialDataset
from .neighborhood import NeighborhoodList


def _neighbors_in_set(
    s: Instance,
    cand: Set[Instance],
    nbs: NeighborhoodList
) -> Set[Instance]:
    """
    Trả về các láng giềng của s nằm trong tập cand.
    Dùng Ns(s) đã materialize trong NeighborhoodList.
    """
    return nbs.ns(s) & cand


def mine_cliques_nds(
    dataset: SpatialDataset,
    nbs: NeighborhoodList
) -> List[Tuple[Instance, ...]]:
    """
    NDS – khai phá N-cliques (maximal cliques) dựa trên head H_s.

    Ý tưởng:
        - Với mỗi instance h trong dataset (đã sort):
            + Body candidates = BNs(h): các "big neighbors" của h.
            + Chạy Bron–Kerbosch để tìm mọi maximal clique chứa h
              trong đồ thị con cảm ứng bởi {h} ∪ BNs(h).
        - Nhờ dùng BNs(h) (neighbors > h) nên:
            + Mỗi clique sẽ chỉ được sinh đúng 1 lần, với head
              là instance nhỏ nhất trong clique.

    Trả về:
        - Danh sách các clique, mỗi clique là tuple Instance (đã sort).
        - Chỉ giữ clique có size >= 2.
    """

    all_cliques: List[Tuple[Instance, ...]] = []

    # Duyệt head theo thứ tự tăng (phù hợp với thứ tự bạn dùng trong Algorithm 1)
    for head in sorted(dataset.instances):

        # Body candidates: các "big neighbors" của head
        body_candidates: Set[Instance] = set(nbs.bns(head))

        # Bron–Kerbosch với:
        #   clique (R)    = {head}
        #   candidates(P) = body_candidates
        #   excluded  (X) = ∅
        def expand(
            clique: Tuple[Instance, ...],
            candidates: Set[Instance],
            excluded: Set[Instance]
        ) -> None:
            """
            Invariant:
                - clique: hiện là 1 clique (luôn chứa head).
                - candidates: các node có thể thêm vào clique
                              (tất cả đều kề với mọi node trong clique).
                - excluded: các node đã được xem xét với gốc clique này.
            """
            # Nếu không còn candidates và excluded:
            #    -> clique là maximal (không thể mở rộng thêm)
            if not candidates and not excluded:
                if len(clique) >= 2:  # chỉ giữ các clique có size >= 2
                    all_cliques.append(tuple(sorted(clique)))
                return

            # Duyệt từng candidate v trong bản copy để không phá vòng lặp
            for v in list(candidates):
                # new_clique = clique ∪ {v}
                new_clique = clique + (v,)

                # Các candidate mới: neighbors của v trong candidates
                new_candidates = _neighbors_in_set(v, candidates, nbs)

                # Các excluded mới: neighbors của v trong excluded
                new_excluded = _neighbors_in_set(v, excluded, nbs)

                # Đệ quy mở rộng
                expand(new_clique, new_candidates, new_excluded)

                # Di chuyển v từ candidates sang excluded (như Bron–Kerbosch gốc)
                candidates.remove(v)
                excluded.add(v)

        # Gọi expand khởi đầu với clique = {head}
        expand((head,), body_candidates, set())

    # Loại trùng cho chắc (trong trường hợp __lt__ / sort có behavior lạ)
    unique: List[Tuple[Instance, ...]] = []
    seen: Set[Tuple[Instance, ...]] = set()

    for c in all_cliques:
        key = tuple(sorted(c))
        if key not in seen:
            seen.add(key)
            unique.append(key)

    return unique
