from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

from .data import Instance, SpatialDataset
from .neighborhood import NeighborhoodList


@dataclass
class NTreeNode:
    instance: Instance
    parent: Optional["NTreeNode"] = None
    children: List["NTreeNode"] = field(default_factory=list)

    def path_instances(self) -> Tuple[Instance, ...]:
        path: List[Instance] = []
        cur: Optional[NTreeNode] = self
        while cur is not None:
            path.append(cur.instance)
            cur = cur.parent
        return tuple(sorted(path))


class NTree:
    def __init__(self) -> None:
        self.heads: List[NTreeNode] = []

    def add_head(self, inst: Instance) -> NTreeNode:
        node = NTreeNode(instance=inst)
        self.heads.append(node)
        return node

    def leaves(self) -> List[NTreeNode]:
        result: List[NTreeNode] = []

        def dfs(node: NTreeNode) -> None:
            if not node.children:
                result.append(node)
            else:
                for ch in node.children:
                    dfs(ch)

        for h in self.heads:
            dfs(h)
        return result


def _build_for_head(head_node: NTreeNode, nbs: NeighborhoodList) -> None:
    """
    Xây N-tree cho H_{head}-Cliques theo 4 strategy, giống Example 6.
    """
    head = head_node.instance
    bodies: List[Set[Instance]] = []  # list các body hiện tại

    # Candidate instances cho body: BNs(head)
    candidates = sorted(nbs.bns(head))

    for s in candidates:
        # relation = SNs(s) ∩ BNs(head)
        relation = nbs.sns(s) & set(candidates)

        if not bodies or not relation:
            # Strategy 1: không có body hoặc relation rỗng
            bodies.append({s})
            node = NTreeNode(instance=s, parent=head_node)
            head_node.children.append(node)
            continue

        new_bodies: List[Set[Instance]] = []

        # Strategy 2–4 trên từng body
        for body in bodies:
            if not body:
                continue

            if body.issubset(relation):
                # Strategy 2a: relation ⊇ body -> body ∪ {s}
                body.add(s)
                continue

            if relation.issubset(body):
                # Strategy 2b/3: relation ⊂ body
                nb = set(relation)
                nb.add(s)
                if nb not in bodies and nb not in new_bodies:
                    new_bodies.append(nb)
                continue

            inter = body & relation
            if inter:
                # Strategy 4: intersection không rỗng
                nb = set(inter)
                nb.add(s)
                if nb not in bodies and nb not in new_bodies:
                    new_bodies.append(nb)

        for nb in new_bodies:
            bodies.append(nb)
            node = NTreeNode(instance=s, parent=head_node)
            head_node.children.append(node)


def mine_cliques_nds(dataset: SpatialDataset, nbs: NeighborhoodList):
    """
    Algorithm 3 – NDS.
    Trả về list N-cliques (maximal, không duplication/subset).
    """
    ntree = NTree()

    for s in dataset.instances:
        # queue = SNs(s) – mỗi small neighbor của s là tiềm năng head
        for head_candidate in sorted(nbs.sns(s)):
            head_node = ntree.add_head(head_candidate)
            _build_for_head(head_node, nbs)

    # Lấy cliques từ leaf nodes
    # cliquecoloc/nds.py
    cliques: List[Tuple[Instance, ...]] = []
    for leaf in ntree.leaves():
        c = leaf.path_instances()
        if len(c) >= 2:          # CHỈ clique size >= 2
            cliques.append(c)

    # loại duplication & subset (Lemma 6,7)
    unique: List[Tuple[Instance, ...]] = []
    for c in sorted(cliques, key=len, reverse=True):
        fs = set(c)
        if any(fs.issubset(set(o)) for o in unique):
            continue
        unique.append(c)

    return unique

