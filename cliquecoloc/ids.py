from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from collections import deque

from .data import Instance, SpatialDataset
from .neighborhood import NeighborhoodList


@dataclass
class ITreeNode:
    instance: Optional[Instance] = None
    parent: Optional["ITreeNode"] = None
    children: List["ITreeNode"] = field(default_factory=list)
    node_link: Optional["ITreeNode"] = None  # right sibling (Def.5)


class ITree:
    def __init__(self) -> None:
        self.root = ITreeNode(instance=None)

    def add_head_node(self, inst: Instance) -> ITreeNode:
        node = ITreeNode(instance=inst, parent=self.root)
        self.root.children.append(node)
        if len(self.root.children) > 1:
            self.root.children[-2].node_link = node
        return node


def _right_sibling_instances(node: ITreeNode) -> Set[Instance]:
    if node.parent is None:
        return set()
    siblings = node.parent.children
    idx = siblings.index(node)
    rs: Set[Instance] = set()
    for sib in siblings[idx + 1 :]:
        if sib.instance is not None and sib.instance.feature != node.instance.feature:
            rs.add(sib.instance)
    return rs


def _get_children(node: ITreeNode, nbs: NeighborhoodList) -> List[Instance]:
    """
    Lemma 3:
        - head-node: children = BNs(hn_s)
        - non-root node: children = BNs(s) ∩ RS(ns)
    """
    s = node.instance
    assert s is not None

    if node.parent is not None and node.parent is node.parent.parent:
        candidates = nbs.bns(s)
    else:
        candidates = nbs.bns(s) & _right_sibling_instances(node)

    # tránh hai instance cùng feature trong 1 clique
    ancestor_features = set()
    cur = node.parent
    while cur is not None and cur.instance is not None:
        ancestor_features.add(cur.instance.feature)
        cur = cur.parent

    return [inst for inst in sorted(candidates) if inst.feature not in ancestor_features]


def _collect_clique(node: ITreeNode) -> Tuple[Instance, ...]:
    clique: List[Instance] = []
    cur = node
    while cur is not None and cur.instance is not None:
        clique.append(cur.instance)
        cur = cur.parent
    return tuple(sorted(clique))


def mine_cliques_ids(dataset: SpatialDataset, nbs: NeighborhoodList):
    """
    Algorithm 2 – IDS: trả về list I-cliques.
    """
    itree = ITree()
    cliques: List[Tuple[Instance, ...]] = []

    for s in dataset.instances:
        queue = deque()
        head = itree.add_head_node(s)
        queue.append(head)

        # cliquecoloc/ids.py  (bên trong while queue:)
        while queue:
            curr = queue.popleft()
            children_instances = _get_children(curr, nbs)

            if not children_instances:
                clique = _collect_clique(curr)
                # CHỈ giữ clique size >= 2
                if len(clique) >= 2:
                    cliques.append(clique)
                continue


            children_nodes: List[ITreeNode] = []
            prev: Optional[ITreeNode] = None
            for inst in children_instances:
                node = ITreeNode(instance=inst, parent=curr)
                curr.children.append(node)
                if prev is not None:
                    prev.node_link = node
                prev = node
                children_nodes.append(node)

            for ch in children_nodes:
                queue.append(ch)

        # reset head-nodes cho instance tiếp theo
        itree.root.children = []

    return cliques
