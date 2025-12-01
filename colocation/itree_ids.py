# colocation/itree_ids.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from data.data import Instance
from .neighbors import Neighborhood


@dataclass
class ITreeNode:
    instance: Optional[Instance]
    parent: Optional["ITreeNode"] = None
    children: List["ITreeNode"] = field(default_factory=list)
    node_link: Optional["ITreeNode"] = None   # right sibling
    is_head: bool = False
    pruned: bool = False


class ITree:
    def __init__(self):
        self.root = ITreeNode(instance=None)

    def add_head_node(self, inst: Instance) -> ITreeNode:
        node = ITreeNode(instance=inst, parent=self.root, is_head=True)
        if self.root.children:
            self.root.children[-1].node_link = node
        self.root.children.append(node)
        return node

    def add_children(self, parent: ITreeNode,
                     child_insts: List[Instance]) -> List[ITreeNode]:
        nodes: List[ITreeNode] = []
        for inst in sorted(child_insts):
            # tránh duplicate child cùng instance
            if any(c.instance == inst for c in parent.children):
                continue
            node = ITreeNode(instance=inst, parent=parent)
            if parent.children:
                parent.children[-1].node_link = node
            parent.children.append(node)
            nodes.append(node)
        return nodes

    @staticmethod
    def right_sibling_instances(node: ITreeNode) -> Set[Instance]:
        rs: Set[Instance] = set()
        sib = node.node_link
        while sib is not None:
            if sib.instance is not None:
                rs.add(sib.instance)
            sib = sib.node_link
        return rs

    @staticmethod
    def path_clique(node: ITreeNode) -> Tuple[Instance, ...]:
        """Return clique = instances from head to node."""
        insts: List[Instance] = []
        cur = node
        while cur and cur.instance is not None:
            insts.append(cur.instance)
            cur = cur.parent
        insts.reverse()
        return tuple(insts)

    def prune_ancestors(self, node: ITreeNode) -> None:
        """
        Theo paper: nếu node là leaf thì có thể xóa node & tổ tiên
        cho đến khi gặp node còn child.
        Ở đây ta thực sự xóa khỏi children list (để sát paper).
        """
        cur = node
        while cur.parent is not None:
            parent = cur.parent
            if cur in parent.children:
                idx = parent.children.index(cur)
                parent.children.pop(idx)
                # rebuild node_link trong parent
                for i in range(len(parent.children) - 1):
                    parent.children[i].node_link = parent.children[i + 1]
                if parent.children:
                    parent.children[-1].node_link = None
            if parent is self.root or parent.children:
                break
            cur = parent


class IDSGenerator:
    """
    Algorithm 2 trong paper:
    - Dùng I-tree (ITree + ITreeNode)
    - BFS từ head-node để sinh tất cả I-cliques (Hs-Cliques).
    """

    def __init__(self, neighborhood: Neighborhood):
        self.SNs = neighborhood.SNs
        self.BNs = neighborhood.BNs
        self.instances = sorted(self.SNs.keys())

    def _get_children_instances(self, node: ITreeNode) -> List[Instance]:
        s = node.instance
        if s is None:
            return []
        if node.is_head:
            # Lemma 3: con của head-node hn_s là BNs(s)
            return sorted(self.BNs[s])
        else:
            # con của node khác: BNs(s) ∩ RS(ns)
            rs = ITree.right_sibling_instances(node)
            return sorted(self.BNs[s].intersection(rs))

    def generate_cliques(self) -> List[Tuple[Instance, ...]]:
        cliques: List[Tuple[Instance, ...]] = []
        itree = ITree()

        for s in self.instances:
            head = itree.add_head_node(s)
            queue: List[ITreeNode] = [head]

            while queue:
                curr = queue.pop(0)
                if curr.instance is None:
                    continue

                children_insts = self._get_children_instances(curr)

                if not children_insts:
                    # leaf => sinh I-clique (size >=2)
                    clique = ITree.path_clique(curr)
                    if len(clique) > 1:
                        cliques.append(clique)
                    itree.prune_ancestors(curr)
                else:
                    children_nodes = itree.add_children(curr, children_insts)
                    queue.extend(children_nodes)

        return cliques
