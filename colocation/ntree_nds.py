# colocation/ntree_nds.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from data.data import Instance
from .neighbors import Neighborhood


@dataclass
class NTreeNode:
    instance: Optional[Instance]
    parent: Optional["NTreeNode"] = None
    children: List["NTreeNode"] = field(default_factory=list)

    def add_child(self, inst: Instance) -> "NTreeNode":
        for c in self.children:
            if c.instance == inst:
                return c
        node = NTreeNode(instance=inst, parent=self)
        self.children.append(node)
        return node


class NTree:
    def __init__(self):
        self.root = NTreeNode(instance=None)

    def add_head_node(self, inst: Instance) -> NTreeNode:
        return self.root.add_child(inst)

    def leaves_under(self, head: NTreeNode) -> List[NTreeNode]:
        leaves: List[NTreeNode] = []

        def dfs(node: NTreeNode):
            if not node.children:
                if node is not head:
                    leaves.append(node)
                return
            for ch in node.children:
                dfs(ch)

        dfs(head)
        return leaves

    @staticmethod
    def path_from_head(node: NTreeNode) -> List[Instance]:
        insts: List[Instance] = []
        cur = node
        while cur and cur.instance is not None:
            insts.append(cur.instance)
            cur = cur.parent
        insts.reverse()
        return insts


class NDSGenerator:
    """
    Algorithm 3 trong paper: Neighborhood Driven Schema (N-tree + 4 strategies).
    Kết quả là danh sách N-cliques (thường gần với maximal cliques, ít trùng).
    """

    def __init__(self, neighborhood: Neighborhood):
        self.SNs = neighborhood.SNs
        self.BNs = neighborhood.BNs
        self.instances = sorted(self.SNs.keys())

    def generate_cliques(self) -> List[Tuple[Instance, ...]]:
        ntree = NTree()

        # bodies[head_instance] = list[body_as_tuple_instances]
        bodies: Dict[Instance, List[Tuple[Instance, ...]]] = {}

        # Algorithm 3: For each instance s (traverse each neighborhood)
        for s in self.instances:
            small_neighbors = sorted(self.SNs[s])  # SNs(s)
            queue: List[Instance] = list(small_neighbors)

            while queue:
                head_inst = queue.pop(0)
                head = ntree.add_head_node(head_inst)

                curr_bodies = bodies.get(head_inst)
                # relation = queue ∩ BNs(headInst)
                relation: Set[Instance] = set(queue).intersection(self.BNs[head_inst])

                if not curr_bodies or not relation:
                    # Strategy 1
                    self._apply_strategy1(ntree, head, s, bodies)
                else:
                    new_bodies: List[Tuple[Instance, ...]] = []
                    for body in curr_bodies:
                        body_set = set(body)
                        inter = body_set.intersection(relation)

                        if body_set == relation:
                            # Strategy 2: l == relation
                            new_body = tuple(sorted(body_set | {s}))
                            new_bodies.append(new_body)

                        elif relation.issubset(body_set):
                            # Strategy 2 (case relation ⊆ l)
                            new_body = tuple(sorted(relation | {s}))
                            new_bodies.append(new_body)

                        elif body_set.issubset(relation):
                            # Strategy 3
                            new_body = tuple(sorted(body_set | {s}))
                            new_bodies.append(new_body)

                        elif inter:
                            # Strategy 4
                            new_body = tuple(sorted(inter | {s}))
                            new_bodies.append(new_body)

                    # cập nhật bodies cho head_inst
                    merged: Set[Tuple[Instance, ...]] = set(curr_bodies)
                    merged.update(new_bodies)
                    bodies[head_inst] = list(merged)

                    # update tree theo bodies
                    self._sync_head_bodies_to_tree(ntree, head, bodies[head_inst])

        # Sau khi build N-tree, sinh N-cliques từ leaf path
        cliques: List[Tuple[Instance, ...]] = []
        for head_node in ntree.root.children:
            for leaf in ntree.leaves_under(head_node):
                clique = tuple(NTree.path_from_head(leaf))
                if len(clique) > 1:
                    cliques.append(clique)
        return cliques

    # ---------- strategy helpers ----------

    def _apply_strategy1(
        self,
        ntree: NTree,
        head: NTreeNode,
        s: Instance,
        bodies: Dict[Instance, List[Tuple[Instance, ...]]],
    ) -> None:
        """
        Strategy 1: body == None hoặc relation == rỗng => head + s tạo clique size-2.
        """
        head_inst = head.instance
        assert head_inst is not None
        body = (s,)
        bodies.setdefault(head_inst, [])
        if body not in bodies[head_inst]:
            bodies[head_inst].append(body)
        self._sync_head_bodies_to_tree(ntree, head, bodies[head_inst])

    def _sync_head_bodies_to_tree(
        self,
        ntree: NTree,
        head: NTreeNode,
        bodies: List[Tuple[Instance, ...]],
    ) -> None:
        """
        Đồng bộ tree theo danh sách bodies cho một head:
        mỗi body tương ứng một đường head -> ... -> leaf.
        """
        # Xóa children cũ và rebuild (đơn giản, dễ hiểu)
        head.children.clear()
        for body in bodies:
            cur = head
            for inst in sorted(body):
                cur = cur.add_child(inst)
