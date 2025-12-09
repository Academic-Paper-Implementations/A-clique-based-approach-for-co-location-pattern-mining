from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from collections import deque

from .data import Instance, SpatialDataset
from .neighborhood import NeighborhoodList
from concurrent.futures import ProcessPoolExecutor
from itertools import islice


@dataclass
class ITreeNode:
    instance: Optional[Instance] = None
    parent: Optional["ITreeNode"] = None
    children: List["ITreeNode"] = field(default_factory=list)
    node_link: Optional["ITreeNode"] = None  # right sibling (Def. 5 trong paper)


class ITree:
    def __init__(self) -> None:
        # root: node gốc, instance=None
        self.root = ITreeNode(instance=None)

    def add_head_node(self, inst: Instance) -> ITreeNode:
        """
        Tạo head-node cho instance inst (con trực tiếp của root).
        Đồng thời nối node_link giữa các head-nodes.
        """
        node = ITreeNode(instance=inst, parent=self.root)
        self.root.children.append(node)
        if len(self.root.children) > 1:
            # node_link: head-node trước đó trỏ sang head-node mới
            self.root.children[-2].node_link = node
        return node


def _right_sibling_instances(node: ITreeNode) -> Set[Instance]:
    """
    RS(ns): tập các instance ở các right-siblings của node.
    Dùng để tránh sinh trùng clique và bảo toàn thứ tự.
    """
    if node.parent is None:
        return set()

    siblings = node.parent.children
    idx = siblings.index(node)
    rs: Set[Instance] = set()

    for sib in siblings[idx + 1:]:
        if sib.instance is not None:
            rs.add(sib.instance)

    return rs


def _get_children(node: ITreeNode, nbs: NeighborhoodList) -> List[Instance]:
    """
    Lemma 3 trong paper:

        - Nếu node là head-node (con trực tiếp của root):
              children = BNs(hn_s)

        - Nếu node là non-root node (khác head-node):
              children = BNs(s) ∩ RS(ns)

    Sau đó lọc thêm ràng buộc:
        - Không cho 2 instance cùng feature xuất hiện trong cùng 1 clique.
    """
    s = node.instance
    assert s is not None

    # node là head-node nếu parent tồn tại và parent.instance is None (tức là root)
    is_head_node = (node.parent is not None and node.parent.instance is None)

    if is_head_node:
        # head-node: chỉ dùng BNs(s)
        candidates = nbs.bns(s)
    else:
        # non-root node: BNs(s) ∩ RS(ns)
        candidates = nbs.bns(s) & _right_sibling_instances(node)

    # tránh 2 instance cùng feature trên cùng đường đi (1 clique)
    ancestor_features = set()
    cur = node.parent
    while cur is not None and cur.instance is not None:
        ancestor_features.add(cur.instance.feature)
        cur = cur.parent

    # sort(candidates) để thứ tự ổn định và ăn khớp RS
    return [inst for inst in sorted(candidates)
            if inst.feature not in ancestor_features]


def _collect_clique(node: ITreeNode) -> Tuple[Instance, ...]:
    """
    Thu clique = tập các instance trên đường đi từ node lên root (loại root).
    Sau đó sort theo thứ tự tổng (định nghĩa của Instance.__lt__).
    """
    clique: List[Instance] = []
    cur = node
    while cur is not None and cur.instance is not None:
        clique.append(cur.instance)
        cur = cur.parent
    return tuple(sorted(clique))


def _process_heads_ids(
    heads: List[Instance], nbs: NeighborhoodList
) -> List[Tuple[Instance, ...]]:
    """
    Worker function: Mine cliques starting from a list of head-nodes.
    """
    local_cliques: List[Tuple[Instance, ...]] = []
    itree = ITree()

    for s in heads:
        queue = deque()

        # create head-node for instance s
        head = itree.add_head_node(s)
        queue.append(head)

        # BFS on I-tree
        while queue:
            curr = queue.popleft()

            # expansion step: calculate children via Lemma 3
            children_instances = _get_children(curr, nbs)

            # if no children -> curr is leaf node -> potential clique
            if not children_instances:
                clique = _collect_clique(curr)
                # only keep cliques with size >= 2
                if len(clique) >= 2:
                    local_cliques.append(clique)
                continue

            # create child nodes for curr
            children_nodes: List[ITreeNode] = []
            prev: Optional[ITreeNode] = None
            for inst in children_instances:
                node = ITreeNode(instance=inst, parent=curr)
                curr.children.append(node)
                # node_link between siblings
                if prev is not None:
                    prev.node_link = node
                prev = node
                children_nodes.append(node)

            # push children to queue for BFS
            for ch in children_nodes:
                queue.append(ch)

        # reset root's children for next head
        itree.root.children = []

    return local_cliques


def mine_cliques_ids(
    dataset: SpatialDataset, nbs: NeighborhoodList, n_workers: int = 1
) -> List[Tuple[Instance, ...]]:
    """
    Algorithm 2 – IDS: Mine all I-cliques.
    Supports parallel execution if n_workers > 1.
    """
    instances = list(dataset.instances)
    
    # Serial path
    if n_workers <= 1:
        return _process_heads_ids(instances, nbs)

    # Parallel path
    chunk_size = max(1, len(instances) // n_workers)
    chunks = [
        instances[i : i + chunk_size]
        for i in range(0, len(instances), chunk_size)
    ]
    
    cliques: List[Tuple[Instance, ...]] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [
            executor.submit(_process_heads_ids, chunk, nbs) 
            for chunk in chunks
        ]
        for fut in futures:
            cliques.extend(fut.result())

    return cliques
