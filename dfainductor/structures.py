from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Union, Optional, Tuple, Set

import click


class APTA:
    class Node:
        class NodeStatus(IntEnum):
            REJECTING = 0
            ACCEPTING = 1
            UNDEFINED = 2

            def is_acc(self) -> bool:
                return self is self.ACCEPTING

            def is_rej(self) -> bool:
                return self is self.REJECTING

        def __init__(self, id_: int, status: NodeStatus) -> None:
            self._id = id_
            self.status = status
            self._children = {}

        @property
        def id_(self) -> int:
            return self._id

        @property
        def children(self) -> Dict[str, APTA.Node]:
            return self._children

        def has_child(self, label: str) -> bool:
            return label in self._children.keys()

        def get_child(self, label: str) -> Optional[APTA.Node]:
            return self._children[label] if self.has_child(label) else None

        def add_child(self, label: str, node: APTA.Node) -> None:
            self._children[label] = node

        def is_accepting(self) -> bool:
            return self.status.is_acc()

        def is_rejecting(self) -> bool:
            return self.status.is_rej()

    @property
    def root(self) -> Node:
        return self._root

    @property
    def alphabet(self) -> List[str]:
        return sorted(self._alphabet)

    @property
    def alphabet_size(self) -> int:
        return len(self._alphabet)

    @property
    def size(self) -> int:
        return len(self.nodes)

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @property
    def accepting_nodes(self) -> List[Node]:
        return self._accepting_nodes

    @property
    def rejecting_nodes(self) -> List[Node]:
        return self._rejecting_nodes

    def get_node(self, i: int) -> Node:
        return self._nodes[i]

    def __init__(self, input_: Union[str, list, None]) -> None:
        self._root = self.Node(0, self.Node.NodeStatus.UNDEFINED)
        self._alphabet = set()
        self._nodes = [self._root]
        self._accepting_nodes = []
        self._rejecting_nodes = []
        if isinstance(input_, str):
            with click.open_file(input_) as file:
                examples_number, alphabet_size = [int(x) for x in next(file).split()]
                for __ in range(examples_number):
                    self.add_example(next(file))
                assert len(self._alphabet) == alphabet_size
        elif isinstance(input_, list):
            self.add_examples(input_)
        elif input_ is None:
            pass

    def _get_node_by_prefix(self, word: List[str]) -> Optional[Node]:
        cur_state = self._root
        for label in word:
            cur_state = cur_state.get_child(label)
            if not cur_state:
                return None
        return cur_state

    def add_examples(self, examples: List[str]) -> Tuple[int, List[int]]:
        changed_statuses = []
        old_size = self.size
        for example in examples:
            existing_node = self._get_node_by_prefix(example.split()[2:])
            if existing_node:
                changed_statuses.append(existing_node.id_)
            self.add_example(example)
        return old_size, changed_statuses

    def add_example(self, example: str) -> None:
        # example: status len l_1 l_2 l_3 ... l_len
        parsed = example.split()
        current_node = self._root
        status = self.Node.NodeStatus(int(parsed[0]))
        assert int(parsed[1]) == len(parsed[2:])
        for label in parsed[2:]:
            self._alphabet.add(label)
            if current_node.has_child(label):
                current_node = current_node.get_child(label)
            else:
                new_node = self.Node(len(self._nodes), self.Node.NodeStatus.UNDEFINED)
                self._nodes.append(new_node)
                current_node.add_child(label, new_node)
                current_node = new_node
        current_node.status = status
        if status.is_acc():
            self._accepting_nodes.append(current_node)
        else:
            self._rejecting_nodes.append(current_node)

    def has_transition(self, from_: int, label: str, to: int) -> bool:
        return self._nodes[from_].has_child(label) and self._nodes[from_].get_child(label).id_ == to

    def to_dot(self) -> str:
        s = (
            "digraph APTA {\n"
            "    node [shape = circle];\n"
            "    rankdir=LR;\n"
            "    0 [style = \"bold\"];\n"
        )
        for node in self._nodes:
            if node.is_accepting():
                s += "    {0} [peripheries=2]\n".format(str(node.id_))
            if node.is_rejecting():
                s += "    {0} [peripheries=3]\n".format(str(node.id_))
            for label, to in node.children.items():
                s += "    {0} -> {1} [label = {2}];\n".format(str(node.id_), str(to.id_), label)
        s += "}\n"
        return s

    def __str__(self) -> str:
        return self.to_dot()

    def __copy__(self) -> APTA:
        new_apta = type(self)(None)
        new_apta._root = self.root
        new_apta._alphabet = self.alphabet
        new_apta._nodes = self._nodes[:]
        new_apta._accepting_nodes = self._accepting_nodes[:]
        new_apta._rejecting_nodes = self._rejecting_nodes[:]
        return new_apta


class DFA:
    class State:
        class StateStatus(IntEnum):
            REJECTING, ACCEPTING = range(2)

            @classmethod
            def from_bool(cls, b: bool) -> DFA.State.StateStatus:
                return cls.ACCEPTING if b else cls.REJECTING

            def to_bool(self) -> bool:
                return True if self is self.ACCEPTING else False

        def __init__(self, id_: int, status: DFA.State.StateStatus) -> None:
            self._id = id_
            self.status = status
            self._children = {}

        @property
        def id_(self) -> int:
            return self._id

        @property
        def children(self) -> Dict[str, DFA.State]:
            return self._children

        def has_child(self, label: str) -> bool:
            return label in self._children.keys()

        def get_child(self, label: str) -> DFA.State:
            return self._children[label]

        def add_child(self, label: str, node: DFA.State) -> None:
            self._children[label] = node

        def is_accepting(self) -> bool:
            return self.status is self.StateStatus.ACCEPTING

    def __init__(self) -> None:
        self._states = []

    def add_state(self, status: DFA.State.StateStatus) -> None:
        self._states.append(DFA.State(self.size(), status))

    def get_state(self, id_: int) -> DFA.State:
        return self._states[id_]

    def get_start(self) -> DFA.State:
        return self._states[0] if self.size() > 0 else None

    def size(self) -> int:
        return len(self._states)

    def add_transition(self, from_: int, label: str, to: int) -> None:
        self._states[from_].add_child(label, self._states[to])

    def run(self, word: List[str], start: DFA.State = None) -> bool:
        cur_state = start if start else self.get_start()
        for label in word:
            cur_state = cur_state.get_child(label)
        return cur_state.is_accepting()

    def check_consistency(self, examples: List[str]) -> bool:
        for example in examples:
            example_split = example.split()
            if (example_split[0] == '1') != self.run(example_split[2:]):
                return False
        return True

    def to_dot(self) -> str:
        s = (
            "digraph DFA {\n"
            "    node [shape = circle];\n"
            "    0 [style = \"bold\"];\n"
        )
        for state in self._states:
            if state.is_accepting():
                s += "    {0} [peripheries=2]\n".format(str(state.id_))
            for label, to in state.children.items():
                s += "    {0} -> {1} [label = {2}];\n".format(str(state.id_), str(to.id_), label)
        s += "}\n"
        return s

    def __str__(self) -> str:
        return self.to_dot()


class InconsistencyGraph:
    def __init__(self, apta: APTA, *, is_empty: bool = False) -> None:
        self._apta = apta
        self._size = apta.size
        self._edges: List[Set[int]] = [set() for _ in range(self.size)]
        if not is_empty:
            for node_id in range(apta.size):
                for other_id in range(node_id):
                    if not self._try_to_merge(self._apta.get_node(node_id), self._apta.get_node(other_id), {}):
                        self._edges[node_id].add(other_id)

    def update(self, new_nodes_from: int):
        for node_id in range(new_nodes_from, self._size):
            self._edges.append(set())
            for other_id in range(node_id):
                if not self._try_to_merge(self._apta.get_node(node_id), self._apta.get_node(other_id), {}):
                    self._edges[node_id].add(other_id)

    def _has_edge(self, id1: int, id2: int):
        return id2 in self._edges[id1] or id1 in self._edges[id2]

    @property
    def size(self) -> int:
        return self._size

    @property
    def edges(self) -> List[Set[int]]:
        return self._edges

    def _try_to_merge(self,
                      node: APTA.Node,
                      other: APTA.Node,
                      reps: Dict[int, Tuple[int, APTA.Node.NodeStatus]]) -> bool:

        (node_rep_num, node_rep_st) = reps.get(node.id_, (node.id_, node.status))
        (other_rep_num, other_rep_st) = (other.id_, other.status)

        if node_rep_st.is_acc() and other_rep_st.is_rej() or node_rep_st.is_rej() and other_rep_st.is_acc():
            return False
        else:
            if node_rep_num < other_rep_num:
                reps[other_rep_num] = (node_rep_num, min(node_rep_st, other_rep_st))
            else:
                reps[node_rep_num] = (other_rep_num, min(node_rep_st, other_rep_st))
            for label, child in node.children.items():
                if other.has_child(label):
                    if not self._try_to_merge(child, other.get_child(label), reps):
                        return False
        return True

    def to_dot(self) -> str:
        s = (
            "digraph IG {\n"
            "    node [shape = circle];\n"
            "    edge [arrowhead=\"none\"];\n"
        )
        for node1 in range(self.size):
            if self._edges[node1]:
                for node2 in self._edges[node1]:
                    s += "    {0} -> {1};\n".format(str(node1), str(node2))
            else:
                s += "    {0};\n".format(str(node1))
        s += "}\n"
        return s

    def __str__(self) -> str:
        return self.to_dot()
