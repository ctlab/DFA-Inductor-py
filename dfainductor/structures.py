from __future__ import annotations

from enum import Enum
from typing import Dict, List, Union

import click


class APTA:
    class Node:
        class NodeStatus(Enum):
            REJECTING, ACCEPTING, UNDEFINED = range(3)

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

        def get_child(self, label: str) -> APTA.Node:
            return self._children[label]

        def add_child(self, label: str, node: APTA.Node) -> None:
            self._children[label] = node

        def is_accepting(self) -> bool:
            return self.status is self.NodeStatus.ACCEPTING

        def is_rejecting(self) -> bool:
            return self.status is self.NodeStatus.REJECTING

    @property
    def root(self) -> Node:
        return self._root

    @property
    def alphabet(self) -> List[str]:
        return sorted(self._alphabet)

    def alphabet_size(self) -> int:
        return len(self._alphabet)

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    def get_node(self, i: int) -> Node:
        return self._nodes[i]

    def __init__(self, input_: Union[str, list]) -> None:
        self._root = self.Node(0, self.Node.NodeStatus.UNDEFINED)
        self._alphabet = set()
        self._nodes = [self._root]
        if isinstance(input_, str):
            with click.open_file(input_) as file:
                examples_number, alphabet_size = [int(x) for x in next(file).split()]
                for __ in range(examples_number):
                    self.add_example(next(file))
                assert len(self._alphabet) == alphabet_size
        elif isinstance(input_, list):
            self.add_examples(input_)

    def add_examples(self, examples: List[str]) -> int:
        old_size = self.size()
        for example in examples:
            self.add_example(example)
        return old_size

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

    def size(self) -> int:
        return len(self._nodes)

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


class DFA:
    class State:
        class StateStatus(Enum):
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
