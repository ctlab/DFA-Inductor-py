from enum import Enum

import click


class APTA:
    class Node:
        class NodeStatus(Enum):
            REJECTING, ACCEPTING, UNDEFINED = range(3)

        def __init__(self, id_, status):
            self._id = id_
            self.status = status
            self._children = {}

        @property
        def id_(self):
            return self._id

        @property
        def children(self):
            return self._children

        def has_child(self, label):
            return label in self._children.keys()

        def get_child(self, label):
            return self._children[label]

        def add_child(self, label, node):
            self._children[label] = node

        def is_accepting(self):
            return self.status is self.NodeStatus.ACCEPTING

        def is_rejecting(self):
            return self.status is self.NodeStatus.REJECTING

    @property
    def root(self):
        return self._root

    @property
    def alphabet(self):
        return sorted(self._alphabet)

    def alphabet_size(self):
        return len(self._alphabet)

    @property
    def nodes(self):
        return self._nodes

    def __init__(self, input):
        self._root = self.Node(0, self.Node.NodeStatus.UNDEFINED)
        self._alphabet = set()
        self._nodes = [self._root]
        if isinstance(input, str):
            with click.open_file(input) as file:
                examples_number, alphabet_size = [int(x) for x in next(file).split()]
                for __ in range(examples_number):
                    self.add_example(next(file))
                assert len(self._alphabet) == alphabet_size
        elif isinstance(input, list):
            self.add_examples(input)
        # TODO: else error

    def add_examples(self, examples):
        old_size = self.size()
        for example in examples:
            self.add_example(example)
        return old_size

    def add_example(self, example):
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

    def size(self):
        return len(self._nodes)

    def has_transition(self, from_, label, to):
        return self._nodes[from_].has_child(label) and self._nodes[from_].get_child(label).id_ == to

    def to_dot(self):
        s = (
            "digraph APTA {\n"
            "    node [shape = circle];\n"
            "    rankdir=LR;\n"
            "    0 [style = \"bold\"];\n"
        )
        for node in self._nodes:
            if node.is_accepting():
                s += "    " + str(node.id_) + " [peripheries=2]\n"
            if node.is_rejecting():
                s += "    " + str(node.id_) + " [peripheries=3]\n"
            for label, to in node.children.items():
                s += "    " + str(node.id_) + " -> " + str(to.id_) + " [label = \"" + label + "\"];\n"
        s += "}\n"
        return s

    def __str__(self):
        return self.to_dot()


class DFA:
    class State:
        class StateStatus(Enum):
            REJECTING, ACCEPTING = range(2)

            @classmethod
            def from_bool(cls, b):
                return cls.ACCEPTING if b else cls.REJECTING

            def to_bool(self):
                return True if self is self.ACCEPTING else False

        def __init__(self, id_, status):
            self._id = id_
            self.status = status
            self._children = {}

        @property
        def id_(self):
            return self._id

        @property
        def children(self):
            return self._children

        def has_child(self, label):
            return label in self._children.keys()

        def get_child(self, label):
            return self._children[label]

        def add_child(self, label, node):
            self._children[label] = node

        def is_accepting(self):
            return self.status is self.StateStatus.ACCEPTING

    def __init__(self):
        self._states = []

    def add_state(self, status):
        self._states.append(DFA.State(self.size(), status))

    def get_state(self, id_):
        return self._states[id_]

    def get_start(self):
        return self._states[0] if self.size() > 0 else None

    def size(self):
        return len(self._states)

    def add_transition(self, from_, label, to):
        self._states[from_].add_child(label, self._states[to])

    def run(self, word, start=None):
        cur_state = start if start else self.get_start()
        for label in word:
            cur_state = cur_state.get_child(label)
        return cur_state.is_accepting()

    def to_dot(self):
        s = (
            "digraph DFA {\n"
            "    node [shape = circle];\n"
            "    0 [style = \"bold\"];\n"
        )
        for state in self._states:
            if state.is_accepting():
                s += "    " + str(state.id_) + " [peripheries=2]\n"
            for label, to in state.children.items():
                s += "    " + str(state.id_) + " -> " + str(to.id_) + " [label = \"" + label + "\"];\n"
        s += "}\n"
        return s

    def __str__(self):
        return self.to_dot()

