from abc import ABC, abstractmethod

from pysat.card import CardEnc
from pysat.formula import CNF

from ..structures import APTA


class BaseClauseGenerator(ABC):
    def __init__(self, apta, dfa_size, vpool):
        self._apta = apta
        self._dfa_size = dfa_size
        self._vpool = vpool
        self._formula = CNF()

    @abstractmethod
    def generate(self):
        pass


class MinDFAToSATClausesGenerator(BaseClauseGenerator):
    def __init__(self, apta, dfa_size, vpool):
        super().__init__(apta, dfa_size, vpool)

    def generate(self):
        formula = self._fix_start_state()
        # print(formula.clauses)
        self._formula.extend(formula.clauses)

        formula = self._one_node_maps_to_at_least_one_state()
        # print(formula.clauses)
        self._formula.extend(formula.clauses)

        formula = self._one_node_maps_to_at_most_one_state()
        # print(formula.clauses)
        self._formula.extend(formula.clauses)

        formula = self._dfa_is_complete()
        # print(formula.clauses)
        self._formula.extend(formula.clauses)

        formula = self._dfa_is_deterministic()
        # print(formula.clauses)
        self._formula.extend(formula.clauses)

        formula = self._state_status_compatible_with_node_status()
        # print(formula.clauses)
        self._formula.extend(formula.clauses)

        formula = self._mapped_adjacent_nodes_force_transition()
        # print(formula.clauses)
        self._formula.extend(formula.clauses)

        formula = self._mapped_node_and_transition_force_mapping()
        # print(formula.clauses)
        self._formula.extend(formula.clauses)

        return self._formula

    def _fix_start_state(self):
        clauses = [[self._vpool.id('x_0_0')]]
        return CNF(from_clauses=clauses)

    def _one_node_maps_to_at_least_one_state(self):
        formula = CNF()
        for i in range(self._apta.size()):
            formula.extend(
                CardEnc.atleast(
                    [self._vpool.id('x_{0}_{1}'.format(i, j)) for j in range(self._dfa_size)],
                    top_id=self._vpool.top
                ).clauses
            )
            self._vpool.top = formula.nv
        return formula

    def _one_node_maps_to_at_most_one_state(self):
        formula = CNF()
        for node in self._apta.nodes:
            formula.extend(
                CardEnc.atmost(
                    [self._vpool.id('x_{0}_{1}'.format(node.id_, j)) for j in range(self._dfa_size)],
                    top_id=self._vpool.top
                ).clauses
            )
            self._vpool.top = formula.nv
        return formula

    def _dfa_is_complete(self):
        formula = CNF()
        for i in range(self._dfa_size):
            for label in self._apta.alphabet:
                formula.extend(
                    CardEnc.atleast(
                        [self._vpool.id('y_{0}_{1}_{2}'.format(i, label, j)) for j in range(self._dfa_size)],
                        top_id=self._vpool.top
                    ).clauses
                )
                self._vpool.top = formula.nv
        return formula

    def _dfa_is_deterministic(self):
        formula = CNF()
        for i in range(self._dfa_size):
            for label in self._apta.alphabet:
                formula.extend(
                    CardEnc.atmost(
                        [self._vpool.id('y_{0}_{1}_{2}'.format(i, label, j)) for j in range(self._dfa_size)],
                        top_id=self._vpool.top
                    ).clauses
                )
                self._vpool.top = formula.nv
        return formula

    def _state_status_compatible_with_node_status(self):
        formula = CNF()
        for node in self._apta.nodes:
            if node.status is APTA.Node.NodeStatus.ACCEPTING:
                formula.extend(
                    [
                        [
                            -self._vpool.id('x_{0}_{1}'.format(node.id_, j)), self._vpool.id('z_{0}'.format(j))
                        ] for j in range(self._dfa_size)
                    ]
                )
            elif node.status is APTA.Node.NodeStatus.REJECTING:
                formula.extend(
                    [
                        [
                            -self._vpool.id('x_{0}_{1}'.format(node.id_, j)), -self._vpool.id('z_{0}'.format(j))
                        ] for j in range(self._dfa_size)
                    ]
                )
        return formula

    def _mapped_adjacent_nodes_force_transition(self):
        formula = CNF()
        for parent in self._apta.nodes:
            for label, child in parent.children.items():
                if child:
                    formula.extend(
                        [
                            [
                                -self._vpool.id('x_{0}_{1}'.format(parent.id_, from_)),
                                -self._vpool.id('x_{0}_{1}'.format(child.id_, to)),
                                self._vpool.id('y_{0}_{1}_{2}'.format(from_, label, to))
                            ] for from_ in range(self._dfa_size) for to in range(self._dfa_size)
                        ]
                    )
        return formula

    def _mapped_node_and_transition_force_mapping(self):
        formula = CNF()
        for parent in self._apta.nodes:
            for label, child in parent.children.items():
                if child:
                    formula.extend(
                        [
                            [
                                -self._vpool.id('x_{0}_{1}'.format(parent.id_, from_)),
                                -self._vpool.id('y_{0}_{1}_{2}'.format(from_, label, to)),
                                self._vpool.id('x_{0}_{1}'.format(child.id_, to))
                            ] for from_ in range(self._dfa_size) for to in range(self._dfa_size)
                        ]
                    )
        return formula
