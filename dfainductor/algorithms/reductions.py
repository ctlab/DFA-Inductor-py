from abc import ABC, abstractmethod
from typing import List, Union

from pysat.card import CardEnc, EncType
from pysat.formula import CNF, IDPool

from ..structures import APTA


def _implication_to_clauses(lhs: int, rhs: int) -> List[List[int]]:
    """
    generates CNF formula of an expression /lhs => rhs/
    :type lhs: int
    :type rhs: int
    """
    return [[-lhs, rhs]]


def _conjunction_implies_to_clauses(lhs: List[int], rhs: int) -> List[List[int]]:
    """
    generates CNF formula of an expression /lhs_1 and lhs_2 and ... and lhs_n => rhs/
    :type lhs: list(int)
    :type rhs: int
    """
    return [[-arg for arg in lhs] + [rhs]]


def _iff_to_clauses(lhs: int, rhs: int) -> List[List[int]]:
    """
    generates CNF formula of an expression /lhs <=> rhs/
    :type lhs: int
    :type rhs: int
    """
    return [[-lhs, rhs], [lhs, -rhs]]


def _iff_disjunction_to_clauses(lhs: int, rhs: List[int]) -> List[List[int]]:
    """
    generates CNF formula of an expression /lhs <=> rhs_1 or rhs_2 or ... or rhs_n/
    :type lhs: int
    :type rhs: list(int)
    """
    return [[-lhs] + (list(rhs))] + [[lhs, -arg] for arg in rhs]


def _iff_conjunction_to_clauses(lhs: int, rhs: List[int]) -> List[List[int]]:
    """
    generates CNF formula of an expression /lhs <=> rhs_1 and rhs_2 and ... and rhs_n/
    :type lhs: int
    :type rhs: list(int)
    """
    return [[lhs] + [-arg for arg in rhs]] + [[-lhs, arg] for arg in rhs]


class BaseClausesGenerator(ABC):

    def __init__(self, apta: APTA, vpool: IDPool, with_assumptions: bool) -> None:
        self._apta = apta
        self._vpool = vpool
        self._with_assumptions = with_assumptions
        self._alphabet = self._apta.alphabet
        self._alphabet_size = len(self._alphabet)

    @abstractmethod
    def generate(self, size: int) -> CNF:
        pass

    @abstractmethod
    def generate_with_new_counterexamples(self, size: int, new_from: int) -> CNF:
        pass

    @abstractmethod
    def generate_with_new_size(self, old_size: int, new_size: int) -> CNF:
        pass

    def _var(self, name: str, *indices: Union[str, int]) -> int:
        var: str = name + '_' + '_'.join(str(index) for index in indices)
        return self._vpool.id(var)


class MinDFAToSATClausesGenerator(BaseClausesGenerator):
    def generate(self, size: int) -> CNF:
        formula = CNF()
        formula.extend(self._fix_start_state())
        formula.extend(self._one_node_maps_to_at_least_one_state(size))
        formula.extend(self._one_node_maps_to_at_most_one_state(size))
        formula.extend(self._dfa_is_complete(size))
        formula.extend(self._dfa_is_deterministic(size))
        formula.extend(self._state_status_compatible_with_node_status(size))
        formula.extend(self._mapped_adjacent_nodes_force_transition(size))
        formula.extend(self._mapped_node_and_transition_force_mapping(size))
        return formula

    def generate_with_new_counterexamples(self, size: int, new_from: int) -> CNF:
        formula = CNF()
        formula.extend(self._one_node_maps_to_at_least_one_state(size, new_node_from=new_from))
        formula.extend(self._one_node_maps_to_at_most_one_state(size, new_node_from=new_from))
        formula.extend(self._state_status_compatible_with_node_status(size, new_node_from=new_from))
        formula.extend(self._mapped_adjacent_nodes_force_transition(size, new_node_from=new_from))
        formula.extend(self._mapped_node_and_transition_force_mapping(size, new_node_from=new_from))
        return formula

    def generate_with_new_size(self, old_size: int, new_size: int) -> CNF:
        formula = CNF()
        formula.extend(self._one_node_maps_to_at_least_one_state(new_size, old_size=old_size))
        formula.extend(self._one_node_maps_to_at_most_one_state(new_size, old_size=old_size))
        formula.extend(self._dfa_is_complete(new_size, old_size=old_size))
        formula.extend(self._dfa_is_deterministic(new_size, old_size=old_size))
        formula.extend(self._state_status_compatible_with_node_status(new_size, old_size=old_size))
        formula.extend(self._mapped_adjacent_nodes_force_transition(new_size, old_size=old_size))
        formula.extend(self._mapped_node_and_transition_force_mapping(new_size, old_size=old_size))
        return formula

    def _fix_start_state(self) -> CNF:
        clauses = [[self._var('x', 0, 0)]]
        return CNF(from_clauses=clauses)

    def _one_node_maps_to_at_least_one_state(self,
                                             size: int,
                                             new_node_from: int = 0,
                                             old_size: int = 0) -> CNF:
        if not self._with_assumptions:
            return self._one_node_maps_to_at_least_one_state_classic(size, new_node_from=new_node_from)
        else:
            return self._one_node_maps_to_at_least_one_state_with_assumptions(size,
                                                                              new_node_from=new_node_from,
                                                                              old_size=old_size)

    def _one_node_maps_to_at_least_one_state_classic(self, size: int, new_node_from: int = 0) -> CNF:
        formula = CNF()
        for i in range(new_node_from, self._apta.size()):
            formula.extend(
                [[self._var('x', i, j) for j in range(size)]]
            )
        return formula

    def _one_node_maps_to_at_least_one_state_with_assumptions(self,
                                                              size: int,
                                                              new_node_from: int = 0,
                                                              old_size: int = 0) -> CNF:
        formula = CNF()
        for i in range(new_node_from, self._apta.size()):
            formula.extend(
                [
                    ([] if old_size == 0 else [self._var('alo_x', old_size, i)]) +
                    [self._var('x', i, j) for j in range(old_size, size)] +
                    [-self._var('alo_x', size, i)]
                ]
            )
        return formula

    def _one_node_maps_to_at_most_one_state(self, size: int, new_node_from: int = 0, old_size: int = 0) -> CNF:
        formula = CNF()
        for v in range(new_node_from, self._apta.size()):
            for i in range(old_size, size):
                for j in range(0, i):
                    formula.append([-self._var('x', v, i), -self._var('x', v, j)])
        return formula

    def _dfa_is_complete(self, size: int, old_size: int = 0):
        if not self._with_assumptions:
            return self._dfa_is_complete_classic(size)
        else:
            return self._dfa_is_complete_with_assumptions(size, old_size=old_size)

    def _dfa_is_complete_classic(self, size: int) -> CNF:
        formula = CNF()
        for i in range(size):
            for l_id in range(self._alphabet_size):
                formula.extend(
                    [[self._var('y', i, l_id, j) for j in range(size)]]
                )
        return formula

    def _dfa_is_complete_with_assumptions(self, size: int, old_size: int = 0) -> CNF:
        formula = CNF()
        for l_id in range(self._alphabet_size):
            for i in range(old_size):
                formula.append(
                    ([] if old_size == 0 else [self._var('alo_y', old_size, i, l_id)]) +
                    [self._var('y', i, l_id, j) for j in range(old_size, size)] +
                    [-self._var('alo_y', size, i, l_id)]
                )
            for i in range(old_size, size):
                formula.append(
                    [self._var('y', i, l_id, j) for j in range(size)] +
                    [-self._var('alo_y', size, i, l_id)]
                )
        return formula

    def _dfa_is_deterministic(self, size: int, old_size: int = 0) -> CNF:
        formula = CNF()
        for l_id in range(self._alphabet_size):
            for i in range(old_size):
                for j in range(old_size, size):
                    for k in range(0, j):
                        formula.append(
                            [-self._var('y', i, l_id, j), -self._var('y', i, l_id, k)]
                        )
            for i in range(old_size, size):
                for j in range(size):
                    for k in range(j):
                        formula.append(
                            [-self._var('y', i, l_id, j), -self._var('y', i, l_id, k)]
                        )
        return formula

    def _state_status_compatible_with_node_status(self, size: int, new_node_from: int = 0, old_size: int = 0) -> CNF:
        formula = CNF()
        for i in range(new_node_from, self._apta.size()):
            if self._apta.get_node(i).is_accepting():
                for j in range(old_size, size):
                    formula.extend(
                        _implication_to_clauses(
                            self._var('x', i, j),
                            self._var('z', j)
                        )
                    )
            elif self._apta.get_node(i).is_rejecting():
                for j in range(old_size, size):
                    formula.extend(
                        _implication_to_clauses(
                            self._var('x', i, j),
                            -self._var('z', j)
                        )
                    )
        return formula

    def _mapped_adjacent_nodes_force_transition(self, size: int, new_node_from: int = 0, old_size: int = 0) -> CNF:
        formula = CNF()
        for parent in self._apta.nodes:
            for label, child in parent.children.items():
                if parent.id_ >= new_node_from or child.id_ >= new_node_from:
                    for from_ in range(old_size, size):
                        for to in range(old_size, size):
                            formula.extend(
                                _conjunction_implies_to_clauses(
                                    [
                                        self._var('x', parent.id_, from_),
                                        self._var('x', child.id_, to),
                                    ],
                                    self._var('y', from_, label, to)
                                )
                            )
                    if old_size > 0:
                        for from_ in range(old_size):
                            for to in range(old_size, size):
                                formula.extend(
                                    _conjunction_implies_to_clauses(
                                        [
                                            self._var('x', parent.id_, from_),
                                            self._var('x', child.id_, to),
                                        ],
                                        self._var('y', from_, label, to)
                                    )
                                )
                        for from_ in range(old_size, size):
                            for to in range(old_size):
                                formula.extend(
                                    _conjunction_implies_to_clauses(
                                        [
                                            self._var('x', parent.id_, from_),
                                            self._var('x', child.id_, to),
                                        ],
                                        self._var('y', from_, label, to)
                                    )
                                )
        return formula

    def _mapped_node_and_transition_force_mapping(self, size: int, new_node_from: int = 0, old_size: int = 0) -> CNF:
        formula = CNF()
        for parent in self._apta.nodes:
            for label, child in parent.children.items():
                if parent.id_ >= new_node_from or child.id_ >= new_node_from:
                    for from_ in range(old_size, size):
                        for to in range(old_size, size):
                            formula.extend(
                                _conjunction_implies_to_clauses(
                                    [
                                        self._var('x', parent.id_, from_),
                                        self._var('y', from_, label, to),
                                    ],
                                    self._var('x', child.id_, to)
                                )
                            )
                    if old_size > 0:
                        for from_ in range(old_size):
                            for to in range(old_size, size):
                                formula.extend(
                                    _conjunction_implies_to_clauses(
                                        [
                                            self._var('x', parent.id_, from_),
                                            self._var('y', from_, label, to),
                                        ],
                                        self._var('x', child.id_, to)
                                    )
                                )
                        for from_ in range(old_size, size):
                            for to in range(old_size):
                                formula.extend(
                                    _conjunction_implies_to_clauses(
                                        [
                                            self._var('x', parent.id_, from_),
                                            self._var('y', from_, label, to),
                                        ],
                                        self._var('x', child.id_, to)
                                    )
                                )
        return formula


class BFSBasedSymBreakingClausesGenerator(BaseClausesGenerator):
    def generate(self, size: int) -> CNF:
        formula = CNF()
        formula.extend(self._define_t_variables(size))
        formula.extend(self._define_p_variables(size))
        formula.extend(self._state_has_at_least_one_parent(size))
        formula.extend(self._preserve_parent_order_on_children(size))
        formula.extend(self._order_children(size))
        return formula

    def generate_with_new_counterexamples(self, size: int, new_from: int) -> CNF:
        return CNF()

    def generate_with_new_size(self, old_size: int, new_size: int) -> CNF:
        formula = CNF()
        formula.extend(self._define_t_variables(new_size, old_size=old_size))
        formula.extend(self._define_p_variables(new_size, old_size=old_size))
        formula.extend(self._state_has_at_least_one_parent(new_size, old_size=old_size))
        formula.extend(self._preserve_parent_order_on_children(new_size, old_size=old_size))
        formula.extend(self._order_children(new_size, old_size=old_size))
        return formula

    def _define_t_variables(self, size: int, old_size: int = 0) -> CNF:
        formula = CNF()
        for to in range(old_size, size):
            for from_ in range(to):
                formula.extend(
                    _iff_disjunction_to_clauses(
                        self._var('t', from_, to),
                        [self._var('y', from_, l_id, to) for l_id in range(self._alphabet_size)]
                    )
                )
        return formula

    def _define_p_variables(self, size: int, old_size: int = 0) -> CNF:
        formula = CNF()
        for child in range(old_size, size):
            for parent in range(child):
                formula.extend(
                    _iff_conjunction_to_clauses(
                        self._var('p', child, parent),
                        [-self._var('t', prev, child) for prev in range(parent)] + [self._var('t', parent, child)]
                    )
                )
        return formula

    def _state_has_at_least_one_parent(self, size: int, old_size: int = 0) -> CNF:
        formula = CNF()
        for child in range(max(1, old_size), size):
            formula.append(
                [self._var('p', child, parent) for parent in range(child)]
            )
        return formula

    def _preserve_parent_order_on_children(self, size: int, old_size: int = 0) -> CNF:
        formula = CNF()
        for child in range(max(2, old_size - 1), size - 1):
            for parent in range(1, child):
                for pre_parent in range(parent):
                    formula.extend(
                        _implication_to_clauses(self._var('p', child, parent), -self._var('p', child + 1, pre_parent))
                    )
        return formula

    def _order_children(self, size: int, old_size: int = 0) -> CNF:
        formula = CNF()
        if self._alphabet_size == 2:
            formula.extend(self._order_children_with_binary_alphabet(size, old_size))
        elif self._alphabet_size > 2:
            formula.extend(self._define_m_variables(size, old_size))
            formula.extend(self._order_children_using_m(size, old_size))
        return formula

    def _order_children_with_binary_alphabet(self, size: int, old_size: int = 0) -> CNF:
        formula = CNF()
        for child in range(max(0, old_size - 1), size - 1):
            for parent in range(child):
                formula.extend(
                    _conjunction_implies_to_clauses(
                        [self._var('p', child, parent), self._var('p', child + 1, parent)],
                        self._var('y', parent, self._alphabet[0], child)
                    )
                )
                formula.extend(
                    _conjunction_implies_to_clauses(
                        [self._var('p', child, parent), self._var('p', child + 1, parent)],
                        self._var('y', parent, self._alphabet[1], child + 1)
                    )
                )
        return formula

    def _define_m_variables(self, size: int, old_size: int = 0) -> CNF:
        formula = CNF()
        for child in range(old_size, size):
            for parent in range(child):
                for l_num in range(self._alphabet_size):
                    formula.extend(
                        _iff_conjunction_to_clauses(
                            self._var('m', parent, l_num, child),
                            [
                                -self._var('y', parent, l_less, child) for l_less in range(l_num)
                            ] + [self._var('y', parent, l_num, child)]
                        )
                    )
        return formula

    def _order_children_using_m(self, size: int, old_size: int = 0) -> CNF:
        formula = CNF()
        for child in range(max(old_size - 1, 0), size - 1):
            for parent in range(child):
                for l_num in range(self._alphabet_size):
                    for l_less in range(l_num):
                        formula.extend(
                            _conjunction_implies_to_clauses(
                                [
                                    self._var('p', child, parent),
                                    self._var('p', child + 1, parent),
                                    self._var('m', parent, l_num, child),
                                ],
                                -self._var('m', parent, l_less, child + 1)
                            )
                        )
        return formula


class TightBFSBasedSymBreakingClausesGenerator(BFSBasedSymBreakingClausesGenerator):

    def generate(self, size: int) -> CNF:
        formula = CNF()
        formula.extend(self._define_t_variables(size))
        formula.extend(self._define_nt_variables(size))
        formula.extend(self._define_p_variables_using_nt(size))
        formula.extend(self._state_has_at_least_one_parent(size))
        formula.extend(self._state_has_at_most_one_parent(size))
        formula.extend(self._preserve_parent_order_on_children(size))
        formula.extend(self._order_parents_using_ng_variables(size))
        formula.extend(self._define_eq_variables(size))
        formula.extend(self._order_children(size))
        return formula

    def _define_nt_variables(self, size: int) -> CNF:
        formula = CNF()
        for child in range(2, size):
            formula.extend(
                _iff_to_clauses(self._var('nt', 0, child), -self._var('t', 0, child))
            )
            for parent in range(1, child):
                formula.extend(
                    _iff_conjunction_to_clauses(
                        self._var('nt', parent, child),
                        [self._var('nt', parent - 1, child), -self._var('t', parent, child)]
                    )
                )
        return formula

    def _define_p_variables_using_nt(self, size: int) -> CNF:
        formula = CNF()
        for child in range(1, size):
            formula.extend(
                _iff_to_clauses(self._var('p', child, 0), self._var('t', 0, child))
            )
            for parent in range(1, child):
                formula.extend(
                    _iff_conjunction_to_clauses(
                        self._var('p', child, parent),
                        [self._var('t', parent, child), self._var('nt', parent - 1, child)]
                    )
                )
        return formula

    def _state_has_at_most_one_parent(self, size: int) -> CNF:
        formula = CNF()
        for child in range(1, size):
            formula.extend(
                CardEnc.atmost(
                    [self._var('p', child, parent) for parent in range(child)],
                    vpool=self._vpool,
                    encoding=EncType.pairwise
                )
            )
        return formula

    def _order_parents_using_ng_variables(self, size: int) -> CNF:
        formula = CNF()
        for child in range(1, size - 1):
            formula.append([self._var('ng', child, child)])
            formula.append([self._var('ng', child, 0)])
            for parent in range(child):
                formula.append([
                    -self._var('ng', child, parent),
                    self._var('ng', child, parent + 1),
                    self._var('p', child, parent)
                ])
                formula.append([
                    -self._var('ng', child, parent),
                    self._var('eq', child, parent),
                    self._var('p', child, parent)
                ])
                formula.append([
                    -self._var('ng', child, parent),
                    self._var('ng', child, parent + 1),
                    -self._var('p', child + 1, parent)
                ])
                formula.append([
                    -self._var('ng', child, parent),
                    self._var('eq', child, parent),
                    -self._var('p', child + 1, parent)
                ])
                formula.append([
                    self._var('ng', child, parent),
                    -self._var('ng', child, parent + 1),
                    -self._var('eq', child, parent)
                ])
                formula.append([
                    self._var('ng', child, parent),
                    -self._var('p', child, parent),
                    self._var('p', child + 1, parent)
                ])
        return formula

    def _define_eq_variables(self, size: int) -> CNF:
        formula = CNF()
        for child in range(1, size - 1):
            for parent in range(child):
                formula.append([
                    self._var('eq', child, parent),
                    self._var('p', child, parent),
                    self._var('p', child + 1, parent)
                ])
                formula.append([
                    self._var('eq', child, parent),
                    -self._var('p', child, parent),
                    -self._var('p', child + 1, parent)
                ])
                formula.append([
                    -self._var('eq', child, parent),
                    -self._var('p', child, parent),
                    self._var('p', child + 1, parent)
                ])
                formula.append([
                    -self._var('eq', child, parent),
                    self._var('p', child, parent),
                    -self._var('p', child + 1, parent)
                ])
        return formula

    def _order_children(self, size: int, old_size: int = 0) -> CNF:
        formula = CNF()
        if self._alphabet_size == 2:
            formula.extend(self._order_children_with_binary_alphabet(size))
        elif self._alphabet_size > 2:
            formula.extend(self._define_ny_variables(size))
            formula.extend(self._define_m_variables_with_ny(size))
            formula.extend(self._define_zm_variables(size))
            formula.extend(self._order_children_using_zm(size))
        return formula

    def _define_ny_variables(self, size: int) -> CNF:
        formula = CNF()
        for child in range(size):
            for parent in range(child):
                formula.extend(
                    _iff_to_clauses(
                        self._var('ny', parent, 0, child),
                        -self._var('y', parent, 0, child),
                    )
                )
                for l_num in range(1, self._alphabet_size):
                    formula.extend(
                        _iff_conjunction_to_clauses(
                            self._var('ny', parent, l_num, child),
                            [
                                -self._var('y', parent, l_num, child),
                                self._var('ny', parent, l_num - 1, child)
                            ]
                        )
                    )
        return formula

    def _define_m_variables_with_ny(self, size: int) -> CNF:
        formula = CNF()
        for child in range(size):
            for parent in range(child):
                formula.extend(
                    _iff_to_clauses(
                        self._var('m', parent, 0, child),
                        self._var('y', parent, 0, child),
                    )
                )
                for l_num in range(1, self._alphabet_size):
                    formula.extend(
                        _iff_conjunction_to_clauses(
                            self._var('m', parent, l_num, child),
                            [
                                self._var('y', parent, l_num, child),
                                self._var('ny', parent, l_num - 1, child)
                            ]
                        )
                    )
        return formula

    def _define_zm_variables(self, size: int) -> CNF:
        formula = CNF()
        for child in range(size):
            for parent in range(child):
                formula.extend(
                    _iff_to_clauses(
                        self._var('zm', parent, 0, child),
                        -self._var('m', parent, 0, child),
                    )
                )
                for l_num in range(1, self._alphabet_size):
                    formula.extend(
                        _iff_conjunction_to_clauses(
                            self._var('zm', parent, l_num, child),
                            [
                                self._var('zm', parent, l_num - 1, child),
                                -self._var('m', parent, l_num, child)
                            ]
                        )
                    )
        return formula

    def _order_children_using_zm(self, size: int) -> CNF:
        formula = CNF()
        for child in range(size - 1):
            for parent in range(child):
                for l_num in range(1, self._alphabet_size):
                    formula.extend(
                        _conjunction_implies_to_clauses(
                            [
                                self._var('p', child, parent),
                                self._var('p', child + 1, parent),
                                self._var('m', parent, l_num, child)
                            ],
                            self._var('zm', parent, l_num - 1, child + 1)
                        )
                    )

        return formula


class NoSymBreakingClausesGenerator(BaseClausesGenerator):

    def generate(self, size: int) -> CNF:
        return CNF()

    def generate_with_new_counterexamples(self, size: int, new_from: int) -> CNF:
        return CNF()

    def generate_with_new_size(self, old_size: int, new_size: int) -> CNF:
        return CNF()


def get_symmetry_breaking_predicates_generator(sb_strategy,
                                               apta,
                                               vpool,
                                               with_assumptions: bool) -> BaseClausesGenerator:
    if sb_strategy == 'BFS':
        return BFSBasedSymBreakingClausesGenerator(apta, vpool, with_assumptions)
    elif sb_strategy == 'TIGHTBFS':
        return TightBFSBasedSymBreakingClausesGenerator(apta, vpool, with_assumptions)
    else:
        return NoSymBreakingClausesGenerator(apta, vpool, with_assumptions)
