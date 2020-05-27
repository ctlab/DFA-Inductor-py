from abc import ABC, abstractmethod
from itertools import chain
from typing import List, Tuple, Iterator

from pysat.solvers import Solver

from ..structures import APTA, InconsistencyGraph
from ..variables import VarPool

CLAUSE = Tuple[int, ...]
CLAUSES = Iterator[CLAUSE]


def _implication_to_clauses(lhs: int, rhs: int) -> CLAUSES:
    """
    generates CNF formula of an expression /lhs => rhs/
    :type lhs: int
    :type rhs: int
    """
    yield (-lhs, rhs)


def _conjunction_implies_to_clauses(lhs: CLAUSE, rhs: int) -> CLAUSES:
    """
    generates CNF formula of an expression /lhs_1 and lhs_2 and ... and lhs_n => rhs/
    :type lhs: list(int)
    :type rhs: int
    """
    yield tuple(-lit for lit in lhs) + (rhs,)


def _iff_to_clauses(lhs: int, rhs: int) -> CLAUSES:
    """
    generates CNF formula of an expression /lhs <=> rhs/
    :type lhs: int
    :type rhs: int
    """
    yield from _implication_to_clauses(lhs, rhs)
    yield from _implication_to_clauses(rhs, lhs)


def _iff_disjunction_to_clauses(lhs: int, rhs: CLAUSE) -> CLAUSES:
    """
    generates CNF formula of an expression /lhs <=> rhs_1 or rhs_2 or ... or rhs_n/
    :type lhs: int
    :type rhs: list(int)
    """
    yield (-lhs,) + rhs
    yield from ((lhs, -lit) for lit in rhs)


def _iff_conjunction_to_clauses(lhs: int, rhs: CLAUSE) -> CLAUSES:
    """
    generates CNF formula of an expression /lhs <=> rhs_1 and rhs_2 and ... and rhs_n/
    :type lhs: int
    :type rhs: list(int)
    """
    yield (lhs,) + tuple(-lit for lit in rhs)
    yield from ((-lhs, lit) for lit in rhs)


class BaseClausesGenerator(ABC):

    def __init__(self, apta: APTA, ig: InconsistencyGraph, var_pool: VarPool, assumptions_mode: str) -> None:
        self._apta = apta
        self._ig = ig
        self._vars = var_pool
        self._assumptions_mode = assumptions_mode
        self._alphabet = self._apta.alphabet
        self._alphabet_size = len(self._alphabet)

    @abstractmethod
    def generate(self, solver: Solver, size: int) -> None:
        pass

    @abstractmethod
    def generate_with_new_counterexamples(self, solver: Solver, size: int, new_from: int,
                                          changed_statuses: List[int]) -> None:
        pass

    @abstractmethod
    def generate_with_new_size(self, solver: Solver, old_size: int, new_size: int) -> None:
        pass

    def build_assumptions(self, cur_size: int, solver: Solver) -> List[int]:
        assumptions = []
        if self._assumptions_mode == 'chain':
            for v in range(self._apta.size):
                assumptions.append(self._vars.var('alo_x', cur_size, v))
            for from_ in range(cur_size):
                for l_id in range(self._apta.alphabet_size):
                    assumptions.append(self._vars.var('alo_y', cur_size, from_, l_id))
        elif self._assumptions_mode == 'switch':
            for v in range(self._apta.size):
                assumptions.append(-self._vars.var('sw_x', cur_size, v))
            for from_ in range(cur_size):
                for l_id in range(self._apta.alphabet_size):
                    assumptions.append(-self._vars.var('sw_y', cur_size, from_, l_id))
        return assumptions


class ClauseGenerator(BaseClausesGenerator):

    def __init__(self, apta: APTA, ig: InconsistencyGraph, var_pool: VarPool, assumptions_mode: str, sb: str) -> None:
        super().__init__(apta, ig, var_pool, assumptions_mode)
        self._mindfa_generator = MinDFAToSATClausesGenerator(apta, ig, var_pool, assumptions_mode)
        if sb == 'BFS':
            self._sb_generator = BFSBasedSymBreakingClausesGenerator(apta, ig, var_pool, assumptions_mode)
        elif sb == 'TIGHTBFS':
            self._sb_generator = TightBFSBasedSymBreakingClausesGenerator(apta, ig, var_pool, assumptions_mode)
        else:
            self._sb_generator = NoSymBreakingClausesGenerator(apta, ig, var_pool, assumptions_mode)

    def generate(self, solver: Solver, size: int) -> None:
        self._mindfa_generator.generate(solver, size)
        self._sb_generator.generate(solver, size)

    def generate_with_new_counterexamples(self, solver: Solver, size: int, new_from: int,
                                          changed_statuses: List[int]) -> None:
        self._mindfa_generator.generate_with_new_counterexamples(solver, size, new_from, changed_statuses)
        self._sb_generator.generate_with_new_counterexamples(solver, size, new_from, changed_statuses)

    def generate_with_new_size(self, solver: Solver, old_size: int, new_size: int) -> None:
        self._mindfa_generator.generate_with_new_size(solver, old_size, new_size)
        self._sb_generator.generate_with_new_size(solver, old_size, new_size)


class MinDFAToSATClausesGenerator(BaseClausesGenerator):
    def generate(self, solver: Solver, size: int) -> None:
        self._fix_start_state(solver)
        self._one_node_maps_to_alo_state(solver, size)
        self._one_node_maps_to_at_most_one_state(solver, size)
        self._dfa_is_complete(solver, size)
        self._dfa_is_deterministic(solver, size)
        self._state_status_compatible_with_node_status(solver, size)
        self._mapped_adjacent_nodes_force_transition(solver, size)
        self._mapped_node_and_transition_force_mapping(solver, size)
        self._inconsistency_graph_constraints(solver, size)

    def generate_with_new_counterexamples(self, solver: Solver, size: int, new_from: int,
                                          changed_statuses: List[int]) -> None:
        self._one_node_maps_to_alo_state(solver, size, new_node_from=new_from)
        self._one_node_maps_to_at_most_one_state(solver, size, new_node_from=new_from)
        self._state_status_compatible_with_node_status(solver,
                                                       size,
                                                       new_node_from=new_from,
                                                       changed_statuses=changed_statuses)
        self._mapped_adjacent_nodes_force_transition(solver, size, new_node_from=new_from)
        self._mapped_node_and_transition_force_mapping(solver, size, new_node_from=new_from)
        self._inconsistency_graph_constraints(solver, size, new_node_from=new_from)

    def generate_with_new_size(self, solver: Solver, old_size: int, new_size: int) -> None:
        self._one_node_maps_to_alo_state(solver, new_size, old_size=old_size)
        self._one_node_maps_to_at_most_one_state(solver, new_size, old_size=old_size)
        self._dfa_is_complete(solver, new_size, old_size=old_size)
        self._dfa_is_deterministic(solver, new_size, old_size=old_size)
        self._state_status_compatible_with_node_status(solver, new_size, old_size=old_size)
        self._mapped_adjacent_nodes_force_transition(solver, new_size, old_size=old_size)
        self._mapped_node_and_transition_force_mapping(solver, new_size, old_size=old_size)

    def _fix_start_state(self, solver: Solver) -> None:
        solver.add_clause((self._vars.var('x', 0, 0),))

    def _one_node_maps_to_alo_state(self,
                                    solver: Solver,
                                    size: int,
                                    new_node_from: int = 0,
                                    old_size: int = 0) -> None:
        if self._assumptions_mode == 'none':
            self._one_node_maps_to_alo_state_classic(solver, size, new_node_from)
        elif self._assumptions_mode == 'chain':
            self._one_node_maps_to_alo_state_chain(solver, size, new_node_from, old_size)
        elif self._assumptions_mode == 'switch':
            self._one_node_maps_to_alo_state_switch(solver, size, new_node_from, old_size)

    def _one_node_maps_to_alo_state_classic(self, solver: Solver, size: int, new_node_from: int = 0) -> None:
        for i in range(new_node_from, self._apta.size):
            solver.add_clause(tuple(self._vars.var('x', i, j) for j in range(size)))

    def _one_node_maps_to_alo_state_chain(self,
                                          solver: Solver,
                                          size: int,
                                          new_node_from: int = 0,
                                          old_size: int = 0) -> None:
        if old_size == 0:
            for i in range(new_node_from, self._apta.size):
                solver.add_clause(
                    tuple(self._vars.var('x', i, j) for j in range(old_size, size)) + (
                        -self._vars.var('alo_x', size, i),)
                )
        else:
            for i in range(new_node_from, self._apta.size):
                solver.add_clause(
                    tuple(self._vars.var('x', i, j) for j in range(old_size, size)) +
                    (-self._vars.var('alo_x', size, i), self._vars.var('alo_x', old_size, i))
                )

    def _one_node_maps_to_alo_state_switch(self,
                                           solver: Solver,
                                           size: int,
                                           new_node_from: int = 0,
                                           old_size: int = 0) -> None:
        for i in range(new_node_from, self._apta.size):
            solver.add_clause(
                tuple(self._vars.var('x', i, j) for j in range(size)) + (self._vars.var('sw_x', size, i),)
            )
        if old_size > 0:
            for v in range(self._apta.size):
                solver.add_clause((self._vars.var('sw_x', old_size, v),))

    def _one_node_maps_to_at_most_one_state(self, solver: Solver, size: int, new_node_from: int = 0,
                                            old_size: int = 0) -> None:
        for v in range(new_node_from, self._apta.size):
            for i in range(old_size, size):
                for j in range(0, i):
                    solver.add_clause(
                        (-self._vars.var('x', v, i), -self._vars.var('x', v, j))
                    )

    def _dfa_is_complete(self, solver: Solver, size: int, old_size: int = 0):
        if self._assumptions_mode == 'none':
            self._dfa_is_complete_classic(solver, size)
        elif self._assumptions_mode == 'chain':
            self._dfa_is_complete_chain(solver, size, old_size)
        elif self._assumptions_mode == 'switch':
            self._dfa_is_complete_switch(solver, size, old_size)

    def _dfa_is_complete_classic(self, solver: Solver, size: int) -> None:
        for i in range(size):
            for l_id in range(self._alphabet_size):
                solver.add_clause(
                    tuple(self._vars.var('y', i, l_id, j) for j in range(size))
                )

    def _dfa_is_complete_chain(self, solver: Solver, size: int, old_size: int = 0) -> None:
        if old_size == 0:
            for l_id in range(self._alphabet_size):
                for i in range(old_size):
                    solver.add_clause(
                        tuple(self._vars.var('y', i, l_id, j) for j in range(old_size, size)) +
                        (-self._vars.var('alo_y', size, i, l_id),)
                    )
        else:
            for l_id in range(self._alphabet_size):
                for i in range(old_size):
                    solver.add_clause(
                        tuple(self._vars.var('y', i, l_id, j) for j in range(old_size, size)) +
                        (-self._vars.var('alo_y', size, i, l_id), self._vars.var('alo_y', old_size, i, l_id))
                    )
        for l_id in range(self._alphabet_size):
            for i in range(old_size, size):
                solver.add_clause(
                    tuple(self._vars.var('y', i, l_id, j) for j in range(size)) +
                    (-self._vars.var('alo_y', size, i, l_id),)
                )

    def _dfa_is_complete_switch(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for i in range(size):
            for l_id in range(self._alphabet_size):
                solver.add_clause(
                    tuple(self._vars.var('y', i, l_id, j) for j in range(size)) + (
                        self._vars.var('sw_y', size, i, l_id),
                    )
                )

        if old_size > 0:
            for from_ in range(old_size):
                for l_id in range(self._alphabet_size):
                    solver.add_clause((self._vars.var('sw_y', old_size, from_, l_id),))

    def _dfa_is_deterministic(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for l_id in range(self._alphabet_size):
            for i in range(old_size):
                for j in range(old_size, size):
                    for k in range(j):
                        solver.add_clause(
                            (-self._vars.var('y', i, l_id, j), -self._vars.var('y', i, l_id, k))
                        )
            for i in range(old_size, size):
                for j in range(size):
                    for k in range(j):
                        solver.add_clause(
                            (-self._vars.var('y', i, l_id, j), -self._vars.var('y', i, l_id, k))
                        )

    def _state_status_compatible_with_node_status(self,
                                                  solver: Solver,
                                                  size: int,
                                                  new_node_from: int = 0,
                                                  old_size: int = 0,
                                                  changed_statuses=None) -> None:
        if changed_statuses is None:
            changed_statuses = []
        for i in chain(range(new_node_from, self._apta.size), changed_statuses):
            if self._apta.get_node(i).is_accepting():
                for j in range(old_size, size):
                    solver.append_formula(
                        _implication_to_clauses(self._vars.var('x', i, j), self._vars.var('z', j)))
            elif self._apta.get_node(i).is_rejecting():
                for j in range(old_size, size):
                    solver.append_formula(
                        _implication_to_clauses(self._vars.var('x', i, j), -self._vars.var('z', j)))

    def _mapped_adjacent_nodes_force_transition(self, solver: Solver, size: int, new_node_from: int = 0,
                                                old_size: int = 0) -> None:
        for parent in self._apta.nodes:
            for label, child in parent.children.items():
                if parent.id_ >= new_node_from or child.id_ >= new_node_from:
                    for from_ in range(old_size, size):
                        for to in range(old_size, size):
                            solver.append_formula(
                                _conjunction_implies_to_clauses(
                                    (
                                        self._vars.var('x', parent.id_, from_),
                                        self._vars.var('x', child.id_, to),
                                    ),
                                    self._vars.var('y', from_, label, to)
                                )
                            )
                    if old_size > 0:
                        for from_ in range(old_size):
                            for to in range(old_size, size):
                                solver.append_formula(
                                    _conjunction_implies_to_clauses(
                                        (
                                            self._vars.var('x', parent.id_, from_),
                                            self._vars.var('x', child.id_, to),
                                        ),
                                        self._vars.var('y', from_, label, to)
                                    )
                                )

                        for from_ in range(old_size, size):
                            for to in range(old_size):
                                solver.append_formula(
                                    _conjunction_implies_to_clauses(
                                        (
                                            self._vars.var('x', parent.id_, from_),
                                            self._vars.var('x', child.id_, to),
                                        ),
                                        self._vars.var('y', from_, label, to)
                                    )
                                )

    def _mapped_node_and_transition_force_mapping(self, solver: Solver, size: int, new_node_from: int = 0,
                                                  old_size: int = 0) -> None:
        for parent in self._apta.nodes:
            for label, child in parent.children.items():
                if parent.id_ >= new_node_from or child.id_ >= new_node_from:
                    for from_ in range(old_size, size):
                        for to in range(old_size, size):
                            solver.append_formula(
                                _conjunction_implies_to_clauses(
                                    (
                                        self._vars.var('x', parent.id_, from_),
                                        self._vars.var('y', from_, label, to),
                                    ),
                                    self._vars.var('x', child.id_, to)
                                )
                            )
                    if old_size > 0:
                        for from_ in range(old_size):
                            for to in range(old_size, size):
                                solver.append_formula(
                                    _conjunction_implies_to_clauses(
                                        (
                                            self._vars.var('x', parent.id_, from_),
                                            self._vars.var('y', from_, label, to),
                                        ),
                                        self._vars.var('x', child.id_, to)
                                    )
                                )
                        for from_ in range(old_size, size):
                            for to in range(old_size):
                                solver.append_formula(
                                    _conjunction_implies_to_clauses(
                                        (
                                            self._vars.var('x', parent.id_, from_),
                                            self._vars.var('y', from_, label, to),
                                        ),
                                        self._vars.var('x', child.id_, to)
                                    )
                                )

    def _inconsistency_graph_constraints(self, solver: Solver, size: int, new_node_from: int = 0,
                                         old_size: int = 0) -> None:
        for node1 in range(self._ig.size):
            for node2 in self._ig.edges[node1]:
                if node1 >= new_node_from or node2 >= new_node_from:
                    for s in range(old_size, size):
                        solver.add_clause((-self._vars.var('x', node1, s), -self._vars.var('x', node2, s)))


class BFSBasedSymBreakingClausesGenerator(BaseClausesGenerator):
    def generate(self, solver: Solver, size: int) -> None:
        self._define_t_variables(solver, size)
        self._define_p_variables(solver, size)
        self._state_has_at_least_one_parent(solver, size)
        self._preserve_parent_order_on_children(solver, size)
        self._order_children(solver, size)

    def generate_with_new_counterexamples(self, solver: Solver, size: int, new_from: int,
                                          changed_statuses: List[int]) -> None:
        pass

    def generate_with_new_size(self, solver: Solver, old_size: int, new_size: int) -> None:
        self._define_t_variables(solver, new_size, old_size)
        self._define_p_variables(solver, new_size, old_size)
        self._state_has_at_least_one_parent(solver, new_size, old_size)
        self._preserve_parent_order_on_children(solver, new_size, old_size)
        self._order_children(solver, new_size, old_size)

    def _define_t_variables(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for to in range(old_size, size):
            for from_ in range(to):
                solver.append_formula(
                    _iff_disjunction_to_clauses(
                        self._vars.var('t', from_, to),
                        tuple(self._vars.var('y', from_, l_id, to) for l_id in range(self._alphabet_size))
                    )
                )

    def _define_p_variables(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(old_size, size):
            for parent in range(child):
                solver.append_formula(
                    _iff_conjunction_to_clauses(
                        self._vars.var('p', child, parent),
                        tuple(-self._vars.var('t', prev, child) for prev in range(parent)) + (
                            self._vars.var('t', parent, child),)
                    )
                )

    def _state_has_at_least_one_parent(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(max(1, old_size), size):
            solver.add_clause(tuple(self._vars.var('p', child, parent) for parent in range(child)))

    def _preserve_parent_order_on_children(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(max(2, old_size - 1), size - 1):
            for parent in range(1, child):
                for pre_parent in range(parent):
                    solver.append_formula(
                        _implication_to_clauses(
                            self._vars.var('p', child, parent), -self._vars.var('p', child + 1, pre_parent)
                        )
                    )

    def _order_children(self, solver: Solver, size: int, old_size: int = 0) -> None:
        if self._alphabet_size == 2:
            self._order_children_with_binary_alphabet(solver, size, old_size)
        elif self._alphabet_size > 2:
            self._define_m_variables(solver, size, old_size)
            self._order_children_using_m(solver, size, old_size)

    def _order_children_with_binary_alphabet(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(max(0, old_size - 1), size - 1):
            for parent in range(child):
                solver.append_formula(
                    _conjunction_implies_to_clauses(
                        (
                            self._vars.var('p', child, parent),
                            self._vars.var('p', child + 1, parent),
                        ),
                        self._vars.var('y', parent, 0, child)
                    )
                )
                solver.append_formula(
                    _conjunction_implies_to_clauses(
                        (
                            self._vars.var('p', child, parent),
                            self._vars.var('p', child + 1, parent),
                        ),
                        self._vars.var('y', parent, 1, child + 1)
                    )
                )

    def _define_m_variables(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(old_size, size):
            for parent in range(child):
                for l_num in range(self._alphabet_size):
                    solver.append_formula(
                        _iff_conjunction_to_clauses(
                            self._vars.var('m', parent, l_num, child),
                            tuple(-self._vars.var('y', parent, l_less, child) for l_less in range(l_num)) +
                            (self._vars.var('y', parent, l_num, child),)
                        )
                    )

    def _order_children_using_m(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(max(old_size - 1, 0), size - 1):
            for parent in range(child):
                for l_num in range(self._alphabet_size):
                    for l_less in range(l_num):
                        solver.append_formula(
                            _conjunction_implies_to_clauses(
                                (
                                    self._vars.var('p', child, parent),
                                    self._vars.var('p', child + 1, parent),
                                    self._vars.var('m', parent, l_num, child),
                                ),
                                -self._vars.var('m', parent, l_less, child + 1)
                            )
                        )


class TightBFSBasedSymBreakingClausesGenerator(BFSBasedSymBreakingClausesGenerator):

    def generate(self, solver: Solver, size: int) -> None:
        self._define_t_variables(solver, size)
        self._define_nt_variables(solver, size)
        self._define_p_variables_using_nt(solver, size)
        self._state_has_at_least_one_parent(solver, size)
        self._state_has_at_most_one_parent(solver, size)
        self._define_eq_variables(solver, size)
        self._order_parents_using_ng_variables(solver, size)
        self._order_children(solver, size)

    def generate_with_new_counterexamples(self, solver: Solver, size: int, new_from: int,
                                          changed_statuses: List[int]) -> None:
        pass

    def generate_with_new_size(self, solver: Solver, old_size: int, new_size: int) -> None:
        self._define_t_variables(solver, new_size, old_size)
        self._define_nt_variables(solver, new_size, old_size)
        self._define_p_variables_using_nt(solver, new_size, old_size)
        self._state_has_at_least_one_parent(solver, new_size, old_size)
        self._state_has_at_most_one_parent(solver, new_size, old_size)
        self._define_eq_variables(solver, new_size, old_size)
        self._order_parents_using_ng_variables(solver, new_size, old_size)
        self._order_children(solver, new_size, old_size)

    def _define_nt_variables(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(max(old_size, 2), size):
            solver.append_formula(
                _iff_to_clauses(self._vars.var('nt', 0, child), -self._vars.var('t', 0, child))
            )
            for parent in range(1, child):
                solver.append_formula(
                    _iff_conjunction_to_clauses(
                        self._vars.var('nt', parent, child),
                        (self._vars.var('nt', parent - 1, child), -self._vars.var('t', parent, child))
                    )
                )

    def _define_p_variables_using_nt(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(max(1, old_size), size):
            solver.append_formula(
                _iff_to_clauses(self._vars.var('p', child, 0), self._vars.var('t', 0, child))
            )
            for parent in range(1, child):
                solver.append_formula(
                    _iff_conjunction_to_clauses(
                        self._vars.var('p', child, parent),
                        (self._vars.var('t', parent, child), self._vars.var('nt', parent - 1, child))
                    )
                )

    def _state_has_at_most_one_parent(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(old_size, size):
            for parent in range(child):
                for other_parent in range(parent):
                    solver.add_clause(
                        (-self._vars.var('p', child, parent), -self._vars.var('p', child, other_parent)))

    def _define_eq_variables(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(max(1, old_size - 1), size - 1):
            for parent in range(child):
                solver.add_clause(
                    (
                        self._vars.var('eq', child, parent),
                        self._vars.var('p', child, parent),
                        self._vars.var('p', child + 1, parent)
                    )
                )
                solver.add_clause(
                    (
                        self._vars.var('eq', child, parent),
                        -self._vars.var('p', child, parent),
                        -self._vars.var('p', child + 1, parent)
                    )
                )
                solver.add_clause(
                    (
                        -self._vars.var('eq', child, parent),
                        -self._vars.var('p', child, parent),
                        self._vars.var('p', child + 1, parent)
                    )
                )
                solver.add_clause(
                    (
                        -self._vars.var('eq', child, parent),
                        self._vars.var('p', child, parent),
                        -self._vars.var('p', child + 1, parent)
                    )
                )

    def _order_parents_using_ng_variables(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(max(1, old_size - 1), size - 1):
            solver.add_clause((self._vars.var('ng', child, child),))
            solver.add_clause((self._vars.var('ng', child, 0),))
            for parent in range(child):
                solver.add_clause(
                    (
                        -self._vars.var('ng', child, parent),
                        self._vars.var('ng', child, parent + 1),
                        self._vars.var('p', child, parent)
                    )
                )
                solver.add_clause(
                    (
                        -self._vars.var('ng', child, parent),
                        self._vars.var('eq', child, parent),
                        self._vars.var('p', child, parent)
                    )
                )
                solver.add_clause(
                    (
                        -self._vars.var('ng', child, parent),
                        self._vars.var('ng', child, parent + 1),
                        -self._vars.var('p', child + 1, parent)
                    )
                )
                solver.add_clause(
                    (
                        -self._vars.var('ng', child, parent),
                        self._vars.var('eq', child, parent),
                        -self._vars.var('p', child + 1, parent)
                    )
                )
                solver.add_clause(
                    (
                        self._vars.var('ng', child, parent),
                        -self._vars.var('ng', child, parent + 1),
                        -self._vars.var('eq', child, parent)
                    )
                )
                solver.add_clause(
                    (
                        self._vars.var('ng', child, parent),
                        -self._vars.var('p', child, parent),
                        self._vars.var('p', child + 1, parent)
                    )
                )

    def _order_children(self, solver: Solver, size: int, old_size: int = 0) -> None:
        if self._alphabet_size == 2:
            self._order_children_with_binary_alphabet(solver, size, old_size)
        elif self._alphabet_size > 2:
            self._define_ny_variables(solver, size, old_size)
            self._define_m_variables_with_ny(solver, size, old_size)
            self._define_zm_variables(solver, size, old_size)
            self._order_children_using_zm(solver, size, old_size)

    def _define_ny_variables(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(old_size, size):
            for parent in range(child):
                solver.append_formula(
                    _iff_to_clauses(
                        self._vars.var('ny', parent, 0, child),
                        -self._vars.var('y', parent, 0, child),
                    )
                )
                for l_num in range(1, self._alphabet_size):
                    solver.append_formula(
                        _iff_conjunction_to_clauses(
                            self._vars.var('ny', parent, l_num, child),
                            (
                                -self._vars.var('y', parent, l_num, child),
                                self._vars.var('ny', parent, l_num - 1, child),
                            )
                        )
                    )

    def _define_m_variables_with_ny(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(old_size, size):
            for parent in range(child):
                solver.append_formula(
                    _iff_to_clauses(
                        self._vars.var('m', parent, 0, child),
                        self._vars.var('y', parent, 0, child),
                    )
                )
                for l_num in range(1, self._alphabet_size):
                    solver.append_formula(
                        _iff_conjunction_to_clauses(
                            self._vars.var('m', parent, l_num, child),
                            (
                                self._vars.var('y', parent, l_num, child),
                                self._vars.var('ny', parent, l_num - 1, child),
                            )
                        )
                    )

    def _define_zm_variables(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(old_size, size):
            for parent in range(child):
                solver.append_formula(
                    _iff_to_clauses(
                        self._vars.var('zm', parent, 0, child),
                        -self._vars.var('m', parent, 0, child),
                    )
                )
                for l_num in range(1, self._alphabet_size):
                    solver.append_formula(
                        _iff_conjunction_to_clauses(
                            self._vars.var('zm', parent, l_num, child),
                            (
                                self._vars.var('zm', parent, l_num - 1, child),
                                -self._vars.var('m', parent, l_num, child),
                            )
                        )
                    )

    def _order_children_using_zm(self, solver: Solver, size: int, old_size: int = 0) -> None:
        for child in range(max(0, old_size - 1), size - 1):
            for parent in range(child):
                for l_num in range(1, self._alphabet_size):
                    solver.append_formula(
                        _conjunction_implies_to_clauses(
                            (
                                self._vars.var('p', child, parent),
                                self._vars.var('p', child + 1, parent),
                                self._vars.var('m', parent, l_num, child)
                            ),
                            self._vars.var('zm', parent, l_num - 1, child + 1)
                        )
                    )


class NoSymBreakingClausesGenerator(BaseClausesGenerator):
    def generate(self, solver: Solver, size: int) -> None:
        pass

    def generate_with_new_counterexamples(self, solver: Solver, size: int, new_from: int,
                                          changed_statuses: List[int]) -> None:
        pass

    def generate_with_new_size(self, solver: Solver, old_size: int, new_size: int) -> None:
        pass
