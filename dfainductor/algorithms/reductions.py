from abc import ABC, abstractmethod
from itertools import chain
from typing import List, Tuple, Iterator

from pysat.formula import CNF, IDPool

from ..structures import APTA

CLAUSE = Tuple[int, ...]
FORMULA = Iterator[CLAUSE]


def _implication_to_clauses(lhs: int, rhs: int) -> FORMULA:
    """
    generates CNF formula of an expression /lhs => rhs/
    :type lhs: int
    :type rhs: int
    """
    yield (-lhs, rhs)


def _conjunction_implies_to_clauses(lhs: CLAUSE, rhs: int) -> FORMULA:
    """
    generates CNF formula of an expression /lhs_1 and lhs_2 and ... and lhs_n => rhs/
    :type lhs: list(int)
    :type rhs: int
    """
    yield tuple(-lit for lit in lhs) + (rhs,)


def _iff_to_clauses(lhs: int, rhs: int) -> FORMULA:
    """
    generates CNF formula of an expression /lhs <=> rhs/
    :type lhs: int
    :type rhs: int
    """
    yield from _implication_to_clauses(lhs, rhs)
    yield from _implication_to_clauses(rhs, lhs)


def _iff_disjunction_to_clauses(lhs: int, rhs: CLAUSE) -> FORMULA:
    """
    generates CNF formula of an expression /lhs <=> rhs_1 or rhs_2 or ... or rhs_n/
    :type lhs: int
    :type rhs: list(int)
    """
    yield (-lhs,) + rhs
    yield from ((lhs, -lit) for lit in rhs)


def _iff_conjunction_to_clauses(lhs: int, rhs: CLAUSE) -> FORMULA:
    """
    generates CNF formula of an expression /lhs <=> rhs_1 and rhs_2 and ... and rhs_n/
    :type lhs: int
    :type rhs: list(int)
    """
    yield (lhs,) + tuple(-lit for lit in rhs)
    yield from ((-lhs, lit) for lit in rhs)


class BaseClausesGenerator(ABC):

    def __init__(self, apta: APTA, vpool: IDPool, with_assumptions: bool) -> None:
        self._apta = apta
        self._vpool = vpool
        self._with_assumptions = with_assumptions
        self._alphabet = self._apta.alphabet
        self._alphabet_size = len(self._alphabet)

    @abstractmethod
    def generate(self, size: int) -> FORMULA:
        pass

    @abstractmethod
    def generate_with_new_counterexamples(self, size: int, new_from: int, changed_statuses: List[int]) -> FORMULA:
        pass

    @abstractmethod
    def generate_with_new_size(self, old_size: int, new_size: int) -> FORMULA:
        pass

    def _var(self, name: str, ind1, ind2=0, ind3=0) -> int:
        var = f'{name}_{ind1}_{ind2}_{ind3}'
        result = self._vpool.id(var)
        return result

    @staticmethod
    def _empty_formula():
        yield ()


class ClauseGenerator(BaseClausesGenerator):

    def __init__(self, apta: APTA, vpool: IDPool, with_assumptions: bool, sb_strategy: str) -> None:
        super().__init__(apta, vpool, with_assumptions)
        self._mindfa_generator = MinDFAToSATClausesGenerator(apta, vpool, with_assumptions)
        if sb_strategy == 'BFS':
            self._sb_generator = BFSBasedSymBreakingClausesGenerator(apta, vpool, with_assumptions)
        elif sb_strategy == 'TIGHTBFS':
            self._sb_generator = TightBFSBasedSymBreakingClausesGenerator(apta, vpool, with_assumptions)
        else:
            self._sb_generator = NoSymBreakingClausesGenerator(apta, vpool, with_assumptions)

    def generate(self, size: int) -> FORMULA:
        yield from self._mindfa_generator.generate(size)
        yield from self._sb_generator.generate(size)

    def generate_with_new_counterexamples(self, size: int, new_from: int, changed_statuses: List[int]) -> FORMULA:
        yield from self._mindfa_generator.generate_with_new_counterexamples(size, new_from, changed_statuses)
        yield from self._sb_generator.generate_with_new_counterexamples(size, new_from, changed_statuses)

    def generate_with_new_size(self, old_size: int, new_size: int) -> FORMULA:
        yield from self._mindfa_generator.generate_with_new_size(old_size, new_size)
        yield from self._sb_generator.generate_with_new_size(old_size, new_size)


class MinDFAToSATClausesGenerator(BaseClausesGenerator):
    def generate(self, size: int) -> FORMULA:
        yield from self._fix_start_state()
        yield from self._one_node_maps_to_at_least_one_state(size)
        yield from self._one_node_maps_to_at_most_one_state(size)
        yield from self._dfa_is_complete(size)
        yield from self._dfa_is_deterministic(size)
        yield from self._state_status_compatible_with_node_status(size)
        yield from self._mapped_adjacent_nodes_force_transition(size)
        yield from self._mapped_node_and_transition_force_mapping(size)

    def generate_with_new_counterexamples(self, size: int, new_from: int, changed_statuses: List[int]) -> FORMULA:
        yield from self._one_node_maps_to_at_least_one_state(size, new_node_from=new_from)
        yield from self._one_node_maps_to_at_most_one_state(size, new_node_from=new_from)
        yield from self._state_status_compatible_with_node_status(size,
                                                                  new_node_from=new_from,
                                                                  changed_statuses=changed_statuses)
        yield from self._mapped_adjacent_nodes_force_transition(size, new_node_from=new_from)
        yield from self._mapped_node_and_transition_force_mapping(size, new_node_from=new_from)

    def generate_with_new_size(self, old_size: int, new_size: int) -> FORMULA:
        yield from self._one_node_maps_to_at_least_one_state(new_size, old_size=old_size)
        yield from self._one_node_maps_to_at_most_one_state(new_size, old_size=old_size)
        yield from self._dfa_is_complete(new_size, old_size=old_size)
        yield from self._dfa_is_deterministic(new_size, old_size=old_size)
        yield from self._state_status_compatible_with_node_status(new_size, old_size=old_size)
        yield from self._mapped_adjacent_nodes_force_transition(new_size, old_size=old_size)
        yield from self._mapped_node_and_transition_force_mapping(new_size, old_size=old_size)

    def _fix_start_state(self) -> FORMULA:
        yield (self._var('x', 0, 0),)

    def _one_node_maps_to_at_least_one_state(self,
                                             size: int,
                                             new_node_from: int = 0,
                                             old_size: int = 0) -> FORMULA:
        if not self._with_assumptions:
            yield from self._one_node_maps_to_at_least_one_state_classic(size, new_node_from=new_node_from)
        else:
            yield from self._one_node_maps_to_at_least_one_state_with_assumptions(size,
                                                                                  new_node_from=new_node_from,
                                                                                  old_size=old_size)

    def _one_node_maps_to_at_least_one_state_classic(self, size: int, new_node_from: int = 0) -> FORMULA:
        yield from (
            tuple(self._var('x', i, j) for j in range(size))
            for i in range(new_node_from, self._apta.size())
        )

    def _one_node_maps_to_at_least_one_state_with_assumptions(self,
                                                              size: int,
                                                              new_node_from: int = 0,
                                                              old_size: int = 0) -> FORMULA:
        if old_size == 0:
            yield from (
                tuple(self._var('x', i, j) for j in range(old_size, size)) +
                (-self._var('alo_x', size, i),)
                for i in range(new_node_from, self._apta.size())
            )
        else:
            yield from (
                tuple(self._var('x', i, j) for j in range(old_size, size)) +
                (-self._var('alo_x', size, i), self._var('alo_x', old_size, i))
                for i in range(new_node_from, self._apta.size())
            )

    def _one_node_maps_to_at_most_one_state(self, size: int, new_node_from: int = 0, old_size: int = 0) -> FORMULA:
        yield from (
            (-self._var('x', v, i), -self._var('x', v, j))
            for v in range(new_node_from, self._apta.size())
            for i in range(old_size, size)
            for j in range(0, i)
        )

    def _dfa_is_complete(self, size: int, old_size: int = 0):
        if not self._with_assumptions:
            yield from self._dfa_is_complete_classic(size)
        else:
            yield from self._dfa_is_complete_with_assumptions(size, old_size=old_size)

    def _dfa_is_complete_classic(self, size: int) -> FORMULA:
        yield from (
            tuple(self._var('y', i, l_id, j) for j in range(size))
            for i in range(size)
            for l_id in range(self._alphabet_size)
        )

    def _dfa_is_complete_with_assumptions(self, size: int, old_size: int = 0) -> FORMULA:
        if old_size == 0:
            yield from (
                tuple(self._var('y', i, l_id, j) for j in range(old_size, size)) +
                (-self._var('alo_y', size, i, l_id),)
                for l_id in range(self._alphabet_size)
                for i in range(old_size)
            )
        else:
            yield from (
                tuple(self._var('y', i, l_id, j) for j in range(old_size, size)) +
                (-self._var('alo_y', size, i, l_id), self._var('alo_y', old_size, i, l_id))
                for l_id in range(self._alphabet_size)
                for i in range(old_size)
            )
        yield from (
            tuple(self._var('y', i, l_id, j) for j in range(size)) +
            (-self._var('alo_y', size, i, l_id),)
            for l_id in range(self._alphabet_size)
            for i in range(old_size, size)
        )

    def _dfa_is_deterministic(self, size: int, old_size: int = 0) -> FORMULA:
        yield from (
            (-self._var('y', i, l_id, j), -self._var('y', i, l_id, k))
            for l_id in range(self._alphabet_size)
            for i in range(old_size)
            for j in range(old_size, size)
            for k in range(j)
        )
        yield from (
            (-self._var('y', i, l_id, j), -self._var('y', i, l_id, k))
            for l_id in range(self._alphabet_size)
            for i in range(old_size, size)
            for j in range(size)
            for k in range(j)
        )

    def _state_status_compatible_with_node_status(self,
                                                  size: int,
                                                  new_node_from: int = 0,
                                                  old_size: int = 0,
                                                  changed_statuses=None) -> FORMULA:
        if changed_statuses is None:
            changed_statuses = []
        for i in chain(range(new_node_from, self._apta.size()), changed_statuses):
            if self._apta.get_node(i).is_accepting():
                for j in range(old_size, size):
                    yield from _implication_to_clauses(self._var('x', i, j), self._var('z', j))
            elif self._apta.get_node(i).is_rejecting():
                for j in range(old_size, size):
                    yield from _implication_to_clauses(self._var('x', i, j), -self._var('z', j))

    def _mapped_adjacent_nodes_force_transition(self, size: int, new_node_from: int = 0, old_size: int = 0) -> FORMULA:
        for parent in self._apta.nodes:
            for label, child in parent.children.items():
                if parent.id_ >= new_node_from or child.id_ >= new_node_from:
                    for from_ in range(old_size, size):
                        for to in range(old_size, size):
                            yield from _conjunction_implies_to_clauses(
                                (self._var('x', parent.id_, from_), self._var('x', child.id_, to),),
                                self._var('y', from_, label, to)
                            )
                    if old_size > 0:
                        for from_ in range(old_size):
                            for to in range(old_size, size):
                                yield from _conjunction_implies_to_clauses(
                                    (self._var('x', parent.id_, from_), self._var('x', child.id_, to),),
                                    self._var('y', from_, label, to)
                                )

                        for from_ in range(old_size, size):
                            for to in range(old_size):
                                yield from _conjunction_implies_to_clauses(
                                    (self._var('x', parent.id_, from_), self._var('x', child.id_, to),),
                                    self._var('y', from_, label, to)
                                )

    def _mapped_node_and_transition_force_mapping(self, size: int, new_node_from: int = 0,
                                                  old_size: int = 0) -> FORMULA:
        for parent in self._apta.nodes:
            for label, child in parent.children.items():
                if parent.id_ >= new_node_from or child.id_ >= new_node_from:
                    for from_ in range(old_size, size):
                        for to in range(old_size, size):
                            yield from _conjunction_implies_to_clauses(
                                (self._var('x', parent.id_, from_), self._var('y', from_, label, to),),
                                self._var('x', child.id_, to)
                            )
                    if old_size > 0:
                        for from_ in range(old_size):
                            for to in range(old_size, size):
                                yield from _conjunction_implies_to_clauses(
                                    (self._var('x', parent.id_, from_), self._var('y', from_, label, to),),
                                    self._var('x', child.id_, to)
                                )
                        for from_ in range(old_size, size):
                            for to in range(old_size):
                                yield from _conjunction_implies_to_clauses(
                                    (self._var('x', parent.id_, from_), self._var('y', from_, label, to),),
                                    self._var('x', child.id_, to)
                                )


class BFSBasedSymBreakingClausesGenerator(BaseClausesGenerator):
    def generate(self, size: int) -> FORMULA:
        yield from self._define_t_variables(size)
        yield from self._define_p_variables(size)
        yield from self._state_has_at_least_one_parent(size)
        yield from self._preserve_parent_order_on_children(size)
        yield from self._order_children(size)

    def generate_with_new_counterexamples(self, size: int, new_from: int, changed_statuses: List[int]) -> FORMULA:
        yield from super()._empty_formula()

    def generate_with_new_size(self, old_size: int, new_size: int) -> FORMULA:
        yield from self._define_t_variables(new_size, old_size=old_size)
        yield from self._define_p_variables(new_size, old_size=old_size)
        yield from self._state_has_at_least_one_parent(new_size, old_size=old_size)
        yield from self._preserve_parent_order_on_children(new_size, old_size=old_size)
        yield from self._order_children(new_size, old_size=old_size)

    def _define_t_variables(self, size: int, old_size: int = 0) -> CNF:
        for to in range(old_size, size):
            for from_ in range(to):
                yield from _iff_disjunction_to_clauses(
                    self._var('t', from_, to),
                    tuple(self._var('y', from_, l_id, to) for l_id in range(self._alphabet_size))
                )

    def _define_p_variables(self, size: int, old_size: int = 0) -> FORMULA:
        for child in range(old_size, size):
            for parent in range(child):
                yield from _iff_conjunction_to_clauses(
                    self._var('p', child, parent),
                    tuple(-self._var('t', prev, child) for prev in range(parent)) + (self._var('t', parent, child),)
                )

    def _state_has_at_least_one_parent(self, size: int, old_size: int = 0) -> FORMULA:
        yield from (
            tuple(self._var('p', child, parent) for parent in range(child))
            for child in range(max(1, old_size), size)
        )

    def _preserve_parent_order_on_children(self, size: int, old_size: int = 0) -> FORMULA:
        for child in range(max(2, old_size - 1), size - 1):
            for parent in range(1, child):
                for pre_parent in range(parent):
                    yield from _implication_to_clauses(
                        self._var('p', child, parent), -self._var('p', child + 1, pre_parent)
                    )

    def _order_children(self, size: int, old_size: int = 0) -> FORMULA:
        if self._alphabet_size == 2:
            yield from self._order_children_with_binary_alphabet(size, old_size)
        elif self._alphabet_size > 2:
            yield from self._define_m_variables(size, old_size)
            yield from self._order_children_using_m(size, old_size)

    def _order_children_with_binary_alphabet(self, size: int, old_size: int = 0) -> FORMULA:
        for child in range(max(0, old_size - 1), size - 1):
            for parent in range(child):
                yield from _conjunction_implies_to_clauses(
                    (self._var('p', child, parent), self._var('p', child + 1, parent)),
                    self._var('y', parent, 0, child)
                )
                yield from _conjunction_implies_to_clauses(
                    (self._var('p', child, parent), self._var('p', child + 1, parent)),
                    self._var('y', parent, 1, child + 1)
                )

    def _define_m_variables(self, size: int, old_size: int = 0) -> FORMULA:
        for child in range(old_size, size):
            for parent in range(child):
                for l_num in range(self._alphabet_size):
                    yield from _iff_conjunction_to_clauses(
                        self._var('m', parent, l_num, child),
                        tuple(-self._var('y', parent, l_less, child) for l_less in range(l_num)) +
                        (self._var('y', parent, l_num, child),)
                    )

    def _order_children_using_m(self, size: int, old_size: int = 0) -> FORMULA:
        for child in range(max(old_size - 1, 0), size - 1):
            for parent in range(child):
                for l_num in range(self._alphabet_size):
                    for l_less in range(l_num):
                        yield from _conjunction_implies_to_clauses(
                            (
                                self._var('p', child, parent),
                                self._var('p', child + 1, parent),
                                self._var('m', parent, l_num, child),
                            ),
                            -self._var('m', parent, l_less, child + 1)
                        )


# TODO: fix. It doesn't support assumptions
class TightBFSBasedSymBreakingClausesGenerator(BFSBasedSymBreakingClausesGenerator):

    def generate(self, size: int) -> FORMULA:
        yield from self._define_t_variables(size)
        yield from self._define_nt_variables(size)
        yield from self._define_p_variables_using_nt(size)
        yield from self._state_has_at_least_one_parent(size)
        yield from self._state_has_at_most_one_parent(size)
        yield from self._define_eq_variables(size)
        yield from self._order_parents_using_ng_variables(size)
        yield from self._order_children(size)

    def generate_with_new_counterexamples(self, size: int, new_from: int, changed_statuses: List[int]) -> FORMULA:
        yield from super()._empty_formula()

    def generate_with_new_size(self, old_size: int, new_size: int) -> FORMULA:
        return super().generate_with_new_size(old_size, new_size)

    def _define_nt_variables(self, size: int) -> FORMULA:
        for child in range(2, size):
            yield from _iff_to_clauses(self._var('nt', 0, child), -self._var('t', 0, child))
            for parent in range(1, child):
                yield from _iff_conjunction_to_clauses(
                    self._var('nt', parent, child),
                    (self._var('nt', parent - 1, child), -self._var('t', parent, child))
                )

    def _define_p_variables_using_nt(self, size: int) -> FORMULA:
        for child in range(1, size):
            yield from _iff_to_clauses(self._var('p', child, 0), self._var('t', 0, child))
            for parent in range(1, child):
                yield from _iff_conjunction_to_clauses(
                    self._var('p', child, parent),
                    (self._var('t', parent, child), self._var('nt', parent - 1, child))
                )

    def _state_has_at_most_one_parent(self, size: int) -> FORMULA:
        yield from (
            (-self._var('p', child, parent), -self._var('p', child, other_parent))
            for child in range(1, size)
            for parent in range(child)
            for other_parent in range(parent)
        )

    def _define_eq_variables(self, size: int) -> FORMULA:
        for child in range(1, size - 1):
            for parent in range(child):
                yield (
                    self._var('eq', child, parent),
                    self._var('p', child, parent),
                    self._var('p', child + 1, parent)
                )
                yield (
                    self._var('eq', child, parent),
                    -self._var('p', child, parent),
                    -self._var('p', child + 1, parent)
                )
                yield (
                    -self._var('eq', child, parent),
                    -self._var('p', child, parent),
                    self._var('p', child + 1, parent)
                )
                yield (
                    -self._var('eq', child, parent),
                    self._var('p', child, parent),
                    -self._var('p', child + 1, parent)
                )

    def _order_parents_using_ng_variables(self, size: int) -> FORMULA:
        for child in range(1, size - 1):
            yield (self._var('ng', child, child),)
            yield (self._var('ng', child, 0),)
            for parent in range(child):
                yield (
                    -self._var('ng', child, parent),
                    self._var('ng', child, parent + 1),
                    self._var('p', child, parent)
                )
                yield (
                    -self._var('ng', child, parent),
                    self._var('eq', child, parent),
                    self._var('p', child, parent)
                )
                yield (
                    -self._var('ng', child, parent),
                    self._var('ng', child, parent + 1),
                    -self._var('p', child + 1, parent)
                )
                yield (
                    -self._var('ng', child, parent),
                    self._var('eq', child, parent),
                    -self._var('p', child + 1, parent)
                )
                yield (
                    self._var('ng', child, parent),
                    -self._var('ng', child, parent + 1),
                    -self._var('eq', child, parent)
                )
                yield (
                    self._var('ng', child, parent),
                    -self._var('p', child, parent),
                    self._var('p', child + 1, parent)
                )

    def _order_children(self, size: int, old_size: int = 0) -> FORMULA:
        if self._alphabet_size == 2:
            yield from self._order_children_with_binary_alphabet(size)
        elif self._alphabet_size > 2:
            yield from self._define_ny_variables(size)
            yield from self._define_m_variables_with_ny(size)
            yield from self._define_zm_variables(size)
            yield from self._order_children_using_zm(size)

    def _define_ny_variables(self, size: int) -> FORMULA:
        for child in range(size):
            for parent in range(child):
                yield from _iff_to_clauses(
                    self._var('ny', parent, 0, child),
                    -self._var('y', parent, 0, child),
                )
                for l_num in range(1, self._alphabet_size):
                    yield from _iff_conjunction_to_clauses(
                        self._var('ny', parent, l_num, child),
                        (-self._var('y', parent, l_num, child), self._var('ny', parent, l_num - 1, child))
                    )

    def _define_m_variables_with_ny(self, size: int) -> FORMULA:
        for child in range(size):
            for parent in range(child):
                yield from _iff_to_clauses(
                    self._var('m', parent, 0, child),
                    self._var('y', parent, 0, child),
                )
                for l_num in range(1, self._alphabet_size):
                    yield from _iff_conjunction_to_clauses(
                        self._var('m', parent, l_num, child),
                        (self._var('y', parent, l_num, child), self._var('ny', parent, l_num - 1, child))
                    )

    def _define_zm_variables(self, size: int) -> FORMULA:
        for child in range(size):
            for parent in range(child):
                yield from _iff_to_clauses(
                    self._var('zm', parent, 0, child),
                    -self._var('m', parent, 0, child),
                )
                for l_num in range(1, self._alphabet_size):
                    yield from _iff_conjunction_to_clauses(
                        self._var('zm', parent, l_num, child),
                        (self._var('zm', parent, l_num - 1, child), -self._var('m', parent, l_num, child))
                    )

    def _order_children_using_zm(self, size: int) -> FORMULA:
        for child in range(size - 1):
            for parent in range(child):
                for l_num in range(1, self._alphabet_size):
                    yield from _conjunction_implies_to_clauses(
                        (
                            self._var('p', child, parent),
                            self._var('p', child + 1, parent),
                            self._var('m', parent, l_num, child)
                        ),
                        self._var('zm', parent, l_num - 1, child + 1)
                    )


class NoSymBreakingClausesGenerator(BaseClausesGenerator):
    def generate(self, size: int) -> FORMULA:
        return super()._empty_formula()

    def generate_with_new_counterexamples(self, size: int, new_from: int, changed_statuses: List[int]) -> FORMULA:
        return super()._empty_formula()

    def generate_with_new_size(self, old_size: int, new_size: int) -> FORMULA:
        return super()._empty_formula()


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
