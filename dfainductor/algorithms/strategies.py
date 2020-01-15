from abc import ABC, abstractmethod

from pysat.formula import IDPool
from pysat.solvers import Solver

from .reductions import *
from ..structures import DFA


class BaseStrategy(ABC):

    def __init__(self, solver, apta, size, sb_strategy):
        self._solver: Solver = solver
        self._apta = apta
        self._size = size
        self._sb_strategy = sb_strategy
        self._vpool = IDPool()

    @abstractmethod
    def synthesize_dfa(self):
        pass

    def _try_to_synthesize_dfa(self, formula):
        self._solver.append_formula(formula.clauses)
        is_sat = self._solver.solve()
        if is_sat:
            assignment = self._solver.get_model()
            dfa = DFA()
            for i in range(self._size):
                dfa.add_state(DFA.State.StateStatus.from_bool(assignment[self._vpool.id('z_{0}'.format(i)) - 1] > 0))
            for i in range(self._size):
                for label in self._apta.alphabet:
                    for j in range(self._size):
                        if assignment[self._vpool.id('y_{0}_{1}_{2}'.format(i, label, j)) - 1] > 0:
                            dfa.add_transition(i, label, j)
            return dfa
        else:
            return None

    def _symmetry_breaking_predicates(self):
        if self._sb_strategy == 'BFS':
            return BFSBasedSymBreakingClausesGenerator(self._apta, self._size, self._vpool).generate()
        elif self._sb_strategy == 'TIGHTBFS':
            return TightBFSBasedSymBreakingClausesGenerator(self._apta, self._size, self._vpool).generate()


class ClassicSynthesizer(BaseStrategy):
    def synthesize_dfa(self):
        formula = MinDFAToSATClausesGenerator(self._apta, self._size, self._vpool).generate()
        formula.extend(self._symmetry_breaking_predicates().clauses)
        return self._try_to_synthesize_dfa(formula)
