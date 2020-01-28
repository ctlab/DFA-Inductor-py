from typing import Optional

from pysat.solvers import Solver

from .reductions import *
from ..examples import BaseExamplesProvider, NonCegarExamplesProvider
from ..logging import *
from ..structures import DFA, APTA


class BaseStrategy(ABC):

    def __init__(self, solver: Solver, apta: APTA, size: int, sb_strategy: str) -> None:
        self._solver: Solver = solver
        self._apta = apta
        self._size = size
        self._sb_strategy = sb_strategy
        self._vpool = IDPool()

    @abstractmethod
    def synthesize_dfa(self) -> Optional[DFA]:
        pass

    def _try_to_synthesize_dfa(self, formula: CNF) -> Optional[DFA]:
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


class ClassicSynthesizer(BaseStrategy):
    def synthesize_dfa(self) -> Optional[DFA]:
        formula = MinDFAToSATClausesGenerator(self._apta, self._size, self._vpool).generate()
        formula.extend(get_symmetry_breaking_predicates_generator(self._sb_strategy, self._apta, self._size,
                                                                  self._vpool).generate())
        return self._try_to_synthesize_dfa(formula)


class CegarSynthesizer(BaseStrategy):

    def __init__(self,
                 solver: Solver,
                 apta: APTA,
                 size: int,
                 sb_strategy: str,
                 examples_provider: BaseExamplesProvider) -> None:
        super().__init__(solver, apta, size, sb_strategy)
        self._examples_provider = examples_provider

    def synthesize_dfa(self):
        min_dfa_generator = MinDFAToSATClausesGenerator(self._apta, self._size, self._vpool)
        symmetry_breaking_generator = get_symmetry_breaking_predicates_generator(self._sb_strategy, self._apta,
                                                                                 self._size, self._vpool)
        formula = min_dfa_generator.generate()
        formula.extend(symmetry_breaking_generator.generate())
        while True:
            dfa = self._try_to_synthesize_dfa(formula)
            if dfa:
                counter_examples = self._examples_provider.get_counter_examples(dfa)
                if counter_examples:
                    log_info('An inconsistent DFA with {0} states is found.'.format(self._size))
                    log_info('Added {0} counterexamples.'.format(len(counter_examples)))
                    log_br()
                    new_nodes_from = self._apta.add_examples(counter_examples)
                    min_dfa_generator.generate_with_new_counterexamples(new_nodes_from)
                    continue
            return dfa


def get_synthesizer(solver: Solver,
                    apta: APTA,
                    size: int,
                    sb_strategy: str,
                    examples_provider: BaseExamplesProvider) -> BaseStrategy:
    if isinstance(examples_provider, NonCegarExamplesProvider):
        return ClassicSynthesizer(solver, apta, size, sb_strategy)
    else:
        return CegarSynthesizer(solver, apta, size, sb_strategy, examples_provider)
