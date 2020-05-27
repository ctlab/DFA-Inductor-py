from typing import List

from pysat.solvers import Solver

from ..variables import VarPool
from .reductions import ClauseGenerator
from ..examples import BaseExamplesProvider
from ..logging import *
from ..statistics import STATISTICS
from ..structures import APTA, DFA, InconsistencyGraph


class LSUS:
    _solver: Solver

    def __init__(self,
                 apta: APTA,
                 ig: InconsistencyGraph,
                 solver_name: str,
                 sb_strategy: str,
                 cegar_mode: str,
                 examples_provider: BaseExamplesProvider,
                 assumptions_mode: str) -> None:
        self._apta = apta
        self._ig = ig
        self._solver_name = solver_name
        self._sb_strategy = sb_strategy
        self._cegar_mode = cegar_mode
        self._examples_provider = examples_provider
        self._assumptions_mode = assumptions_mode

        self._var_pool: VarPool = VarPool()
        self._clause_generator = ClauseGenerator(self._apta,
                                                 self._ig,
                                                 self._var_pool,
                                                 self._assumptions_mode,
                                                 self._sb_strategy)

    def _try_to_synthesize_dfa(self, size: int, assumptions: List[int]) -> Optional[DFA]:
        log_info('Vars in CNF: {0}'.format(self._solver.nof_vars()))
        log_info('Clauses in CNF: {0}'.format(self._solver.nof_clauses()))

        STATISTICS.start_solving_timer()
        is_sat = self._solver.solve(assumptions=assumptions)
        STATISTICS.stop_solving_timer()

        if is_sat:
            assignment = self._solver.get_model()
            dfa = DFA()
            for i in range(size):
                dfa.add_state(
                    DFA.State.StateStatus.from_bool(assignment[self._var_pool.var('z', i) - 1] > 0)
                )
            for i in range(size):
                for label in range(self._apta.alphabet_size):
                    for j in range(size):
                        if assignment[self._var_pool.var('y', i, label, j) - 1] > 0:
                            dfa.add_transition(i, self._apta.alphabet[label], j)
            return dfa
        else:
            return None

    def search(self, lower_bound: int, upper_bound: int) -> Optional[DFA]:
        self._solver = Solver(self._solver_name)
        log_info('Solver has been started.')
        for size in range(lower_bound, upper_bound + 1):
            if self._assumptions_mode == 'none' and size > lower_bound:
                self._solver = Solver(self._solver_name)
                log_info('Solver has been restarted.')
            log_br()
            log_info('Trying to build a DFA with {0} states.'.format(size))

            STATISTICS.start_formula_timer()
            if self._assumptions_mode != 'none' and size > lower_bound:
                self._clause_generator.generate_with_new_size(self._solver, size - 1, size)
            else:
                self._clause_generator.generate(self._solver, size)
            STATISTICS.stop_formula_timer()
            assumptions = self._clause_generator.build_assumptions(size, self._solver)
            while True:
                dfa = self._try_to_synthesize_dfa(size, assumptions)
                if dfa:
                    counter_examples = self._examples_provider.get_counter_examples(dfa)
                    if counter_examples:
                        log_info('An inconsistent DFA with {0} states is found.'.format(size))
                        log_info('Added {0} counterexamples.'.format(len(counter_examples)))

                        STATISTICS.start_apta_building_timer()
                        (new_nodes_from, changed_statuses) = self._apta.add_examples(counter_examples)
                        STATISTICS.stop_apta_building_timer()

                        STATISTICS.start_ig_building_timer()
                        self._ig.update(new_nodes_from)
                        STATISTICS.stop_ig_building_timer()

                        STATISTICS.start_formula_timer()
                        self._clause_generator.generate_with_new_counterexamples(self._solver, size,
                                                                                 new_nodes_from,
                                                                                 changed_statuses)
                        STATISTICS.stop_formula_timer()
                        continue
                break
            if not dfa:
                log_info('Not found a DFA with {0} states.'.format(size))
            else:
                log_success('The DFA with {0} states is found!'.format(size))
                return dfa
        return None
