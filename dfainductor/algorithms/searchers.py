from typing import List

from pysat.formula import IDPool
from pysat.solvers import Solver

from .reductions import FORMULA
from . import reductions
from ..examples import BaseExamplesProvider
from ..logging import *
from ..statistics import STATISTICS
from ..structures import APTA, DFA, InconsistencyGraph


class LSUS:

    def __init__(self,
                 apta: APTA,
                 ig: InconsistencyGraph,
                 solver_name: str,
                 sb_strategy: str,
                 cegar_mode: str,
                 examples_provider: BaseExamplesProvider,
                 with_assumptions: bool) -> None:
        self._apta = apta
        self._ig = ig
        self._solver_name = solver_name
        self._solver = None
        self._sb_strategy = sb_strategy
        self._cegar_mode = cegar_mode
        self._examples_provider = examples_provider
        self._with_assumptions = with_assumptions
        self._vpool = IDPool()
        self._clause_generator = reductions.ClauseGenerator(
            self._apta,
            self._ig,
            self._vpool,
            self._with_assumptions,
            self._sb_strategy
        )

    def _try_to_synthesize_dfa(self, formula: FORMULA, size: int) -> Optional[DFA]:
        STATISTICS.start_feeding_timer()
        self._solver.append_formula(formula)
        STATISTICS.stop_feeding_timer()

        assumptions = self._build_assumptions(size) if self._with_assumptions else []
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
                    DFA.State.StateStatus.from_bool(assignment[self._vpool.id(f'z_{i}_0_0') - 1] > 0))
            for i in range(size):
                for label in self._apta.alphabet:
                    for j in range(size):
                        if assignment[self._vpool.id(f'y_{i}_{label}_{j}') - 1] > 0:
                            dfa.add_transition(i, label, j)
            return dfa
        else:
            return None

    def _build_assumptions(self, size: int) -> List[int]:
        assumptions = []
        for v in range(self._apta.size):
            assumptions.append(self._vpool.id('alo_x_{0}_{1}'.format(size, v)))

        for i in range(size):
            for l_id in range(self._apta.alphabet_size):
                assumptions.append(self._vpool.id('alo_y_{0}_{1}_{2}'.format(size, i, l_id)))
        return assumptions

    def search(self, lower_bound: int, upper_bound: int) -> Optional[DFA]:
        self._solver = Solver(self._solver_name)
        log_info('Solver has been started.')
        for size in range(lower_bound, upper_bound + 1):
            if not self._with_assumptions and size > lower_bound:
                self._solver = Solver(self._solver_name)
                log_info('Solver has been restarted.')
            log_br()
            log_info('Trying to build a DFA with {0} states.'.format(size))

            STATISTICS.start_formula_timer()
            if self._with_assumptions and size > lower_bound:
                formula = self._clause_generator.generate_with_new_size(size - 1, size)
            else:
                formula = self._clause_generator.generate(size)
            STATISTICS.stop_formula_timer()

            while True:
                dfa = self._try_to_synthesize_dfa(formula, size)
                if dfa:
                    counter_examples = self._examples_provider.get_counter_examples(dfa)
                    if counter_examples:
                        log_info('An inconsistent DFA with {0} states is found.'.format(size))
                        log_info('Added {0} counterexamples.'.format(len(counter_examples)))

                        STATISTICS.start_apta_building_timer()
                        (new_nodes_from, changed_statuses) = self._apta.add_examples(counter_examples)
                        STATISTICS.stop_apta_building_timer()

                        STATISTICS.start_formula_timer()
                        formula = self._clause_generator.generate_with_new_counterexamples(size,
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
