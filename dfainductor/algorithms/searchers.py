from pysat.solvers import Solver

from .strategies import ClassicSynthesizer
from ..logging import *


class LSUS:
    def __init__(self, lower_bound, upper_bound, apta, solver_name, sb_strategy):
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._apta = apta
        self._solver_name = solver_name
        self._sb_strategy = sb_strategy

    def search(self):
        for size in range(self._lower_bound, self._upper_bound + 1):
            log_br()
            log_info('Trying to build a DFA with {0} states.'.format(size))
            synthesizer = ClassicSynthesizer(Solver(self._solver_name), self._apta, size, self._sb_strategy)
            dfa = synthesizer.synthesize_dfa()
            if not dfa:
                log_info('Not found a DFA with {0} states.'.format(size))
            else:
                log_success('The DFA with {0} states is found!'.format(size))
                return dfa
        return None
