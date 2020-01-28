from typing import Optional

from pysat.solvers import Solver

from .strategies import CegarSynthesizer, ClassicSynthesizer
from ..examples import BaseExamplesProvider
from ..logging import *
from ..structures import APTA, DFA


class LSUS:

    def __init__(self,
                 lower_bound: int,
                 upper_bound: int,
                 apta: APTA,
                 solver_name: str,
                 sb_strategy: str,
                 cegar_mode: str,
                 examples_provider: BaseExamplesProvider) -> None:
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._apta = apta
        self._solver_name = solver_name
        self._sb_strategy = sb_strategy
        self._cegar_mode = cegar_mode
        self._examples_provider = examples_provider

    def search(self) -> Optional[DFA]:
        for size in range(self._lower_bound, self._upper_bound + 1):
            log_br()
            log_info('Trying to build a DFA with {0} states.'.format(size))
            if self._cegar_mode == 'none':
                synthesizer = ClassicSynthesizer(Solver(self._solver_name),
                                                 self._apta,
                                                 size,
                                                 self._sb_strategy)
            else:
                synthesizer = CegarSynthesizer(Solver(self._solver_name),
                                               self._apta,
                                               size,
                                               self._sb_strategy,
                                               self._examples_provider)
            dfa = synthesizer.synthesize_dfa()
            if not dfa:
                log_info('Not found a DFA with {0} states.'.format(size))
            else:
                log_success('The DFA with {0} states is found!'.format(size))
                return dfa
        return None
