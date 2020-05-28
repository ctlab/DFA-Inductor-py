import sys

import click

from . import examples
from .__about__ import __version__
from .algorithms.searchers import LSUS
from .logging import *
from .statistics import STATISTICS
from .structures import APTA, InconsistencyGraph


@click.command(context_settings=dict(
    max_content_width=999,
    help_option_names=['-h', '--help']
))
@click.option('-i', '--input', 'input_', metavar='<PATH>', required=True, type=click.Path(exists=True),
              help='a DFA learning input file in Abbadingo format')
@click.option('-l', '--lower-bound', metavar='<INT>', type=int, default=1, show_default=True,
              help='lower bound of the DFA size')
@click.option('-u', '--upper-bound', metavar='<INT>', type=int, default=100, show_default=True,
              help='upper bound of the DFA size')
@click.option('-o', '--output', metavar='<PATH>', type=click.Path(allow_dash=True),
              help='write the found DFA using DOT language in <PATH> file; if not set, write to logging destination')
@click.option('-b', '--sym-breaking', type=click.Choice(['BFS', 'NOSB', 'TIGHTBFS']), default='BFS', show_default=True,
              help='symmetry breaking strategies')
# TODO: implement timeout
# @click.option('-t', '--timeout', metavar='<SECONDS>', type=int, help='set timeout')
@click.option('-s', '--solver', metavar='<SOLVER>', required=True, help='solver name')
@click.option('-cegar', '--cegar-mode', type=click.Choice(['none', 'lin-abs', 'lin-rel', 'geom']), default='none',
              show_default=True,
              help='counterexamples providing mode for CEGAR')
@click.option('-init', '--initial-amount', metavar='<INT>', type=int, default=10, show_default=True,
              help='initial amount of examples for CEGAR')
@click.option('-step', '--step-amount', metavar='<INT>', type=int, default=10, show_default=True,
              help='amount of examples added on each step for CEGAR')
@click.option('-a', '--assumptions', 'assumptions_mode', type=click.Choice(['none', 'switch', 'chain']),
              default='none', show_default=True, help='assumptions mode')
@click.option('-stat', '--statistics', 'print_statistics', is_flag=True, default=False, show_default=True,
              help='prints time statistics summary in the end')
@click.option('-ig', '--inconsistency-graph', 'use_ig', is_flag=True, default=False, show_default=True,
              help='use inconsistency graph')
@click.version_option(__version__, '-v', '--version')
def cli(input_: str,
        lower_bound: int,
        upper_bound: int,
        output: str,
        sym_breaking: str,
        solver: str,
        cegar_mode: str,
        initial_amount: int,
        step_amount: int,
        assumptions_mode: str,
        print_statistics: bool,
        use_ig: bool) -> None:
    try:
        STATISTICS.start_whole_timer()
        examples_provider = examples.get_examples_provider(input_, cegar_mode, initial_amount, step_amount)

        STATISTICS.start_apta_building_timer()
        apta = APTA(examples_provider.get_init_examples())
        log_success('Successfully built an APTA from file \'{0}\''.format(input_))
        log_info('The APTA size: {0}'.format(apta.size))
        STATISTICS.stop_apta_building_timer()

        if use_ig:
            STATISTICS.start_ig_building_timer()
        ig = InconsistencyGraph(apta, is_empty=not use_ig)
        if use_ig:
            log_success('Successfully built an IG')
            STATISTICS.stop_ig_building_timer()

        searcher = LSUS(apta,
                        ig,
                        solver,
                        sym_breaking,
                        cegar_mode,
                        examples_provider,
                        assumptions_mode)
        dfa = searcher.search(lower_bound, upper_bound)
        if not dfa:
            log_info('There is no such DFA.')
        else:
            if not output:
                log_br()
                log_info(str(dfa))
            else:
                log_info('Dumping found DFA to {0}'.format(output))
                try:
                    with click.open_file(output, mode='w') as file:
                        file.write(str(dfa))
                except IOError as err:
                    log_error('Something wrong with an output file: \'{0}\': {1}'.format(output, err))
                    log_info('Dumping found DFA to console instead.')
                    log_br()
                    log_info(str(dfa))
            if dfa.check_consistency(examples_provider.get_all_examples()):
                log_success('DFA is consistent with the given examples.')
            else:
                log_error('DFA is not consistent with the given examples.')
        STATISTICS.stop_whole_timer()
        if print_statistics:
            STATISTICS.print_statistics()
    except IOError as err:
        log_error('Cannot build an APTA from file \'{0}\': {1}'.format(input_, err))
        sys.exit(err.errno)


if __name__ == "__main__":
    cli()
