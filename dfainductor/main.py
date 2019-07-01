import sys

import click

from .__version__ import __version__
from .algorithms.searchers import LSUS
from .logging import *
from .structures import APTA


@click.command(context_settings=dict(
                   max_content_width=999,
                   help_option_names=['-h', '--help']
               ))
@click.option('-i', '--input', metavar='<PATH>', required=True, type=click.Path(exists=True),
              help='a DFA learning input file in Abbadingo format')
@click.option('-l', '--lower-bound', metavar='<INT>', type=int, default=1, show_default=True,
              help='lower bound of the DFA size')
@click.option('-u', '--upper-bound', metavar='<INT>', type=int, default=100, show_default=True,
              help='upper bound of the DFA size')
@click.option('-o', '--output', metavar='<PATH>', type=click.Path(allow_dash=True),
              help='write the found DFA using DOT language in <PATH> file; if not set, write to logging destination')
@click.option('-b', '--sym-breaking', type=click.Choice(['BFS', 'NOSB']), default='BFS', show_default=True,
              help='symmetry breaking strategies')
# TODO: implement timeout
# @click.option('-t', '--timeout', metavar='<SECONDS>', type=int, help='set timeout')
@click.option('-s', '--solver', metavar='<SOLVER>', required=True, help='solver name')
@click.version_option(__version__, '-v', '--version')
def cli(input, lower_bound, upper_bound, output, sym_breaking, solver):
    try:
        apta = APTA(input)
        log_success('Successfully built an APTA from file \'{0}\''.format(input))
        log_info('The APTA size: {0}'.format(apta.size()))
        searcher = LSUS(lower_bound, upper_bound, apta, solver, sym_breaking)
        dfa = searcher.search()
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
    except IOError as err:
        log_error('Cannot build an APTA from file \'{0}\': {1}'.format(input, err))
        sys.exit(err.errno)


if __name__ == "__main__":
    cli()
