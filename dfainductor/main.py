import click

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
@click.option('-t', '--timeout', metavar='<SECONDS>', type=int, help='set timeout')
@click.option('-s', '--solver', metavar="<SOLVER>", required=True, help='solver name')
@click.option('--incremental', is_flag=True, help='Use the given SAT solver in incremental mode.')
def cli(input, lower_bound, upper_bound, output, sym_breaking, timeout, solver, incremental):
    print(str(APTA(input)))
