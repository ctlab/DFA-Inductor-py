#  author: Konstantin Chukharev, https://github.com/Lipen
from typing import Optional


def log(text: str, symbol: Optional[str], *, fg: str = None, bg: str = None, bold: bool = None,
        nl: bool = True) -> None:
    if symbol is None:
        pre = ''
    else:
        pre = '[{: >1}] '.format(symbol)
    import click
    click.secho('{}{}'.format(pre, text), fg=fg, bg=bg, bold=bold, nl=nl)


def log_debug(text: str) -> None:
    log(text, '.', fg='white')


def log_info(text: str) -> None:
    log(text, '*', fg='blue', bold=True)


def log_success(text: str) -> None:
    log(text, '+', fg='green', bold=True)


def log_time(text: str, time: float) -> None:
    log('{}: {:.2f}'.format(text, time), symbol='t', fg='cyan', bold=None)


def log_statistics(text: str, time: float) -> None:
    log('{}: {:.2f}'.format(text, time), symbol='s', fg='bright_cyan', bold=True)


def log_warn(text):
    log(text, '!', fg='magenta', bold=True)


def log_error(text):
    log(text, '#', fg='red', bold=True)


def log_br():
    log(' '.join('=' * 40), symbol=None, fg='white', bold=False)
