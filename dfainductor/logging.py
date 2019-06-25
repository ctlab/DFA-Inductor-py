#  author: Konstantin Chukharev, https://github.com/Lipen

import click

__all__ = ['log_debug', 'log_info', 'log_success', 'log_warn', 'log_error', 'log_br']


def log(text, symbol, *, fg=None, bg=None, bold=None, nl=True):
    if symbol is None:
        pre = ''
    else:
        pre = '[{: >1}] '.format(symbol)
    click.secho('{}{}'.format(pre, text), fg=fg, bg=bg, bold=bold, nl=nl)


def log_debug(text, symbol='.', *, fg='white', bg=None, bold=None, nl=True):
    log(text, symbol, fg=fg, bg=bg, bold=bold, nl=nl)


def log_info(text, symbol='*', *, fg='blue', bg=None, bold=True, nl=True):
    log(text, symbol, fg=fg, bg=bg, bold=bold, nl=nl)


def log_success(text, symbol='+', *, fg='green', bg=None, bold=True, nl=True):
    log(text, symbol, fg=fg, bg=bg, bold=bold, nl=nl)


def log_warn(text, symbol='!', *, fg='magenta', bg=None, bold=True, nl=True):
    log(text, symbol, fg=fg, bg=bg, bold=bold, nl=nl)


def log_error(text, symbol='!', *, fg='red', bg=None, bold=True, nl=True):
    log(text, symbol, fg=fg, bg=bg, bold=bold, nl=nl)


def log_br(*, fg='white', bg=None, bold=False, nl=True):
    log(' '.join('=' * 40), symbol=None, fg=fg, bg=bg, bold=bold, nl=nl)
