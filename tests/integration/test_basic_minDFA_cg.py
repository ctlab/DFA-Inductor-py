from click.testing import CliRunner

from dfainductor import cli


def test_basic_minDFA_ig_BFS_5(dfa_5_states):
    basic_minDFA_ig_BFS(dfa_5_states, 5)


def test_basic_minDFA_ig_BFS_10(dfa_10_states):
    basic_minDFA_ig_BFS(dfa_10_states, 10)


def basic_minDFA_ig_BFS(input_, size):
    runner = CliRunner()
    result = runner.invoke(cli, ['-i', input_, '-b', 'BFS', '-s', 'g4', '-ig'])
    assert f'[+] Successfully built an APTA from file \'{input_}\'' in result.output
    assert '[+] Successfully built an IG' in result.output
    assert f'[+] The DFA with {size} states is found!' in result.output
    assert '[+] DFA is consistent with the given examples.' in result.output
