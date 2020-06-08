import pytest


@pytest.fixture(scope='session')
def dfa_5_states():
    yield "tests/integration/fixtures/dfa_300_2_5"


@pytest.fixture(scope='session')
def dfa_10_states():
    yield "tests/integration/fixtures/dfa_300_2_10"
