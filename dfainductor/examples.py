from abc import ABC, abstractmethod

import click

from .structures import DFA


class BaseExamplesProvider(ABC):

    def __init__(self, input_):
        with click.open_file(input_) as file:
            self._examples = []
            examples_number, alphabet_size = [int(x) for x in next(file).split()]
            for __ in range(examples_number):
                self._examples.append(next(file))

    def get_init_examples(self):
        init_examples = []
        for i in range(min(self._init_examples_size(), len(self._examples))):
            init_examples.append(self._examples[i])
        return init_examples

    def get_counter_examples(self, dfa: DFA):
        counter_examples = []
        it = iter(self._examples)
        try:
            while len(counter_examples) < self._counter_examples_size():
                word = next(it)
                word_split = word.split()
                if (word_split[0] == '1') != dfa.run(word_split[2:]):
                    counter_examples.append(word)
        except StopIteration:
            pass
        return counter_examples

    @abstractmethod
    def _init_examples_size(self):
        pass

    @abstractmethod
    def _counter_examples_size(self):
        pass


class LinearAbsoluteExamplesProvider(BaseExamplesProvider):
    def __init__(self, input_, initial_examples_amount, counter_examples_amount):
        super().__init__(input_)
        self._initial_examples_amount = initial_examples_amount
        self._counter_examples_amount = counter_examples_amount

    def _init_examples_size(self):
        return self._initial_examples_amount

    def _counter_examples_size(self):
        return self._counter_examples_amount


class LinearRelativeExamplesProvider(BaseExamplesProvider):
    def __init__(self, input_, initial_examples_amount, counter_examples_amount):
        super().__init__(input_)
        self._initial_examples_amount = len(self._examples) / 100 * initial_examples_amount
        self._counter_examples_amount = len(self._examples) / 100 * counter_examples_amount

    def _init_examples_size(self):
        return self._initial_examples_amount

    def _counter_examples_size(self):
        return self._counter_examples_amount


class NonCegarExamplesProvider(BaseExamplesProvider):
    def _init_examples_size(self):
        return len(self._examples)

    def _counter_examples_size(self):
        return 0
