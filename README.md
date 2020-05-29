# DFA-Inductor-py
A python tool for solving minDFA problem.

[![Build Status](https://travis-ci.org/ctlab/DFA-Inductor-py.svg?branch=master)](https://travis-ci.org/ctlab/DFA-Inductor-py)

## Requirements 

Need to be installed manually:

* python 3.7+
* [PySAT](https://github.com/pysathq/pysat)
* click

## Installation

Clone the repository and install it via **pip**, **pipx** (*recommended*), or **pipenv**:

```shell script
pip install .
pipx install .
pipenv install
```

## Usage

For a list of options check <b>`dfainductor --help`</b>.

TODO: add details

#### SAT solvers
 
All the work with SAT solvers is done by [PySAT toolkit](https://github.com/pysathq/pysat).

One can check the list of available options [here](https://pysathq.github.io/docs/html/api/solvers.html#pysat.solvers.SolverNames).