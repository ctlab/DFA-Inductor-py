# DFA-Inductor-py
A python tool for solving minDFA problem.

[![Build Status](https://travis-ci.org/ctlab/DFA-Inductor-py.svg?branch=master)](https://travis-ci.org/ctlab/DFA-Inductor-py)

## Requirements 

Need to be installed manually:

* Python 3.7+
* [pipenv](https://github.com/pypa/pipenv)
```shell script
pip install --user pipenv
```

Will be installed automatically:

* [PySAT](https://github.com/pysathq/pysat)
* click

## Installation

Clone the repository and use [pipenv](https://pipenv.pypa.io/en/latest/basics/) to install:

It can be installed into virtualenv:
```shell script
pipenv install
```

Or it can be installed globally:
```shell script
pipenv install --system
```

## Usage

If the tool is installed into virtualenv, one can spawn a command from virtualenv:
```shell script
pipenv run dfainductor [args]
```
or can spawn a sub-shell within virtualenv and work in it:
```shell script
pipenv shell
dfainductor [args]
...
exit
```

If the tool installed globally, just run it:
```shell script
dfainductor [args]
```

For detailed information check <b>`dfainductor --help`</b>.

#### SAT solvers
 
All the work with SAT solvers is done by [PySAT toolkit](https://github.com/pysathq/pysat).

One can check the list of available options [here](https://pysathq.github.io/docs/html/api/solvers.html#pysat.solvers.SolverNames).