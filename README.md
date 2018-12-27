# Qiskit Aqua

[![License](https://img.shields.io/github/license/Qiskit/qiskit-aqua.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://img.shields.io/travis/Qiskit/qiskit-aqua/master.svg?style=popout-square)](https://travis-ci.com/Qiskit/qiskit-aqua)![](https://img.shields.io/pypi/v/qiskit-aqua.svg?style=popout-square)![](https://img.shields.io/pypi/dm/qiskit-aqua.svg?style=popout-square)

**Qiskit** is an open-source framework for working with noisy intermediate-scale quantum computers (NISQ) at the level of pulses, circuits, algorithms, and applications.

Qiskit is made up elements that work together to enable quantum computing. This element is **Aqua**.
Aqua provides a library of cross-domain algorithms upon which domain-specific applications can be
built. [Qiskit Chemistry](https://github.com/Qiskit/qiskit-chemistry) has
been created to utilize Aqua for quantum chemistry computations. Aqua is also showcased for other
domains, such as Optimization, Artificial Intelligence, and
Finance, with both code and notebook examples available in the
[qiskit/aqua](https://github.com/Qiskit/qiskit-tutorials/tree/master/qiskit/aqua)
and [community/aqua](https://github.com/Qiskit/qiskit-tutorials/tree/master/community/aqua)
folders of the [qiskit-tutorials GitHub Repository](https://github.com/Qiskit/qiskit-tutorials).  

Aqua was designed to be extensible, and uses a pluggable framework where algorithms and support objects used
by algorithms—such as optimizers, variational forms, and oracles—are derived from a defined base class for the type and
discovered dynamically at run time.

## Installation

We encourage installing Qiskit Aqua via the PIP tool (a python package manager):

```bash
pip install qiskit-aqua
```

PIP will handle all dependencies automatically for you, including the other Qiskit elements on which
Aqua is built, and you will always install the latest (and well-tested) version.

To install from source, follow the instructions in the [contribution guidelines](.github/CONTRIBUTING.rst).

## Creating your first quantum program in Qiskit Aqua

Now that Qiskit Aqua is installed, it's time to begin working with it.

We are ready to try out an experiment using Qiskit Aqua:

```
$ python
```

```python
>>> from qiskit import Aer
>>> from qiskit_aqua.components.oracles import SAT
>>> from qiskit_aqua.algorithms import Grover
>>> sat_cnf = """
>>> c Example DIMACS 3-sat
>>> p cnf 3 5
>>> -1 -2 -3 0
>>> 1 -2 3 0
>>> 1 2 -3 0
>>> 1 -2 -3 0
>>> -1 2 3 0
>>> """
>>> backend = Aer.get_backend('qasm_simulator')
>>> oracle = SAT(sat_cnf)
>>> algorithm = Grover(oracle)
>>> result = algorithm.run(backend)
>>> print(result["result"])
```

The code above demonstrates how Grover’s search algorithm can be used in conjunction with the
Satisfiability (SAT) oracle to compute one of the many possible solutions of a Conjunctive Normal
Form (CNF).  Variable `sat_cnf` corresponds to the following CNF:

```
(&not;<i>x</i><sub>1</sub> &or; &not;<i>x</i><sub>2</sub> &or; &not;<i>x</i><sub>3</sub>) &and;
(<i>x</i><sub>1</sub> &or; &not;<i>x</i><sub>2</sub> &or; <i>x</i><sub>3</sub>) &and;
(<i>x</i><sub>1</sub> &or; <i>x</i><sub>2</sub> &or; &not;<i>x</i><sub>3</sub>) &and;
(<i>x</i><sub>1</sub> &or; &not;<i>x</i><sub>2</sub> &or; &not;<i>x</i><sub>3</sub>) &and;
(&not;<i>x</i><sub>1</sub> &or; <i>x</i><sub>2</sub> &or; <i>x</i><sub>3</sub>)  
```

The Python code above prints out one possible solution for this CNF. For example, output `1, -2, 3` indicates
that logical expression (<i>x</i><sub>1</sub> &or; &not;<i>x</i><sub>2</sub> &or; <i>x</i><sub>3</sub>) satisfies the given CNF.

You can also use Qiskit to execute your code on a
**real quantum chip**.
In order to do so, you need to configure Qiskit for using the credentials in
your IBM Q account.  Please consult the relevant instructions in the
[Qiskit Terra GitHub repository](https://github.com/Qiskit/qiskit-terra/blob/master/README.md#executing-your-code-on-a-real-quantum-chip). 

## Contribution guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](.github/CONTRIBUTING.rst). This project adheres to Qiskit's [code of conduct](.github/CODE_OF_CONDUCT.rst). By participating, you are expected to uphold to this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-aqua/issues) for tracking requests and bugs. Please use the [Aqua Slack channel](https://qiskit.slack.com/messages/aqua)
for discussion and simple questions.  To join our Slack community use the [link](https://join.slack.com/t/qiskit/shared_invite/enQtNDc2NjUzMjE4Mzc0LTMwZmE0YTM4ZThiNGJmODkzN2Y2NTNlMDIwYWNjYzA2ZmM1YTRlZGQ3OGM0NjcwMjZkZGE0MTA4MGQ1ZTVmYzk).
For questions that are more suited for a forum we use the Qiskit tag in the [Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit).

### Next Steps

Now you're set up and ready to check out some of the other examples from our
[Qiskit Tutorials](https://github.com/Qiskit/qiskit-tutorials) repository.

## Authors

Aqua was inspired, authored and brought about by the collective work of a team of researchers.
Aqua continues to grow with the help and work of [many people](./CONTRIBUTORS.rst), who contribute
to the project at different levels.

## License

[Apache License 2.0](LICENSE.txt)


