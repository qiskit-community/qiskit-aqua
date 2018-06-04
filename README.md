# QISKit ACQUA

QISKit Algorithms and Circuits for QUantum Applications, `QISKit ACQUA`, is a library of algorithms for quantum computing
that uses [QISkit](https://qiskit.org/) to build out and run quantum circuits. More detail can be found behind the
architecture, motivation and design philosophy of QISKit ACQUA in the [Background](Background) section. 

QISKit ACQUA provides a library of cross-domain algorithms upon which domain specific applications and stacks can be
built. At the time of writing [QISKit ACQUA Chemistry](https://github.ibm.com/IBMQuantum/qiskit-acqua-chemistry) has
been created to utilize QISKit ACQUA for quantum chemistry computations. QISKIt ACQUA is also showcased for other
domains with both code and notebook examples. Please see
[QISKit ACQUA Optimzation](https://github.ibm.com/IBMQuantum/qiskit-acqua-optimization) and
[QISKit ACQUA Artifical Intelligence](https://github.ibm.com/IBMQuantum/qiskit-acqua-artifical-intelligence)

QISKit ACQUA was designed to be extensible and uses a pluggable framework where algorithms and support objects used
by algorithms, e.g optimizers, variational forms, oracles etc. are derived from a defined base class for the type and
are discovered dynamically at runtime.

**If you'd like to contribute to QISKit ACQUA, please take a look at our**
[contribution guidelines](.github/CONTRIBUTING.rst).

Links to Sections:

* [Installation](#installation)
* [Running an algorithm](#running-an-algorithm)
* [Authors](#authors-alphabetical)
* [License](#license)
* [Background](#background)

## Installation

### Dependencies

As QISKit ACQUA is built upon QISKit you are encouraged to look over the 
[QISKit installation](https://github.com/QISKit/qiskit-sdk-py/blob/master/README.md#installation) too.

Like QISKit at least [Python 3.5 or later](https://www.python.org/downloads/) is needed to use QISKit ACQUA. In
addition, [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) is recommended
for interacting with the tutorials.
For this reason we recommend installing the [Anaconda 3](https://www.continuum.io/downloads)
Python distribution, as it comes with all of these dependencies pre-installed.

### Installation

We encourage you to install QISKit ACQUA via the PIP tool (a Python package manager):

```
pip install qiskit_acqua
```

PIP will handle all dependencies automatically and you will always install the latest (and well-tested)
release version.

We recommend using Python virtual environments to improve your experience.


## Running an algorithm

Now that you have installed QISKit ACQUA you can run an algorithm. This can be done programmatically and can also
use json input that was saved from a prior run. The json input can be given to either the UI or the command line tool
to run the algorithm as specified by that json.

### GUI

The GUI allows you load to load and save a json file to run an algorithm as well as create a new one or edit an
existing one. So for example using UI you can alter parameters of algorithm and/or its dependent objects to see how
the changes affect the outcome. PIP install setups a small script to invoke the GUI
as follows:

```> qiskit_acqua_ui```

Note: if you clone and run direct from the repository instead of using the pip install recommended way then it can be
run using `> python qiskit_acqua/ui/run` from the root folder of the qiskit-acqua repository clone. 

### Command Line

The command line tool will run an algorithm from the supplied json file. Run without any arguments it will print some
help. PIP install setups a small script to invoke the command line, e.g:

```> qiskit_acqua_cmd examples/H2-0.735.json```

Note: if you clone and run direct from the repository instead of using the pip install recommended way then it can be
run using `> python qiskit_acqua` from the root folder of the qiskit-acqua repository clone. 

### Browser

As QISKit ACQUA is extensible with pluggable components we have provided a GUI that shows all the pluggable
components along with the schema for their parameters. PIP install setups a small script to invoke the browser GUI
as follows:

```> qiskit_acqua_browser```

Note: if you clone and run direct from the repository instead of using the pip install recommended way then it can be
run using `> python qiskit_acqua/ui/browse` from the root folder of the qiskit-acqua repository clone. 

### Programming

The algorithms can be run programmatically too. Please refer to the [examples](examples) folder for a number of
examples. Here you will see there is a `run_algorithm` method used that takes either the json or an equivalent 
Python dictionary and optional AlgorithmInput object for the algorithm. There is also a `run_algorithm_to_json` that
simply takes the input and saves it to json in a self-contained form that can later be used by the command line or UI. 


## Authors

QISKit ACQUA was inspired, authored and brought about by the collective work of many individuals.

QISKit ACQUA continues now to grow with the help and work of [many people](CONTRIBUTORS.md) who contribute
to the project at different levels.

## License

This project uses the [Apache License Version 2.0 software license](https://www.apache.org/licenses/LICENSE-2.0).

## Background

Quantum computing has the potential to solve problems that, due to their computational complexity, cannot be solved,
either at all or for all practical purposes, on a classical computer.  On the other hand, quantum computing requires
very specialized skills.  Programming to a quantum machine is not an easy task, and requires specialized knowledge.
Problems that can benefit from the power of quantum computing, and for which no computationally affordable solution
has been discovered in the general case on classical computers, have been identified in numerous domains, such as
Chemistry, Artificial Intelligence (AI), Optimization and Finance.

QISKit Algorithms and Circuits for QUantum Applications (QISKit ACQUA) is a library of algorithms that allows
practitioners with different types of expertise to contribute to the quantum software stack at different levels.

Industry-domain experts, who are most likely very familiar with existing computational software specific to their
own domain, may be interested in the benefits of quantum computing in terms of performance, accuracy and computational
complexity, but at the same time they might be unwilling to learn about the underlying quantum infrastructure.
Ideally, such practitioners would like to use the computational software they are used to as a front end to the
quantum computing system.  It is also likely that such practitioners may have collected, over time, numerous problem
configurations corresponding to various experiments. In such cases, it would be desirable for a system that enables
classical computational software to run on a quantum infrastructure, to accept the same configuration files as used
classically, with no modifications, and without requiring a practitioner experienced in a particular domain to have
to learn a quantum programming language.

QISKit ACQUA algorithms and applications allow computational software specific to any domain to be executed on a quantum
computing machine.  The computational software is used both as a form of domain-specific input specification and a
form of quantum-specific input generation.  The specification of the computational problem may be defined using the
classical computational software.  The classical computational software may be executed classically to extract some
additional intermediate data necessary to form the input to the quantum system.  And finally, the problem configuration
and (if present) the additional intermediate data obtained from the classical execution of the computational software
are combined to form the input to the quantum system.

In order to form the input to the quantum machine, the input coming from the classical computational software and the
user-provided configuration needs to be translated.  The translation layer is domain- and problem-specific.
For example, in chemistry, in order to compute some molecular properties, such as the ground-state molecular energy,
dipole moment and excited states of a molecule, QISKit ACQUA Chemistry translates the classically computed input into
a Fermionic Hamiltonian and from that it will generate a Qubit Hamiltonian, which will then be passed to a quantum
algorithm in the QISKit ACQUA library for the energy computation.  Viable algorithms that can solve these problems
quantumly include Variational Quantum Eigensolver (VQE) and Quantum Phase Estimation (QPE).

The quantum algorithm in QISKit ACQUA forms the circuits to be executed by a quantum device or simulator.  The
major novelty of QISKit ACQUA is that the applications running on top of it allow for classical computational software
to be used without having to be wrapped around a common infrastructure,  The users of QISKit ACQUA will not have to
learn a new programming paradigm, and they will be still able to use the computational software they are used to.

A novel characteristic of QISKit ACQUA is that it allows researchers, developers and practitioners with different types 
of expertise to contribute at different levels of the QISKit ACQUA stack, such as the Hamiltonian generation layer,
and the algorithm layer (which includes, among other things, quantum algorithms, optimizers, variational 
forms, and initial states).

A unique feature of QISKit ACQUA is that the software stack is applicable to different domains, 
such as Chemistry, Artificial Intelligence and Optimization.  QISKIT ACQUA is a common infrastructure among 
the various domains, and the application layers built on top of QISKit ACQUA library are all structured according to the 
same architecture.  New domains can be added easily, taking advantage of the shared quantum algorithm infrastructure,
and new algorithms and algorithm components can be plugged in and automatically discovered at run time via
dynamic lookup.

QISKit ACQUA offers another unique feature.  Given that QISKit ACQUA allows traditional software to be executed on a
quantum system, configuring an experiment in a particular domain may require a hybrid configuration that involves both
domain-specific and quantum-specific configuration parameters.  The chances of introducing configuration errors,
making typos, or selecting incompatible configuration parameters are very high, especially for people who are expert
in a given domain but new to the realm of quantum computing.  To address such issues, in QISKit ACQUA the
problem-specific configuration information and the quantum-specific configuration information are dynamically verified
for correctness so that the combination of classical and quantum inputs is resilient to configuration errors.  Very 
importantly, configuration correctness is dynamically enforced even for components that are dynamically discovered and
loaded, which includes traditional computational software packages, input translation modules, algorithms, variational
forms, optimizers, and initial states.

In essence, QISKit ACQUA is a novel software framework that allows users to experience the flexibility provided by the
integration of classical computational software, the error-resilient configuration, the ability to contribute new
components at different levels of the quantum software stack, and the ability to extend QISKit ACQUA to new domains.
