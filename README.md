# Qiskit Aqua

[![License](https://img.shields.io/github/license/Qiskit/qiskit-aqua.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://img.shields.io/travis/Qiskit/qiskit-aqua/master.svg?style=popout-square)](https://travis-ci.org/Qiskit/qiskit-aqua)[![](https://img.shields.io/github/release/Qiskit/qiskit-aqua.svg?style=popout-square)](https://github.com/Qiskit/qiskit-aqua/releases)[![](https://img.shields.io/pypi/dm/qiskit-aqua.svg?style=popout-square)](https://pypi.org/project/qiskit-aqua/)[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-aqua/badge.svg?branch=master)](https://coveralls.io/github/Qiskit/qiskit-aqua?branch=master)

**Qiskit** is an open-source framework for working with noisy quantum computers at the level of pulses, circuits, and algorithms.

Qiskit is made up elements that work together to enable quantum computing. This element is **Aqua**
(Algorithms for QUantum computing Applications) providing a library of cross-domain algorithms
upon which domain-specific applications can be built.
 
* [Aqua](#aqua)
 
Aqua includes domain application support for:

* [Chemistry](#chemistry)
* [Finance](#finance)
* [Machine Learning](#machine-learning)
* [Optimization](#optimization)

_**Note**: the Chemistry application stack was the first domain worked on. At the time of writing
the other domains have some logic in them but are not as fully realised. Future work is expected to
build out functionality in all application areas._ 

Aqua was designed to be extensible, and uses a framework where algorithms and support objects used
by algorithms, such as optimizers, variational forms, and oracles etc,. are derived from a defined 
base class for the type. These along with other building blocks provide a means for end-users and
developers alike to have flexibility and facilitate building and experimenting with different
configurations and capability.

Note: Aqua provides some classical algorithms that take the same input data as quantum algorithms
solving the same problem. For instance a Hamiltonian operator input to VQE can be used as an input
to the ExactEigensolver. This allows near-term experiments, that can still be solved classically,
for their outcome to be compared and used as reference etc.

## Installation

We encourage installing Qiskit via the pip tool (a python package manager), which installs all
Qiskit elements, including Aqua.

```bash
pip install qiskit
```

**pip** will handle all dependencies automatically and you will always install the latest
(and well-tested) version.

If you want to work on the very latest work-in-progress versions, either to try features ahead of
their official release or if you want to to contribute to Aqua, then you can install from source.
To do this follow the instructions in the
 [documentation](https://qiskit.org/documentation/contributing_to_qiskit.html#installing-from-source).

Note: there some optional packages that can be installed such as IBM CPLEX for Aqua and ab-initio
chemistry libraries/programs. Refer to Optional Install information in the sections below.
 
Note: _Optional install links are currently pointing to the source documentation in the code.
At the time of writing the qiskit.org [API Documentation](https://qiskit.org/documentation) 
is being reworked and these links will be redone to point there once the documentation is
updated and republished._ 

----------------------------------------------------------------------------------------------------

## Aqua

The `qiskit.aqua` package contains the core cross-domain algorithms and supporting logic to run
these on a quantum backend, whether a real device or simulator. 

### Optional Installs

* **IBM CPLEX** may be [installed](qiskit/aqua/algorithms/classical/cplex/__init__.py) 
  to allow use of the `CPLEX_Ising` classical solver algorithm.
* **PyTorch**, may be installed either using command `pip install qiskit-aqua[torch]` to install the
  package or refer to PyTorch [getting started](https://pytorch.org/get-started/locally/). PyTorch
  being installed will enable the neural networks `PyTorchDiscriminator` component to be used with
  the QGAN algorithm.  
  

### Creating Your First Quantum Program in Qiskit Aqua

Now that Qiskit is installed, it's time to begin working with Aqua.
Let's try an experiment using `Grover`'s algorithm that is supplied with Aqua:

```
$ python
```

```python
from qiskit import Aer
from qiskit.aqua.components.oracles import LogicalExpressionOracle
from qiskit.aqua.algorithms import Grover

sat_cnf = """
c Example DIMACS 3-sat
p cnf 3 5
-1 -2 -3 0
1 -2 3 0
1 2 -3 0
1 -2 -3 0
-1 2 3 0
"""

backend = Aer.get_backend('qasm_simulator')
oracle = LogicalExpressionOracle(sat_cnf)
algorithm = Grover(oracle)
result = algorithm.run(backend)
print(result["result"])
```

The code above demonstrates how `Grover`’s search algorithm can be used with the
`LogicalExpressionOracle` to find one satisfying assignment
for the Satisfiability (SAT) problem instance encoded in the
[DIMACS CNF format](http://www.satcompetition.org/2009/format-benchmarks2009.html).
The input string `sat_cnf` corresponds to the following Conjunctive Normal
Form (CNF):

(&not;<i>x</i><sub>1</sub> &or; &not;<i>x</i><sub>2</sub> &or; &not;<i>x</i><sub>3</sub>) &and;
(<i>x</i><sub>1</sub> &or; &not;<i>x</i><sub>2</sub> &or; <i>x</i><sub>3</sub>) &and;
(<i>x</i><sub>1</sub> &or; <i>x</i><sub>2</sub> &or; &not;<i>x</i><sub>3</sub>) &and;
(<i>x</i><sub>1</sub> &or; &not;<i>x</i><sub>2</sub> &or; &not;<i>x</i><sub>3</sub>) &and;
(&not;<i>x</i><sub>1</sub> &or; <i>x</i><sub>2</sub> &or; <i>x</i><sub>3</sub>)

The Python code above prints out one possible solution for this CNF.
For example, output `1, -2, 3` indicates that logical expression 
(<i>x</i><sub>1</sub> &or; &not;<i>x</i><sub>2</sub> &or; <i>x</i><sub>3</sub>)
satisfies the given CNF.

### Further examples

Jupyter notebooks containing further examples, for Qiskit Aqua, may be found here in the following
Qiskit GitHub repositories at
[qiskit-iqx-tutorials/qiskit/advanced/aqua](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/aqua)
and
[qiskit-community-tutorials/aqua](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/aqua).

----------------------------------------------------------------------------------------------------

## Chemistry

The `qiskit.chemistry` package supports problems including ground state energy computations,
excited states and dipole moments of molecule, both open and closed-shell.  

The code comprises chemistry drivers, which when provided with a molecular
configuration will return one and two-body integrals as well as other data that is efficiently 
computed classically. This output data from a driver can then be used as input to the chemistry
application stack that contains logic which is able to translate this into a form that is suitable
for quantum algorithms. The conversion first creates a FermionicOperator which must then be mapped,
e.g. by a Jordan Wigner mapping, to a qubit operator in readiness for the quantum computation. 

### Optional Installs

To run chemistry experiments using Qiskit Chemistry, it is recommended that you to install a
classical computation chemistry software program/library interfaced by Qiskit Chemistry. 
Several, as listed below, are supported, and while logic to interface these programs is supplied by
Qiskit Chemistry via the above pip installation, the dependent programs/libraries themselves need
to be installed separately.

Note: As `PySCF` can be installed via pip the installation of Qiskit (Aqua) will install PySCF
where it's supported (MacOS and Linux x86). For other platforms see the PySCF information as to
whether this might be possible manually. 

1. [Gaussian 16&trade;](qiskit/chemistry/drivers/gaussiand/__init__.py), a commercial chemistry program
2. [PSI4](qiskit/chemistry/drivers/psi4d/__init__.py), a chemistry program that exposes a Python interface allowing for accessing internal objects
3. [PySCF](qiskit/chemistry/drivers/pyscfd/__init__.py), an open-source Python chemistry program
4. [PyQuante](qiskit/chemistry/drivers/pyquanted/__init__.py), a pure cross-platform open-source Python chemistry program

### HDF5 Driver

A useful functionality integrated into Qiskit Chemistry is its ability to serialize a file in
Hierarchical Data Format 5 (HDF5) format representing all the output data from a chemistry driver.
 
The [HDF5 driver](qiskit/chemistry/drivers/hdf5d/hdf5driver.py)
accepts such such HDF5 files as input so molecular experiments can be run, albeit on the fixed data
as stored in the file. As such, if you have some pre-created HDF5 files from created from Qiskit
Chemistry, you can use these with the HDF5 driver even if you do not install one of the classical
computation packages listed above.  

A few sample HDF5 files for different are provided in the 
[chemistry folder](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/chemistry)
of the [Qiskit Community Tutorials](https://github.com/Qiskit/qiskit-community-tutorials)
repository. This
[HDF5 Driver tutorial](https://github.com/Qiskit/qiskit-community-tutorials/blob/master/chemistry/hdf5_files_and_driver.ipynb)
contains further information about creating and using such HDF5 files.

### Creating Your First Qiskit Chemistry Programming Experiment

Now that Qiskit is installed, it's time to begin working with Aqua.
Let's try a chemistry application experiment using VQE (Variational Quantum Eigensolver) algorithm
to compute the ground-state (minimum) energy of a molecule.  

```python
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.aqua.operators import Z2Symmetries

# Use PySCF, a classical computational chemistry software
# package, to compute the one-body and two-body integrals in
# molecular-orbital basis, necessary to form the Fermionic operator
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                    unit=UnitsType.ANGSTROM,
                    basis='sto3g')
molecule = driver.run()
num_particles = molecule.num_alpha + molecule.num_beta
num_spin_orbitals = molecule.num_orbitals * 2

# Build the qubit operator, which is the input to the VQE algorithm in Aqua
ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
map_type = 'PARITY'
qubitOp = ferOp.mapping(map_type)
qubitOp = Z2Symmetries.two_qubit_reduction(qubitOp, num_particles)
num_qubits = qubitOp.num_qubits

# setup a classical optimizer for VQE
from qiskit.aqua.components.optimizers import L_BFGS_B
optimizer = L_BFGS_B()

# setup the initial state for the variational form
from qiskit.chemistry.components.initial_states import HartreeFock
init_state = HartreeFock(num_qubits, num_spin_orbitals, num_particles)

# setup the variational form for VQE
from qiskit.aqua.components.variational_forms import RYRZ
var_form = RYRZ(num_qubits, initial_state=init_state)

# setup and run VQE
from qiskit.aqua.algorithms import VQE
algorithm = VQE(qubitOp, var_form, optimizer)

# set the backend for the quantum computation
from qiskit import Aer
backend = Aer.get_backend('statevector_simulator')

result = algorithm.run(backend)
print(result['energy'])
```
The program above uses a quantum computer to calculate the ground state energy of molecular Hydrogen,
H<sub>2</sub>, where the two atoms are configured to be at a distance of 0.735 angstroms. The molecular
input specification is processed by PySCF driver and data is output that includes one- and
two-body molecular-orbital integrals. From the output a fermionic-operator is created which is then
parity mapped to generate a qubit operator. Parity mappings allow a precision-preserving optimization 
that two qubits can be tapered off; a reduction in complexity that is particularly advantageous for NISQ
computers.
 
The qubit operator is then passed as an input to the Variational Quantum Eigensolver (VQE) algorithm,
instantiated with a classical optimizer and an RyRz variational form (ansatz). A Hartree-Fock 
initial state is used as a starting point for the variational form.

The VQE algorithm is then run, in this case on the Qiskit Aer statevector simulator backend.
Here we pass a backend but it can be wrapped into a `QuantumInstance`, and that passed to the
`run` instead. The `QuantumInstance` API allows you to customize run-time properties of the backend,
such as the number of shots, the maximum number of credits to use, settings for the simulator,
initial layout of qubits in the mapping and the Terra `PassManager` that will handle the compilation
of the circuits. By passing in a backend as is done above it is internally wrapped into a 
`QuantumInstance` and is a convenience when default setting suffice.

### Further examples

Jupyter notebooks containing further examples, for Qiskit Chemistry, may be found here in the following
Qiskit GitHub repositories at
[qiskit-iqx-tutorials/qiskit/advanced/aqua/chemistry](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/aqua/chemistry)
and
[qiskit-community-tutorials/chemistry](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/chemistry).

----------------------------------------------------------------------------------------------------

## Finance

The `qiskit.finance` package contains.... 

### Further examples

Jupyter notebooks containing further examples, for Qiskit Finance, may be found here in the following
Qiskit GitHub repositories at
[qiskit-iqx-tutorials/qiskit/advanced/aqua/finance](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/aqua/finance)
and
[qiskit-community-tutorials/finance](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/finance).

----------------------------------------------------------------------------------------------------
 
## Machine Learning

The `qiskit.ml` package contains only sample datasets at present. `qiskit.aqua` does have some
classification algorithms such as QSVM and VQC (Variational Quantum Classifier). There is also  

### Further examples

Jupyter notebooks containing further examples, for Qiskit Finance, may be found here in the following
Qiskit GitHub repositories at
[qiskit-iqx-tutorials/qiskit/advanced/aqua/machine_learning](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/aqua/machine_learning)
and
[qiskit-iqx-tutorials/qiskit/advanced/aqua/finance/machine_learning](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/aqua/finance/machine_learning)
and
[qiskit-community-tutorials/machine_learning](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/machine_learning).

----------------------------------------------------------------------------------------------------

## Optimization

The `qiskit.optimization` package contains.... 

### Further examples

Jupyter notebooks containing further examples, for Qiskit Finance, may be found here in the following
Qiskit GitHub repositories at
[qiskit-iqx-tutorials/qiskit/advanced/aqua/optimization](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/aqua/optimization)
and
[qiskit-iqx-tutorials/qiskit/advanced/aqua/finance/optimization](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/aqua/finance/optimization)
and
[qiskit-community-tutorials/optimization](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/optimization).

----------------------------------------------------------------------------------------------------

## Using a Real Device

You can also use Qiskit to execute your code on a **real quantum chip**.
In order to do so, you need to configure Qiskit to use the credentials in
your [IBM Quantum Experience](https://quantum-computing.ibm.com) account.
For more detailed information refer to the relevant instructions in the
[Qiskit Terra GitHub repository](https://github.com/Qiskit/qiskit-terra/blob/master/README.md#executing-your-code-on-a-real-quantum-chip)
.

## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](./CONTRIBUTING.md).
This project adheres to Qiskit's [code of conduct](./CODE_OF_CONDUCT.md).
By participating, you are expected to uphold to this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-aqua/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://join.slack.com/t/qiskit/shared_invite/enQtODQ2NTIyOTgwMTQ3LTI0NzM2NzkzZjJhNDgzZjY5MTQzNDY3MGNiZGQzNTNkZTE4Nzg1MjMwMmFjY2UwZTgyNDlmYWQwYmZjMjE1ZTM)
and use the [Aqua Slack channel](https://qiskit.slack.com/messages/aqua) for discussion and simple questions.
For questions that are more suited for a forum, we use the **Qiskit** tag in [Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from the 
[Qiskit IQX Tutorials](https://github.com/Qiskit/qiskit-iqx-tutorials)
repository, that are used for the IBM Quantum Experience, and from the 
[Qiskit Community Tutorials](https://github.com/Qiskit/qiskit-community-tutorials). 


## Authors and Citation

Aqua was inspired, authored and brought about by the collective work of a team of researchers.
Aqua continues to grow with the help and work of 
[many people](https://github.com/Qiskit/qiskit-aqua/graphs/contributors), who contribute
to the project at different levels. 
If you use Qiskit, please cite as per the provided 
[BibTeX file](https://github.com/Qiskit/qiskit/blob/master/Qiskit.bib).

Please note that if you do not like the way your name is cited in the BibTex file then consult
the information found in the [.mailmap](https://github.com/Qiskit/qiskit-aqua/blob/master/.mailmap)
file. 

## License

This project uses the [Apache License 2.0](LICENSE.txt).

However there is some code that is included under other licensing as follows:

* The [Gaussian 16 driver](qiskit/chemistry/drivers/gaussiand) in `qiskit.chemistry`
  contains [work](qiskit/chemistry/drivers/gaussiand/gauopen)) licensed under the
  [Gaussian Open-Source Public License](qiskit/chemistry/drivers/gaussiand/gauopen/LICENSE.txt).
