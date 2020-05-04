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

_**Note**: the chemistry module was the first domain worked on. At the time of writing
the other domains have some logic in them but are not as fully realised. Future work is expected to
build out functionality in all application areas._

Aqua was designed to be extensible, and uses a framework where algorithms and support objects used
by algorithms, such as optimizers, variational forms, and oracles etc,. are derived from a defined
base class for the type. These along with other building blocks provide a means for end-users and
developers alike to have flexibility and facilitate building and experimenting with different
configurations and capability.

Note: Aqua provides some classical algorithms that take the same input data as quantum algorithms
solving the same problem. For instance a Hamiltonian operator input to VQE can be used as an input
to the NumPyEigensolver. This may be useful for near-term quantum experiments, for problems
that can still be solved classically, as their outcome can be easily compared against a classical
equivalent since the same input data can be used.

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

* **IBM CPLEX** may be [installed](qiskit/aqua/algorithms/minimum_eigen_solvers/cplex/__init__.py#L16)
  to allow use of the `ClassicalCPLEX` classical solver algorithm.
* **PyTorch**, may be installed either using command `pip install qiskit-aqua[torch]` to install the
  package or refer to PyTorch [getting started](https://pytorch.org/get-started/locally/). PyTorch
  being installed will enable the neural networks `PyTorchDiscriminator` component to be used with
  the QGAN algorithm.


### Creating Your First Quantum Program in Qiskit Aqua

Now that Qiskit is installed, it's time to begin working with Aqua.
Let's try an experiment using `Grover`'s algorithm to find a solution for a
Satisfiability (SAT) problem.

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
module that contains logic which is able to translate this into a form that is suitable
for quantum algorithms. The conversion first creates a FermionicOperator which must then be mapped,
e.g. by a Jordan Wigner mapping, to a qubit operator in readiness for the quantum computation.

### Optional Installs

To run chemistry experiments using Qiskit's chemistry module, it is recommended that you to install
a classical computation chemistry software program/library interfaced by Qiskit.
Several, as listed below, are supported, and while logic to interface these programs is supplied by
the chemistry module via the above pip installation, the dependent programs/libraries themselves need
to be installed separately.

Note: As `PySCF` can be installed via pip the installation of Qiskit (Aqua) will install PySCF
where it's supported (MacOS and Linux x86). For other platforms see the PySCF information as to
whether this might be possible manually.

1. [Gaussian 16&trade;](qiskit/chemistry/drivers/gaussiand/__init__.py#L16), a commercial chemistry program
2. [PSI4](qiskit/chemistry/drivers/psi4d/__init__.py#L16), a chemistry program that exposes a Python interface allowing for accessing internal objects
3. [PySCF](qiskit/chemistry/drivers/pyscfd/__init__.py#L16), an open-source Python chemistry program
4. [PyQuante](qiskit/chemistry/drivers/pyquanted/__init__.py#L16), a pure cross-platform open-source Python chemistry program

### HDF5 Driver

A useful functionality integrated into Qiskit's chemistry module is its ability to serialize a file
in hierarchical Data Format 5 (HDF5) format representing all the output data from a chemistry driver.

The [HDF5 driver](qiskit/chemistry/drivers/hdf5d/hdf5driver.py#L25)
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

### Creating Your First Chemistry Programming Experiment in Qiskit

Now that Qiskit is installed, it's time to begin working with the chemistry module.
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
ferm_op = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
map_type = 'PARITY'
qubit_op = ferm_op.mapping(map_type)
qubit_op = Z2Symmetries.two_qubit_reduction(qubit_op, num_particles)
num_qubits = qubit_op.num_qubits

# setup a classical optimizer for VQE
from qiskit.aqua.components.optimizers import L_BFGS_B
optimizer = L_BFGS_B()

# setup the initial state for the variational form
from qiskit.chemistry.components.initial_states import HartreeFock
init_state = HartreeFock(num_spin_orbitals, num_particles)

# setup the variational form for VQE
from qiskit.circuit.library import TwoLocal
var_form = TwoLocal(num_qubits, ['ry', 'rz'], 'cz', initial_state=init_state)

# setup and run VQE
from qiskit.aqua.algorithms import VQE
algorithm = VQE(qubit_op, var_form, optimizer)

# set the backend for the quantum computation
from qiskit import Aer
backend = Aer.get_backend('statevector_simulator')

result = algorithm.run(backend)
print(result.eigenvalue.real)
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

Jupyter notebooks containing further chemistry examples may be found in the
following Qiskit GitHub repositories at
[qiskit-iqx-tutorials/qiskit/advanced/aqua/chemistry](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/aqua/chemistry)
and
[qiskit-community-tutorials/chemistry](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/chemistry).

----------------------------------------------------------------------------------------------------

## Finance

The `qiskit.finance` package contains uncertainty components for stock/securities problems,
Ising translators for portfolio optimizations and data providers to source real or random data to
finance experiments.

### Creating Your First Finance Programming Experiment in Qiskit

Now that Qiskit is installed, it's time to begin working with the finance module.
Let's try a experiment using Amplitude Estimation algorithm to
evaluate a fixed income asset with uncertain interest rates.

```python
import numpy as np
from qiskit import BasicAer
from qiskit.aqua.algorithms import AmplitudeEstimation
from qiskit.aqua.components.uncertainty_models import MultivariateNormalDistribution
from qiskit.finance.components.uncertainty_problems import FixedIncomeExpectedValue

# Create a suitable multivariate distribution
mvnd = MultivariateNormalDistribution(num_qubits=[2, 2],
                                      low=[0, 0], high=[0.12, 0.24],
                                      mu=[0.12, 0.24], sigma=0.01 * np.eye(2))

# Create fixed income component
fixed_income = FixedIncomeExpectedValue(mvnd, np.eye(2), np.zeros(2),
                                        cash_flow=[1.0, 2.0], c_approx=0.125)

# Set number of evaluation qubits (samples)
num_eval_qubits = 5

# Construct and run amplitude estimation
algo = AmplitudeEstimation(num_eval_qubits, fixed_income)
result = algo.run(BasicAer.get_backend('statevector_simulator'))

print('Estimated value:\t%.4f' % result['estimation'])
print('Probability:    \t%.4f' % result['max_probability'])
```
When running the above the estimated value result should be 2.46 and probability 0.8487.

### Further examples

Jupyter notebooks containing further finance examples may be found in the
following Qiskit GitHub repositories at
[qiskit-iqx-tutorials/qiskit/advanced/aqua/finance](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/aqua/finance)
and
[qiskit-community-tutorials/finance](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/finance).

----------------------------------------------------------------------------------------------------

## Machine Learning

The `qiskit.ml` package simply contains sample datasets at present. `qiskit.aqua` has some
classification algorithms such as QSVM and VQC (Variational Quantum Classifier), where this data
can be used for experiments, and there is also QGAN (Quantum Generative Adversarial Network)
algorithm.

### Creating Your First Machine Learning Programming Experiment in Qiskit

Now that Qiskit is installed, it's time to begin working with Machine Learning.
Let's try a experiment using VQC (Variational Quantum Classified) algorithm to
train and test samples from a data set to see how accurately the test set can
be classified.

```python
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.feature_maps import RawFeatureVector
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.ml.datasets import wine

seed = 1376
aqua_globals.random_seed = seed

# Use Wine data set for training and test data
feature_dim = 4  # dimension of each data point
_, training_input, test_input, _ = wine(training_size=12,
                                        test_size=4,
                                        n=feature_dim)

feature_map = RawFeatureVector(feature_dimension=feature_dim)
vqc = VQC(COBYLA(maxiter=100),
          feature_map,
          RYRZ(feature_map.num_qubits, depth=3),
          training_input,
          test_input)
result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                 shots=1024, seed_simulator=seed, seed_transpiler=seed))

print('Testing accuracy: {:0.2f}'.format(result['testing_accuracy']))
```

### Further examples

Jupyter notebooks containing further Machine Learning examples may be found in the
following Qiskit GitHub repositories at
[qiskit-iqx-tutorials/qiskit/advanced/aqua/machine_learning](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/aqua/machine_learning)
and
[qiskit-iqx-tutorials/qiskit/advanced/aqua/finance/machine_learning](https://github.com/Qiskit/qiskit-iqx-tutorials/tree/master/qiskit/advanced/aqua/finance/machine_learning)
and
[qiskit-community-tutorials/machine_learning](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/machine_learning).

----------------------------------------------------------------------------------------------------

## Optimization

The `qiskit.optimization` package contains Ising translators for various optimization problems such
as Max-Cut, Traveling Salesman and Vehicle Routing. It also has a has an automatic Ising
generator for a problem model specified by the user as a model in
[docplex](qiskit/optimization/ising/docplex.py#L16).

### Creating Your First Optimization Programming Experiment in Qiskit

Now that Qiskit is installed, it's time to begin working with the optimization module.
Let's try a optimization experiment using QAOA (Quantum Approximate Optimization Algorithm)
to compute the solution of a [Max-Cut](https://en.wikipedia.org/wiki/Maximum_cut) problem using
a docplex model to create the Ising Hamiltonian operator for QAOA.

```python
import networkx as nx
import numpy as np
from docplex.mp.model import Model

from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.components.optimizers import SPSA
from qiskit.optimization.applications.ising import docplex, max_cut
from qiskit.optimization.applications.ising.common import sample_most_likely

# Generate a graph of 4 nodes
n = 4
graph = nx.Graph()
graph.add_nodes_from(np.arange(0, n, 1))
elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
graph.add_weighted_edges_from(elist)
# Compute the weight matrix from the graph
w = np.zeros([n, n])
for i in range(n):
    for j in range(n):
        temp = graph.get_edge_data(i, j, default=0)
        if temp != 0:
            w[i, j] = temp['weight']

# Create an Ising Hamiltonian with docplex.
mdl = Model(name='max_cut')
mdl.node_vars = mdl.binary_var_list(list(range(n)), name='node')
maxcut_func = mdl.sum(w[i, j] * mdl.node_vars[i] * (1 - mdl.node_vars[j])
                      for i in range(n) for j in range(n))
mdl.maximize(maxcut_func)
qubit_op, offset = docplex.get_operator(mdl)

# Run quantum algorithm QAOA on qasm simulator
seed = 40598
aqua_globals.random_seed = seed

spsa = SPSA(max_trials=250)
qaoa = QAOA(qubit_op, spsa, p=5)
backend = BasicAer.get_backend('qasm_simulator')
quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed,
                                    seed_transpiler=seed)
result = qaoa.run(quantum_instance)

x = sample_most_likely(result.eigenstate)
print('energy:', result.eigenvalue.real)
print('time:', result.optimizer_time)
print('max-cut objective:', result.eigenvalue.real + offset)
print('solution:', max_cut.get_graph_solution(x))
print('solution objective:', max_cut.max_cut_value(x, w))
```

### Further examples

Jupyter notebooks containing further examples, for the optimization module, may be found in the
following Qiskit GitHub repositories at
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
  contains [work](qiskit/chemistry/drivers/gaussiand/gauopen) licensed under the
  [Gaussian Open-Source Public License](qiskit/chemistry/drivers/gaussiand/gauopen/LICENSE.txt).
