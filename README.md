# Qiskit Aqua (_NOW DEPRECATED_)

[![License](https://img.shields.io/github/license/Qiskit/qiskit-aqua.svg?style=popout-square)](https://opensource.org/licenses/Apache-2.0)[![Build Status](https://github.com/Qiskit/qiskit-aqua/workflows/Aqua%20Unit%20Tests/badge.svg?branch=main)](https://github.com/Qiskit/qiskit-aqua/actions?query=workflow%3A"Aqua%20Unit%20Tests"+branch%3Amain+event%3Apush)[![](https://img.shields.io/github/release/Qiskit/qiskit-aqua.svg?style=popout-square)](https://github.com/Qiskit/qiskit-aqua/releases)[![](https://img.shields.io/pypi/dm/qiskit-aqua.svg?style=popout-square)](https://pypi.org/project/qiskit-aqua/)[![Coverage Status](https://coveralls.io/repos/github/Qiskit/qiskit-aqua/badge.svg?branch=main)](https://coveralls.io/github/Qiskit/qiskit-aqua?branch=main)

---

**_PLEASE NOTE:_** _As of version 0.9.0, released on 2nd April 2021, Qiskit Aqua has been deprecated
with its support ending and eventual archival being no sooner than 3 months from that date. The
function provided by Qiskit Aqua is not going away rather it has being split out to separate
application repositories, with core algorithm and operator function moving to qiskit-terra.
Please see the [Migration Guide](#migration-guide) section below for more detail. We encourage you
to migrate over at your earliest convenience._

---

**Qiskit** is an open-source framework for working with noisy quantum computers at the level of
pulses, circuits, and algorithms.

Qiskit is made up elements that work together to enable quantum computing. This element is **Aqua**
(Algorithms for QUantum computing Applications) providing a library of cross-domain algorithms
upon which domain-specific applications can be built.

* [Aqua](#aqua)

Aqua includes domain application support for:

* [Chemistry](#chemistry)
* [Finance](#finance)
* [Machine Learning](#machine-learning)
* [Optimization](#optimization)

_**Note**: the chemistry module was the first domain worked on. Aqua version 0.7.0 introduced a new
optimization module for solving quadratic problems. At the time of writing the other domains have
some logic in them but are not as fully realised. Future work is expected to
build out functionality in all application areas._

Aqua was designed to be extensible, and uses a framework where algorithms and support objects used
by algorithms, such as optimizers, variational forms, and oracles etc, are derived from a defined
base class for the type. These along with other building blocks provide a means for end-users and
developers alike to have flexibility and facilitate building and experimenting with different
configurations and capability.

_**Note**: Aqua provides some classical algorithms that take the same input data as quantum algorithms
solving the same problem. For instance a Hamiltonian operator input to VQE can be used as an input
to the NumPyEigensolver. This may be useful for near-term quantum experiments, for problems
that can still be solved classically, as their outcome can be easily compared against a classical
equivalent since the same input data can be used._

## Migration Guide

As of version 0.9.0, released on 2nd April 2021, Qiskit Aqua has been deprecated
with its support ending and eventual archival being no sooner than 3 months from that date.

All the functionality that qiskit-aqua provides has been migrated to either new packages or to
other qiskit packages. The application modules that are provided by qiskit-aqua have been split
into several new packages: 

* [qiskit-finance](https://github.com/Qiskit/qiskit-finance/)
  
  Aqua's `qiskit.finance` package was moved here

* [qiskit-machine-learning](https://github.com/Qiskit/qiskit-machine-learning/)

  Aqua's `qiskit.ml` package was moved here

* [qiskit-nature](https://github.com/Qiskit/qiskit-nature/)

  Aqua's `qiskit.chemistry` package was moved here, where this new repository for natural science
  applications, will have a broader scope than chemistry. 
  
* [qiskit-optimization](https://github.com/Qiskit/qiskit-optimization/)

  Aqua's `qiskit.optimization` package was moved here

These new packages can be installed by themselves (via the standard pip install command,
e.g. ``pip install qiskit-nature``) or with the rest of the Qiskit metapackage as
optional extras (e.g. ``pip install 'qiskit[finance,optimization]'`` or
``pip install 'qiskit[all]'``.

The core building blocks for algorithms and of the operator flow have been moved, to become core
function of Qiskit, and now exist as part of

* [qiskit-terra](https://github.com/Qiskit/qiskit-terra/)
  
  See the `qiskit.algorithms` and `qiskit.opflow` packages respectively. A
  [Qiskit Algorithms Migration Guide](https://qiskit.org/documentation/aqua_tutorials/Qiskit%20Algorithms%20Migration%20Guide.html)
  has been created to inform and assist the migration, from using the algorithms as they are
  in Aqua, to their newly refactored equivalents as they exist now in their new location.

#### Migration by package/class

The following table gives a more detailed breakdown that relates the function, 
as it existed in Aqua, to where it now lives after this move.

| Old | New | Library |
| :---: | :---: | :---: |
| qiskit.aqua.algorithms.amplitude_amplifiers | qiskit.algorithms.amplitude_amplifiers | qiskit-terra |
| qiskit.aqua.algorithms.amplitude_estimators | qiskit.algorithms.amplitude_estimators | qiskit-terra |
| qiskit.aqua.algorithms.classifiers | qiskit_machine_learning.algorithms.classifiers | qiskit-machine-learning |
| qiskit.aqua.algorithms.distribution_learners | qiskit_machine_learning.algorithms.distribution_learners | qiskit-machine-learning |
| qiskit.aqua.algorithms.eigen_solvers | qiskit.algorithms.eigen_solvers | qiskit-terra |
| qiskit.aqua.algorithms.factorizers | qiskit.algorithms.factorizers | qiskit-terra |
| qiskit.aqua.algorithms.minimum_eigen_solvers | qiskit.algorithms.minimum_eigen_solvers | qiskit-terra|
| qiskit.aqua.algorithms.VQAlgorithm | qiskit.algorithms.VariationalAlgorithm | qiskit-terra |
| qiskit.aqua.aqua_globals | qiskit.utils.algorithm_globals | qiskit-terra|
| qiskit.aqua.components.multiclass_extensions | | |
| qiskit.aqua.components.neural_networks | qiskit_machine_learning.algorithms.distribution_learners.qgan | qiskit-machine-learning |
| qiskit.aqua.components.optimizers | qiskit.algorithms.optimizers | qiskit-terra |
| qiskit.aqua.components.variational_forms | |
| qiskit.aqua.operators | qiskit.opflow | qiskit-terra|
| qiskit.aqua.QuantumInstance | qiskit.utils.QuantumInstance | qiskit-terra |
| qiskit.chemistry | qiskit_nature | qiskit-nature|
| qiskit.finance | qiskit_finance | qiskit-finance|
| qiskit.ml | qiskit_machine_learning | qiskit-machine-learning |
| qiskit.optimization | qiskit_optimization | qiskit-optimization |

## Installation

We encourage installing Qiskit via the pip tool (a python package manager), which installs all
Qiskit elements, including Aqua.

```bash
pip install qiskit
```

**pip** will handle all dependencies automatically and you will always install the latest
(and well-tested) version.

If you want to work on the very latest work-in-progress versions, either to try features ahead of
their official release or if you want to contribute to Aqua, then you can install from source.
To do this follow the instructions in the
 [documentation](https://qiskit.org/documentation/contributing_to_qiskit.html#installing-from-source).

_**Note**: there some optional packages that can be installed such as IBM CPLEX for Aqua and
ab-initio chemistry libraries/programs. Refer to Optional Install information in the sections
below._

----------------------------------------------------------------------------------------------------

## Aqua

The `qiskit.aqua` package contains the core cross-domain algorithms and supporting logic to run
these on a quantum backend, whether a real device or simulator.

* [API reference](https://qiskit.org/documentation/apidoc/qiskit_aqua.html)

### Optional Installs

_**Note:** while the packages below can be installed directly by pip install, e.g. `pip install cplex`
by doing so via the Aqua extra_requires, in this case `pip 'install qiskit-aqua[cplex]'` will ensure
that a version compatible with Qiskit is installed._

* **IBM CPLEX** may be [installed](https://qiskit.org/documentation/apidoc/qiskit.aqua.algorithms.minimum_eigen_solvers.cplex.html)
  to allow the use of the `CplexOptimizer` classical solver algorithm.
  `pip 'install qiskit-aqua[cplex]'` may be used to install the community version.
* **PyTorch**, may be installed either using command `pip install 'qiskit-aqua[torch]'` to install the
  package or refer to PyTorch [getting started](https://pytorch.org/get-started/locally/). PyTorch
  being installed will enable the neural networks `PyTorchDiscriminator` component to be used with
  the QGAN algorithm.
* **CVXPY**, may be installed using command `pip install 'qiskit-aqua[cvx]'` to enable use of the
  `QSVM` and the classical `SklearnSVM` algorithms.


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
print(result.assignment)
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

Learning path notebooks may be found in the
[algorithms tutorials](https://qiskit.org/documentation/tutorials/algorithms/index.html) section
of the documentation and are a great place to start

Jupyter notebooks containing further examples, for Qiskit Aqua, may be found here in the following
Qiskit GitHub repositories at
[qiskit-tutorials/tutorials/algorithms](https://github.com/Qiskit/qiskit-tutorials/tree/master/tutorials/algorithms)
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

* [API reference](https://qiskit.org/documentation/apidoc/qiskit_chemistry.html)

### Optional Installs

To run chemistry experiments using Qiskit's chemistry module, it is recommended that you install
a classical computation chemistry software program/library interfaced by Qiskit.
Several, as listed below, are supported, and while logic to interface these programs is supplied by
the chemistry module via the above pip installation, the dependent programs/libraries themselves need
to be installed separately.

1. [Gaussian 16&trade;](https://qiskit.org/documentation/apidoc/qiskit.chemistry.drivers.gaussiand.html), a commercial chemistry program
2. [PSI4](https://qiskit.org/documentation/apidoc/qiskit.chemistry.drivers.psi4d.html), a chemistry program that exposes a Python interface allowing for accessing internal objects
3. [PySCF](https://qiskit.org/documentation/apidoc/qiskit.chemistry.drivers.pyscfd.html), an open-source Python chemistry program
4. [PyQuante](https://qiskit.org/documentation/apidoc/qiskit.chemistry.drivers.pyquanted.html), a pure cross-platform open-source Python chemistry program

### HDF5 Driver

A useful functionality integrated into Qiskit's chemistry module is its ability to serialize a file
in hierarchical Data Format 5 (HDF5) format representing all the output data from a chemistry driver.

The [HDF5 driver](https://qiskit.org/documentation/stubs/qiskit.chemistry.drivers.HDF5Driver.html#qiskit.chemistry.drivers.HDF5Driver)
accepts such HDF5 files as input so molecular experiments can be run, albeit on the fixed data
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
from qiskit.chemistry.circuit.library import HartreeFock
init_state = HartreeFock(num_spin_orbitals, num_particles)

# setup the variational form for VQE
from qiskit.circuit.library import TwoLocal
var_form = TwoLocal(num_qubits, ['ry', 'rz'], 'cz')

# add the initial state
var_form.compose(init_state, front=True)

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
instantiated with a classical optimizer and a RyRz variational form (ansatz). A Hartree-Fock
initial state is used as a starting point for the variational form.

The VQE algorithm is then run, in this case on the Qiskit Aer statevector simulator backend.
Here we pass a backend but it can be wrapped into a `QuantumInstance`, and that passed to the
`run` instead. The `QuantumInstance` API allows you to customize run-time properties of the backend,
such as the number of shots, the maximum number of credits to use, settings for the simulator,
initial layout of qubits in the mapping and the Terra `PassManager` that will handle the compilation
of the circuits. By passing in a backend as is done above it is internally wrapped into a
`QuantumInstance` and is a convenience when default setting suffice.

### Further examples

Learning path notebooks may be found in the
[chemistry tutorials](https://qiskit.org/documentation/tutorials/chemistry/index.html) section
of the documentation and are a great place to start

Jupyter notebooks containing further chemistry examples may be found in the
following Qiskit GitHub repositories at
[qiskit-tutorials/tutorials/chemistry](https://github.com/Qiskit/qiskit-tutorials/tree/master/tutorials/chemistry)
and
[qiskit-community-tutorials/chemistry](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/chemistry).

----------------------------------------------------------------------------------------------------

## Finance

The `qiskit.finance` package contains uncertainty components for stock/securities problems,
Ising translators for portfolio optimizations and data providers to source real or random data to
finance experiments.

* [API reference](https://qiskit.org/documentation/apidoc/qiskit_finance.html)

### Creating Your First Finance Programming Experiment in Qiskit

Now that Qiskit is installed, it's time to begin working with the finance module.
Let's try an experiment using Amplitude Estimation algorithm to
evaluate a fixed income asset with uncertain interest rates.

```python
import numpy as np
from qiskit import BasicAer
from qiskit.aqua.algorithms import AmplitudeEstimation
from qiskit.circuit.library import NormalDistribution
from qiskit.finance.applications import FixedIncomeExpectedValue

# Create a suitable multivariate distribution
num_qubits = [2, 2]
bounds = [(0, 0.12), (0, 0.24)]
mvnd = NormalDistribution(num_qubits,
                          mu=[0.12, 0.24], sigma=0.01 * np.eye(2),
                          bounds=bounds)

# Create fixed income component
fixed_income = FixedIncomeExpectedValue(num_qubits, np.eye(2), np.zeros(2),
                                        cash_flow=[1.0, 2.0], rescaling_factor=0.125,
                                        bounds=bounds)

# the FixedIncomeExpectedValue provides us with the necessary rescalings
post_processing = fixed_income.post_processing

# create the A operator for amplitude estimation by prepending the
# normal distribution to the function mapping
state_preparation = fixed_income.compose(mvnd, front=True)

# Set number of evaluation qubits (samples)
num_eval_qubits = 5

# Construct and run amplitude estimation
backend = BasicAer.get_backend('statevector_simulator')
algo = AmplitudeEstimation(num_eval_qubits, state_preparation,
                            post_processing=post_processing)
result = algo.run(backend)

print('Estimated value:\t%.4f' % result.estimation)
print('Probability:    \t%.4f' % result.max_probability)
```
When running the above the estimated value result should be 2.46 and probability 0.8487.

### Further examples

Learning path notebooks may be found in the
[finance tutorials](https://qiskit.org/documentation/tutorials/finance/index.html) section
of the documentation and are a great place to start

Jupyter notebooks containing further finance examples may be found in the
following Qiskit GitHub repositories at
[qiskit-tutorials/tutorials/finance](https://github.com/Qiskit/qiskit-tutorials/tree/master/tutorials/finance)
and
[qiskit-community-tutorials/finance](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/finance).

----------------------------------------------------------------------------------------------------

## Machine Learning

The `qiskit.ml` package simply contains sample datasets at present. `qiskit.aqua` has some
classification algorithms such as QSVM and VQC (Variational Quantum Classifier), where this data
can be used for experiments, and there is also QGAN (Quantum Generative Adversarial Network)
algorithm.

* [API reference](https://qiskit.org/documentation/apidoc/qiskit_ml.html)

### Creating Your First Machine Learning Programming Experiment in Qiskit

Now that Qiskit is installed, it's time to begin working with Machine Learning.
Let's try an experiment using VQC (Variational Quantum Classified) algorithm to
train and test samples from a data set to see how accurately the test set can
be classified.

```python
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.aqua.components.feature_maps import RawFeatureVector
from qiskit.ml.datasets import wine
from qiskit.circuit.library import TwoLocal

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
          TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3),
          training_input,
          test_input)
result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                 shots=1024, seed_simulator=seed, seed_transpiler=seed))

print('Testing accuracy: {:0.2f}'.format(result['testing_accuracy']))
```

### Further examples

Learning path notebooks may be found in the
[machine learning tutorials](https://qiskit.org/documentation/tutorials/machine_learning/index.html) section
of the documentation and are a great place to start

Jupyter notebooks containing further Machine Learning examples may be found in the
following Qiskit GitHub repositories at
[qiskit-tutorials/tutorials/machine_learning](https://github.com/Qiskit/qiskit-tutorials/tree/master/tutorials/machine_learning)
and
[qiskit-community-tutorials/machine_learning](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/machine_learning).

----------------------------------------------------------------------------------------------------

## Optimization

The `qiskit.optimization` package covers the whole range from high-level modeling of optimization
problems, with automatic conversion of problems to different required representations, to a suite
of easy-to-use quantum optimization algorithms that are ready to run on classical simulators,
as well as on real quantum devices via Qiskit.

This optimization module enables easy, efficient modeling of optimization problems using
[docplex](https://developer.ibm.com/docloud/documentation/optimization-modeling/modeling-for-python/).
A uniform interface as well as automatic conversion between different problem representations
allows users to solve problems using a large set of algorithms, from variational quantum algorithms,
such as the Quantum Approximate Optimization Algorithm
[QAOA](https://qiskit.org/documentation/stubs/qiskit.aqua.algorithms.QAOA.html),
to [Grover Adaptive Search](https://arxiv.org/abs/quant-ph/9607014>) using the
[GroverOptimizer](https://qiskit.org/documentation/stubs/qiskit.optimization.algorithms.GroverOptimizer.html)
leveraging fundamental algorithms provided by Aqua. Furthermore, the modular design
of the optimization module allows it to be easily extended and facilitates rapid development and
testing of new algorithms. Compatible classical optimizers are also provided for testing,
validation, and benchmarking.

* [API reference](https://qiskit.org/documentation/apidoc/qiskit_optimization.html)

### Optional Installs

* **IBM CPLEX** may be installed using `pip install 'qiskit-aqua[cplex]'` to allow the use of the
 `CplexOptimzer` classical solver algorithm, as well as enabling the reading of `LP` files.

### Creating Your First Optimization Programming Experiment in Qiskit

Now that Qiskit is installed, it's time to begin working with the optimization module.
Let's try an optimization experiment to compute the solution of a
[Max-Cut](https://en.wikipedia.org/wiki/Maximum_cut). The Max-Cut problem can be formulated as
quadratic program, which can be solved using many several different algorithms in Qiskit.
In this example, the [MinimumEigenOptimizer](https://qiskit.org/documentation/stubs/qiskit.optimization.algorithms.MinimumEigenOptimizer.html)
is employed in combination with the Quantum Approximate Optimization Algorithm (QAOA) as minimum
eigensolver routine.

```python
import networkx as nx
import numpy as np

from qiskit.optimization import QuadraticProgram
from qiskit.optimization.algorithms import MinimumEigenOptimizer

from qiskit import BasicAer
from qiskit.aqua.algorithms import QAOA
from qiskit.aqua.components.optimizers import SPSA

# Generate a graph of 4 nodes
n = 4
graph = nx.Graph()
graph.add_nodes_from(np.arange(0, n, 1))
elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
graph.add_weighted_edges_from(elist)

# Compute the weight matrix from the graph
w = nx.adjacency_matrix(graph)

# Formulate the problem as quadratic program
problem = QuadraticProgram()
_ = [problem.binary_var('x{}'.format(i)) for i in range(n)]  # create n binary variables
linear = w.dot(np.ones(n))
quadratic = -w
problem.maximize(linear=linear, quadratic=quadratic)

# Fix node 0 to be 1 to break the symmetry of the max-cut solution
problem.linear_constraint([1, 0, 0, 0], '==', 1)

# Run quantum algorithm QAOA on qasm simulator
spsa = SPSA(max_trials=250)
backend = BasicAer.get_backend('qasm_simulator')
qaoa = QAOA(optimizer=spsa, p=5, quantum_instance=backend)
algorithm = MinimumEigenOptimizer(qaoa)
result = algorithm.solve(problem)
print(result)  # prints solution, x=[1, 0, 1, 0], the cost, fval=4
```

### Further examples

Learning path notebooks may be found in the
[optimization tutorials](https://qiskit.org/documentation/tutorials/optimization/index.html) section
of the documentation and are a great place to start.

Jupyter notebooks containing further examples, for the optimization module, may be found in the
following Qiskit GitHub repositories at
[qiskit-tutorials/tutorials/optimization](https://github.com/Qiskit/qiskit-tutorials/tree/master/tutorials/optimization)
and
[qiskit-community-tutorials/optimization](https://github.com/Qiskit/qiskit-community-tutorials/tree/master/optimization).

----------------------------------------------------------------------------------------------------

## Using a Real Device

You can also use Qiskit to execute your code on a **real quantum chip**.
In order to do so, you need to configure Qiskit to use the credentials in
your [IBM Quantum Experience](https://quantum-computing.ibm.com) account.
For more detailed information refer to the relevant instructions in the
[Qiskit Terra GitHub repository](https://github.com/Qiskit/qiskit-terra/blob/main/README.md#executing-your-code-on-a-real-quantum-chip)
.

## Contribution Guidelines

If you'd like to contribute to Qiskit, please take a look at our
[contribution guidelines](./CONTRIBUTING.md).
This project adheres to Qiskit's [code of conduct](./CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.

We use [GitHub issues](https://github.com/Qiskit/qiskit-aqua/issues) for tracking requests and bugs. Please
[join the Qiskit Slack community](https://ibm.co/joinqiskitslack)
and use the [Aqua Slack channel](https://qiskit.slack.com/messages/aqua) for discussion and simple questions.
For questions that are more suited for a forum, we use the **Qiskit** tag in [Stack Overflow](https://stackoverflow.com/questions/tagged/qiskit).

## Next Steps

Now you're set up and ready to check out some of the other examples from the
[Qiskit Tutorials](https://github.com/Qiskit/qiskit-tutorials)
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
the information found in the [.mailmap](https://github.com/Qiskit/qiskit-aqua/blob/main/.mailmap)
file.

## License

This project uses the [Apache License 2.0](LICENSE.txt).

However there is some code that is included under other licensing as follows:

* The [Gaussian 16 driver](qiskit/chemistry/drivers/gaussiand) in `qiskit.chemistry`
  contains [work](qiskit/chemistry/drivers/gaussiand/gauopen) licensed under the
  [Gaussian Open-Source Public License](qiskit/chemistry/drivers/gaussiand/gauopen/LICENSE.txt).

