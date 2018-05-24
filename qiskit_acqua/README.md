# IBM Quantum Library

The IBM Quantum Library is a set of algorithms and support software for use with Quantum computers
to carry out research and investigate how to solve problems using near term Quantum computing power having
short depth circuits. This library uses [QISKit](https://www.qiskit.org/) for its Quantum computation.
  
This library has algorithms that may be used to solve problems across different application domains.

# Algorithms

The following algorithms are part of the library:

* [VQE](#vqe): Variational Quantum Eigensolver 
* [VQkE](#vqke): Variational Quantum k-Eigensolver 
* [QPE](#qpe): Quantum Phase Estimation 
* [IQPE](#iqpe): Iterative Quantum Phase Estimation 
* [Dynamics](#dynamics): Quantum Dynamics evolution
* [ExactEigensolver](#exacteigensolver): Classical eigenvalue solver 
* [Grover](#grover): Grover search 
* [SVM_QKernel](#svm_qkernel): Feature Map Classifier 


## VQE

[VQE](https://arxiv.org/abs/1304.3061), the Variational Quantum Eigensolver algorithm, as its name suggests, uses a variational approach to find the minimum eigenvalue of a Hamiltonian energy problem. It is configured with a trial wavefunction, supplied by a [variational form](./utils/variational_forms), and an [optimizer](./utils/optimizers). An
[initial state](./utils/initial_states) may be supplied too.    


VQE can be configured with the following parameters:

* `operator_mode`=**matrix** | paulis | group_paulis

  Mode used by the operator.py class for the computation
  
* `initial_point`=*optional array of numbers*

  An optional array of numbers may be provided as the starting point for the
  [variational form](./utils/variational_forms). The length of this array must match the variational form being used.
  
  If not provided VQE will create a random starting point for the optimizer where its values are randomly chosen to
  lie within the bounds of the variational form. If the variational form provides None back for any or all elements
  of its bounds then VQE will default that aspect of the bound to the range -2pi to 2pi and the corresponding value
  for the point will be randomly generated in this default range.  


## VQkE

VQkE, the Variational Quantum k-Eigensolver algorithm, is a variant of [VQE](#vqe) that, in addition to finding the
minimum eignevalue, can also find the next k-1 lowest eigenvalues. Like VQE, it is configured with a trial wavefunction,
supplied by a [variational form](./utils/variational_forms), and an [optimizer](./utils/optimizers).
An [initial state](./utils/initial_states) may be supplied too.    

VQkE can be configured with the following parameters:

* See [VQE](#vqe) for common parameter settings

* `k`=*integer 1 to n*

  Returns up to k smallest eigenvalues. k defaults to 1, in which case VQkE coincides with VQE.

* `eigval_bounds`=optional array of pairs

  A list of k pairs of floats indicating lower and upper bounds for the desired eigenvalues.

* `penalty_factor`=*number default 100*

  A penalty factor for condition number of the eigenvalue in the objective function of the
  metalevel search problem, and for violating the bounds on the eigenvalue.

* `norm_threshold`=*number default 1e-04*

  Minimum value of `|<psi|H^2|psi> - <psi|H|psi>^2|` before we start penalizing it in the objective function

* `gap_lb`=*number default 0.1*

  Lower bound on gap between eigenvalues.


## Dynamics
Dynamics provides the lower-level building blocks for simulating universal quantum systems.
For any given quantum system that can be decomposed into local interactions
(e.g., a global hamiltonian *H* as the weighted sum of several pauli spin operators),
the local interactions can then be used to approximate the global quantum system via,
for example, Lloyd's method, or Trotter-Suzuki decomposition.

Note: this algorithm **_only_** supports the `local_state_vector` simulator.

Dynamics can be configured with the following parameter settings:

* `evo_time`=*number, default 1*

  Evolution time, defaults tp 1.0
   
* `evo_mode`=matrix | **circuit**

  Evolution mode of computation, matrix or circuit

* `num_time_slices`=*integer, default 1*

  Number of time slices: non-negative integer

* `paulis_grouping`=**default** | random

  When *default* paulis are grouped

* `expansion_mode`=naive \ **trotter**

  Expansion mode: *naive* (Lloyd's method) or *trotter* for Trotter-Suzuki expansion

* `expansion_order`=*integer, default 2* 

  Trotter-Suzuki expansion order: positive integer


## QPE

QPE, the Quantum Phase Estimation algorithm (also sometimes abbreviated as PEA),
takes two quantum registers, *control* and *target*,
where the control consists of several qubits initially put in uniform superposition
and the target a set of qubits prepared in an eigenstate
(or, oftentimes, an guess of the eigenstate)
of the unitary operator of a quantum system.
QPE then evolves the target under the control using [Dynamics](#dynamics) of the unitary operator.
The information of the corresponding eigenvalue is then *kicked-back* into the phases of the control register,
which can then be deconvoluted by the inverse Quantum Fourier Transform,
and then measured for read-out in binary decimal format.

QPE is configured with an [initial state](./utils/initial_states) 
and an [inverse quantum fourier transform](./utils/iqfts)

Note: this algorithm **_does not_** support the `local_state_vector` simulator.

QPE is also configured with the following parameter settings:

* `num_time_slices`=*integer, default 1*

  Number of time slices: non-negative integer

* `paulis_grouping`=**default** | random

  When *default* paulis are grouped

* `expansion_mode`=naive | **trotter**

  Expansion mode: *naive* (Lloyd's method) or *trotter* for Trotter-Suzuki expansion

* `expansion_order`=*integer, default 2* 

  Trotter-Suzuki expansion order: positive integer

* `num_ancillae`=*integer, default 1*

  Number of ancillae bits


## IQPE

IQPE, the Iterative Quantum Phase Estimation algorithm, as its name suggests, iteratively computes the
phase so as to require less qubits. It takes in the same set of parameters as QPE except for the number of ancillary qubits `num_ancillae`, which is replaced by `num_iterations`. Also, the inverse quantum fourier transform isn't used for IQPE.

For more detail, please see [https://arxiv.org/abs/quant-ph/0610214](https://arxiv.org/abs/quant-ph/0610214).


## ExactEigensolver

While this algorithm does not use a quantum computer, and relies on a purely classical approach to find eigenvalues,
it may be useful in the near term while experimenting with, developing and testing quantum algorithms.

ExactEigensolver can be configured with the following parameter:

* `k`=*integer 1 to n*

  Returns up to k smallest eigenvalues. k defaults to 1.


## Grover

Grover's Search is a well known quantum algorithm for searching through unstructured collection of records for particular targets with quadratic speedups. All that's needed for carrying out a search is an oracle for specifying the search criterion, which basically indicates a hit or miss for any given record. Currently the SAT (satisfiability) oracle implementation is provided, which takes as input a SAT problem in [DIMACS CNF format](http://www.satcompetition.org/2009/format-benchmarks2009.html) and constructs the corresponding quantum circuit.


## SVM_QKernel

SVM QKernel is a feature map classification algorithm....

 --- Todo add more description and parameters ---


# Developers  

Algorithms and some of the utils objects have been designed to be pluggable. A new object may be developed according to
the pattern that will be described and by simply adding the code to set of existing code it will be immediately 
recognized and be made available for use within the framework.

To develop/deploy here any new algorithms the new classes should be under their own folder here in *algorithms*, like
the existing algorithms, vqe, vqke etc., and should derive from QuantumAlgorithm class. 

The [utils](./utils) folder here has common utility classes and other pluggable entities that may be used by the
algorithms

* [Optimizers](./utils/optimizers) 

  Optimizers should go under *utils/optimizers* and derive from Optimizer class.

* [Variational Forms](./utils/variational_forms)
 
  Trial wavefunction objects should go under *utils/variational_forms* and derive from VariationalForm class.

* [Initial States](./utils/initial_states)
 
  Initial state objects should go under *utils/initial_states* and derive from InitialState class.

* [IQFTs](./utils/iqfts)
 
  IQFT (Inverse Quantum Fourier Transform) objects should go under *utils/iqfts* and derive from IQFT class.


All the above classes above should have a configuration dictionary with "name", "description" and "input_schema" 
properties.

You can follow the implementations already in the repo.
 
