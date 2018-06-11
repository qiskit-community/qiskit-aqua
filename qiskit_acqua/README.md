# QISKit ACQUA

QISKit Algorithms and Circuits for QUantum Applications (QISKit ACQUA) is a set of algorithms and utilities
for use with quantum computers to carry out research and investigate how to solve problems using near-term
quantum computing power having short depth circuits. This library uses [QISKit](https://www.qiskit.org/) for 
its quantum computation.
  
This library has algorithms that may be used to solve problems across different application domains.

_**Note**: This library has also some classical algorithms that may be useful in the near term while
experimenting with, developing and testing quantum algorithms to compare and contrast results._

Links to Sections:

* [Algorithms](#algorithms)
* [Additional Configuration](#additional-configuration)
* [Developers](#developers)


# Algorithms

The following quantum algorithms are part of the library:

* [VQE](#vqe): Variational Quantum Eigensolver 
* [VQkE](#vqke): Variational Quantum k-Eigensolver 
* [QPE](#qpe): Quantum Phase Estimation 
* [IQPE](#iqpe): Iterative Quantum Phase Estimation 
* [Dynamics](#dynamics): Quantum Dynamics evolution
* [Grover](#grover): Quantum Grover search 
* [SVM_QKernel](#svm_qkernel): Quantum feature-map classifier via direct estimation of the kernel
* [SVM_Variational](#svm_variational): Variational Quantum feature-map classifier

and these are classical:

* [ExactEigensolver](#exacteigensolver): Classical eigenvalue solver 
* [CPLEX](#cplex): Optimization solver for Ising modelled problems  
* [SVM_RBF_Kernel](#svm_rbf_kernel): RBF SVM algorithm  

The [Additional Configuration](#additional-configuration) section has overall problem and quantum backend
configuration that will be needed when running an algorithm. 

## VQE

[VQE](https://arxiv.org/abs/1304.3061), the Variational Quantum Eigensolver algorithm, as its name suggests, 
uses a variational approach to find the minimum eigenvalue of a Hamiltonian energy problem. It is configured 
with a trial wavefunction, supplied by a [variational form](./utils/variational_forms/README.md), and
an [optimizer](./utils/optimizers/README.md). An [initial state](./utils/initial_states/README.md) may be supplied too.    


VQE supports the problems: `energy` and `ising`

VQE can be configured with the following parameters:

* `operator_mode`=**matrix** | paulis | group_paulis

  Mode used by the operator.py class for the computation
  
* `initial_point`=*optional array of numbers*

  An optional array of numbers may be provided as the starting point for the
  [variational form](./utils/variational_forms/README.md). The length of this array must match the variational
  form being used.
  
  If not provided VQE will create a random starting point for the optimizer where its values are randomly chosen to
  lie within the bounds of the variational form. If the variational form provides None back for any or all elements
  of its bounds then VQE will default that aspect of the bound to the range -2pi to 2pi and the corresponding value
  for the point will be randomly generated in this default range.  


## VQkE

VQkE, the Variational Quantum k-Eigensolver algorithm, is a variant of [VQE](#vqe) that, in addition to finding the
minimum eignevalue, can also find the next k-1 lowest eigenvalues. Like VQE, it is configured with a trial wavefunction,
supplied by a [variational form](./utils/variational_forms/README.md), and an [optimizer](./utils/optimizers/README.md).
An [initial state](./utils/initial_states/README.md) may be supplied too.    

VQkE supports the problems: `energy`, `excited_states` and `ising`

VQkE can be configured with the following parameters:

* See [VQE](#vqe) for common parameter settings

* `k`=*integer 1 to n, default 1*

  Returns up to k smallest eigenvalues. k defaults to 1, in which case VQkE coincides with VQE.

* `eigval_bounds`=optional array of pairs

  A list of k pairs of floats indicating lower and upper bounds for the desired eigenvalues.

* `penalty_factor`=*number, default 100*

  A penalty factor for condition number of the eigenvalue in the objective function of the
  metalevel search problem, and for violating the bounds on the eigenvalue.

* `norm_threshold`=*number, default 1e-04*

  Minimum value of `|<psi|H^2|psi> - <psi|H|psi>^2|` before we start penalizing it in the objective function

* `gap_lb`=*number, default 0.1*

  Lower bound on gap between eigenvalues.


## Dynamics

Dynamics provides the lower-level building blocks for simulating universal quantum systems.
For any given quantum system that can be decomposed into local interactions
(e.g., a global hamiltonian *H* as the weighted sum of several pauli spin operators),
the local interactions can then be used to approximate the global quantum system via,
for example, Lloyd's method, or Trotter-Suzuki decomposition.

Note: this algorithm **_only_** supports the `local_state_vector` simulator.

Dynamics supports the problems: `dynamics`

Dynamics can be configured with the following parameter settings:

* `evo_time`=*number, default 1*

  Evolution time, defaults tp 1.0
   
* `evo_mode`=matrix | **circuit**

  Evolution mode of computation, matrix or circuit

* `num_time_slices`=*integer, default 1*

  Number of time slices: non-negative integer

* `paulis_grouping`=**default** | random

  When *default* paulis are grouped

* `expansion_mode`= **trotter** | suzuki

  Expansion mode: *trotter* (Lloyd's method) or *suzuki* for Trotter-Suzuki expansion

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

QPE is configured with an [initial state](./utils/initial_states/README.md) 
and an [inverse quantum fourier transform](./utils/iqfts/README.md)

Note: this algorithm **_does not_** support the `local_state_vector` simulator.

QPE supports the problems: `energy`

QPE is also configured with the following parameter settings:

* `num_time_slices`=*integer, default 1*

  Number of time slices: non-negative integer

* `paulis_grouping`=**default** | random

  When *default* paulis are grouped

* `expansion_mode`=**trotter** | suzuki

  Expansion mode: *trotter* (Lloyd's method) or *suzuki* for Trotter-Suzuki expansion

* `expansion_order`=*integer, default 2* 

  Trotter-Suzuki expansion order: positive integer

* `num_ancillae`=*integer, default 1*

  Number of ancillae bits


## IQPE

IQPE, the Iterative Quantum Phase Estimation algorithm, as its name suggests, iteratively computes the
phase so as to require less qubits. It takes in the same set of parameters as QPE except for the number
of ancillary qubits `num_ancillae`, which is replaced by `num_iterations`. Also, an inverse quantum fourier
transform isn't used for IQPE.

For more detail, please see [https://arxiv.org/abs/quant-ph/0610214](https://arxiv.org/abs/quant-ph/0610214).

Note: this algorithm **_does not_** support the `local_state_vector` simulator.

IQPE supports the problems: `energy`


## Grover

Grover's Search is a well known quantum algorithm for searching through unstructured collection of records for
particular targets with quadratic speedups. All that's needed for carrying out a search is an oracle for specifying 
the search criterion, which basically indicates a hit or miss for any given record. Currently the SAT (satisfiability)
oracle implementation is provided, which takes as input a SAT problem in 
[DIMACS CNF format](http://www.satcompetition.org/2009/format-benchmarks2009.html) and constructs the corresponding 
quantum circuit.

Note: this algorithm **_does not_** support the `local_state_vector` simulator.

Grover supports the problems: `search`

## SVM_QKernel

Classification algorithms and methods for machine learning are essential for pattern recognition and data mining applications. 
Well known techniques, such as support vector machines or neural networks, have blossomed over the last two decades as a 
result of the spectacular advances in classical hardware computational capabilities and speed. This progress in computer power
made it possible to apply techniques theoretically developed towards the middle of the XX century on classification problems 
that soon became increasingly challenging.

A key concept in classification methods is that of a kernel. Data cannot typically be separated by a hyperplane in its 
original space. A common technique used to find such a hyperplane consists on applying a non-linear transformation function to 
the data. This function is called a _feature map_, as it transforms the raw features, or measurable properties, of the 
phenomenon or subject under study. Classifying in this new feature space -- and, as a matter of fact, also in any other space, 
including the raw original one -- is nothing more than seeing how close data points are to each other. This is the same as 
computing the inner product for each pair of data in the set. In fact we do not need to compute the non-linear feature map for 
each datum, but only the inner product of each pair of data points in the new feature space. This collection of inner products 
is called the _kernel_ and it is perfectly possible to have feature maps that are hard to compute but whose kernels are not.

The SVM_QKernel algorithm applies to classification problems that require a feature map for which computing the kernel 
is not efficient classically.  This means that the required computational resources are expected to scale exponentially with 
the size of the problem. SVM_QKernel uses a Quantum processor to solve this problem by a direct estimation of the kernel in 
the feature space. The method used falls in the category of what is called _supervised learning_, consisting of a _training 
phase_ (where the kernel is calculated and the support vectors obtained) and a _test or classification phase_ (where new 
unlabelled data is classified according to the solution found in the training phase).

For more detail, please see [https://arxiv.org/abs/1804.11326](https://arxiv.org/abs/1804.11326).

Note: this algorithm **_does not_** support the `local_state_vector` simulator.

SVM_QKernel supports the problems: `svm_classification`

SVM_QKernel can be configured with the following parameters:

* `num_qubits`=*integer 2 to n, defaults to 2*

  Number of qubits required is equal to the number of features used for classification.

* `print_info`=**False** | True

  Whether to print additional information or not when the algorithm is running.


## SVM_Variational

Just like SVM_Kernel, the SVM_Variational algorithm applies to classification problems that require a feature map for which 
computing the kernel is not efficient classically.  SVM_Variational solves such problems in a quantum processor by 
variational method that optimizes a parameterized quantum circuit to provide a solution that cleanly separates the data.

For more detail, please see [https://arxiv.org/abs/1804.11326](https://arxiv.org/abs/1804.11326).

Note: this algorithm **_does not_** support the `local_state_vector` simulator.

SVM_Variational supports the problems: `svm_classification`

SVM_Variational can be configured with the following parameters:

* `num_qubits`=*integer 2 to n, defaults to 2*

  Number of qubits required is equal to the number of features used for classification.

* `circuit_depth`=*integer 3 to n, defaults to 3*

  Depth of variational circuit that is optimized. 

* `print_info`=**False** | True

  Whether to print additional information or not when the algorithm is running.


## ExactEigensolver

While this algorithm does not use a quantum computer, and relies on a purely classical approach to find eigenvalues,
it may be useful in the near term while experimenting with, developing and testing quantum algorithms.

ExactEigensolver supports the problems: `'energy`, `excited_states` and `ising`

ExactEigensolver can be configured with the following parameter:

* `k`=*integer 1 to n, default 1*

  Returns up to k smallest eigenvalues. k defaults to 1.


## CPLEX

While this algorithm does not use a quantum computer, and relies on a purely classical approach,
it may be useful in the near term while experimenting with, developing and testing quantum algorithms. This algorithm
uses the [IBM ILOG CPLEX Optimization Studio](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.studio.help/Optimization_Studio/topics/COS_home.html)
which should be installed, and its 
[Python API](https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html)
setup, for this algorithm to be operational. This algorithm currently supports computing the energy of an Ising model
Hamiltonian. 

CPLEX supports the problems: `ising`

CPLEX can be configured with the following parameter:

* `timelimit`=*integer 1 to n, default 600*

  Time limit, defaults to 600.

* `thread`=*integer, default 1*

  Thread, defaults to 1.

* `display`=*integer, default 2*

  Display, defaults to 2.


## SVM_RBF_Kernel

This is uses classical approach to solving feature map classification problem. It may be useful in the near term while
experimenting with, developing and testing quantum algorithms to compare and contrast results.

SVM_RBF_Kernel supports the problems: `svm_classification`

SVM_RBF_Kernel can be configured with the following parameters:

* `print_info`=**False** | True

  Whether to print additional information or not when the algorithm is running.

# Additional configuration

To run an algorithm a [problem](#problem) configuration is needed and for quantum algorithms a [backend](#backend)

## PROBLEM

PROBLEM is an optional section that includes the overall problem being solved and overall problem level configuration

* `name`=**energy** | excited_states | ising | dynamics | search | svm_classification

  Specifies the problem being solved. Ensures that algorithms that can handle this class of problem are used.
 
* `random_seed`=*An integer, default None*  

  Aspects of the computation may include use of random numbers. For instance VQE will often use a random initial
  point if the variation form does not supply any preference based on the initial state (and not overridden by a user
  supplied initial point). In this case each run of VQE, for an otherwise a constant problem, can result in a different
  result. And even if the final value might be the same the number of evaluations may differ. To enable repeatable
  experiments, with the exact same outcome, an integer random seed can be set so as the (pseudo-)random numbers will
  be generated the same each time the experiment is run.


## BACKEND   

BACKEND is an optional section that includes naming the [QISKit](https://www.qiskit.org/) quantum computational
backend to be used for the quantum computation. This defaults to a local quantum simulator backend. 

* `name`=*'qiskit backend name'* 

  Defaults to 'local_statevector_simulator' but any suitable quantum backend can be selected. The QConfig.py file
  may need to be setup for QISKit to access remote devices.
  See [QISKit installation](https://qiskit.org/documentation/install.html#installation) for information on how to
  configure the QConfig.py
  
* `shots`=*integer defaults to 1024*

  With a backend such as local_qasm_simulator, or a real device, this is number of repetitions of each circuit
  for sampling to be used.

* `skip_transpiler`=**false** | true

  Skip circuit translation phase. If the algorithm uses only basis gates directly supported then no translation of
  the circuit into basis gates is required. Skipping the translation may improve overall performance a little
  especially when many circuits are used repeatedly such as is teh case with the VQE algorithm.
   
  *Note: use with caution as if the algorithm does not restrict itself to the set of basis gates supported by the
   backend then the circuit (algorithm) will fail to run.*     

* `noise_params`=*dictionary of noise control key/values, optional, defaults to None*

   When a local simulator is used an optional dictionary can be supplied to control its noise model. For more
   information see 
   [Noise Parameters](https://github.com/QISKit/qiskit-sdk-py/tree/master/src/qasm-simulator-cpp#noise-parameters)
   The following is an example of such a dictionary that can be used:
   
   ```
   "noise_params": {"U": {"p_depol": 0.001,
                          "p_pauli": [0, 0, 0.01],
                          "gate_time": 1,
                          "U_error": [ [[1, 0], [0, 0]],
                                       [[0, 0], [0.995004165, 0.099833417]]
                                     ]
                         }
                   }
   ```

# Developers  

Algorithms and many of the utils objects have been designed to be pluggable. A new object may be developed according to
the pattern that will be described and by simply adding the new code to set of existing code it will be immediately 
recognized and be made available for use within the framework of QISKit ACQUA.

To develop/deploy here any new algorithms the new class and module(s) should be under their own folder here in
*qiskit_acqua*, like the existing algorithms, vqe, vqke etc., and should derive from QuantumAlgorithm class. 

The [utils](./utils/README.md) folder here has common utility classes and other pluggable entities that may be used by the
algorithms

* [Optimizers](./utils/optimizers/README.md) 

  Optimizers should go under *utils/optimizers* and derive from Optimizer class.

* [Variational Forms](./utils/variational_forms/README.md)
 
  Trial wavefunction objects should go under *utils/variational_forms* and derive from VariationalForm class.

* [Initial States](./utils/initial_states/README.md)
 
  Initial state objects should go under *utils/initial_states* and derive from InitialState class.

* [IQFTs](./utils/iqfts/README.md)
 
  IQFT (Inverse Quantum Fourier Transform) objects should go under *utils/iqfts* and derive from IQFT class.

* [Oracles](./utils/oracles/README.md)

  Oracles, for use with algorithms like Grover, should go under *utils/oracles* and derive from Oracle class.  

All the above classes above should have a configuration dictionary with "name", "description" and "input_schema" 
properties.

You can follow the implementations already in the repository here.
 
