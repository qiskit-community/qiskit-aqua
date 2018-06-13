Algorithms
==========

QISKit Algorithms and Circuits for QUantum Applications (QISKit ACQUA)
is an extensible collection of algorithms and utilities for use with quantum computers to
carry out research and investigate how to solve problems using near-term
quantum applications on short depth circuits. The applications can span
different domains  QISKit ACQUA uses
`QISKit <https://www.qiskit.org/>`__ for its quantum computation.

The following :ref:`quantum algorithms` are part of QISKit ACQUA:

-  :ref:`Variational Quantum Eigensolver (VQE)`
-  :ref:`Quantum Dynamics`
-  :ref:`Quantum Phase Estimation (QPE)`
-  :ref:`Iterative Quantum Phase Estimation (IQPE)`
-  :ref:`Quantum Grover Search`
-  :ref:`Support Vector Machine Quantum Kernel (SVM Q Kernel)`
-  :ref:`Support Vector Machine Variational (SVM Variational)`

QISKit ACQUA includes  also some :ref:`classical algorithms`, which may be
useful to compare and contrast results in the near term while experimenting with, developing and testing
quantum algorithms:

-  :ref:`Exact Eigensolver`
-  :ref:`CPLEX`
-  :ref:`Support Vector Machine Radial Basis Function Kernel (SVM RBF Kernel)`

.. topic:: Extending the Algorithm Library

    Algorithms and many of the components they used have been designed to be
    pluggable. A new algorithm may be developed according to the specific API
    provided by QISKit ACQUA, and by simply adding its code to the collection of existing
    algorithms, that new algorithm  will be immediately recognized via dynamic lookup, and made available for use
    within the framework of QISKit ACQUA.

    To develop and deploy any new algorithm, the new algorithm class should derive from the ``QuantumAlgorithm`` class.
    Along with any supporting  module, the new algorithm class
    should be installed under its own folder in the ``qiskit_acqua`` directory, just like  the
    existing algorithms.



Quantum Algorithms
------------------
In this section, we describe the quantum algorithms currently available in QISKit ACQUA.

.. note::
    QISKit ACQUA requires associating a quantum device or simulator to any experiment that uses a quantum
    algorithm.  This is done by configuring the ``backend`` section of the experiment to be run.

Variational Quantum Eigensolver (VQE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

`VQE <https://arxiv.org/abs/1304.3061>`__ uses a variational approach
to find the minimum eigenvalue of a Hamiltonian energy problem. It is
configured with a trial wavefunction, supplied by a `variational
form <./variational_forms.html>`__, and an
`optimizer <./optimizers.html>`__. An `initial
state <./initial_states.html>`__ may be supplied too.

Additionally, VQE can be configured with the following parameters:

-  A ``string`` indicating the mode used by the ``Operator`` class for the computation:

   .. code:: python

       operator_mode : "matrix" | "paulis" | "group_paulis"

   If no value for ``operator_mode`` ia specified, the default is ``"matrix"``. 

-  The initial point for the search of the minimum eigenvalue:

   .. code:: python

       initial_point : [float, float, ... , float]

   An optional list of ``float`` values  may be provided as the starting point
   for the variational form.
   The length of this list must match the number of the parameters expected by the variational form being used.
   If such list is not provided, VQE will create a random starting point for the
   optimizer, with values randomly chosen to lie within the
   bounds of the variational form. If the variational form provides no lower bound, the VQE
   will default it to :math:`-2\pi`; if the upper bound is missing, the default value is :math:`2\pi`.


.. topic:: Declarative Name

   When referring to VQE declaratively inside QISKit ACQUA, its code ``name``, by which QISKit ACQUA dynamically discovers and loads it,
   is ``VQE``.

.. topic:: Problems Supported

   In QISKit ACQUA, VQE supports the ``energy`` and ``ising`` problema.

Quantum Dynamics
~~~~~~~~~~~~~~~~

Dynamics provides the lower-level building blocks for simulating
universal quantum systems. For any given quantum system that can be
decomposed into local interactions (for example, a global hamiltonian *H* as
the weighted sum of several pauli spin operators), the local
interactions can then be used to approximate the global quantum system
via, for example, Lloyd’s method or Trotter-Suzuki decomposition.

.. note::
    This algorithm **only** supports the ``local_state_vector`` simulator.

Dynamics can be configured with the following parameter settings:

-  Evolution time:

   .. code:: python
   
       evo_time : float

   A number is expected.  The minimum value is ``0.0``.  The default value is ``1.0``.

-  The evolution mode of the computation:

   .. code:: python 

       evo_mode = "matrix" | "circuit"

   Two ``string`` values are permitted: ``"matrix"`` or ``"circuit"``, with ``"circuit"`` being the default.

-  The number of time slices: 

   .. code:: python

       num_time_slices = 0 | 1 | ...
   
   This has to be a non-negative ``int`` value.  The default is ``1``.

-  Paulis grouping mode:

   .. code:: python

       paulis_grouping = "default" | "random"

   Two ``string`` values are permitted: ``"default"`` or ``"random"``, with ``"default"`` being the default and indicating
   that the paulis should be grouped.

-  The expansion mode: 

   .. code:: python

       expansion_mode = "trotter" | "suzuki"

   Two ``string`` values are permitted: ``"trotter"`` (Lloyd's method) or ``"suzuki"`` (for Trotter-Suzuki expansion),
   with  ``"trotter"`` being the default one.

-  The expansion order:

   .. code:: python

       expansion_order = 1 | 2 | ...

   This parameter sets the Trotter-Suzuki expansion order.  A positive ``int`` value is expected.  The default value is ``2``.

.. topic:: Declarative Name

   When referring to Quantum Dynamics declaratively inside QISKit ACQUA, its code ``name``, by which
   QISKit ACQUA dynamically discovers and loads it, is ``Dynamics``.

.. topic:: Problems Supported

   In QISKit ACQUA, Quantum Dynamics supports the ``dynamics`` problem.


Quantum Phase Estimation (QPE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

QPE (also sometimes abbreviated
as PEA, for **Phase Estimation Algorithm**), takes two quantum registers, **control** and **target**, where the
control consists of several qubits initially put in uniform
superposition, and the target a set of qubits prepared in an eigenstate
(or, oftentimes, a guess of the eigenstate) of the unitary operator of
a quantum system. QPE then evolves the target under the control using
:ref:`Dynamics` of the unitary operator. The information of the
corresponding eigenvalue is then *kicked-back* into the phases of the
control register, which can then be deconvoluted by an `Inverse Quantum
Fourier Transform (IQFT) <./iqfts.html>`__, and measured for read-out in binary decimal
format.

.. note::
    This algorithm **does not** support the ``local_state_vector`` simulator.

QPE is configured with an `initial
state <initial_states.html>`__ and an `IQFT <./iqfts.html>`__.

QPE is also configured with the following parameter settings:

-  The number of time slices: 

   .. code:: python

       num_time_slices = 0 | 1 | ...

   This has to be a non-negative ``int`` value.  The default value is ``1``.

-  Paulis grouping mode:

   .. code:: python

       paulis_grouping = "default" | "random"

   Two string values are permitted: ``"default"`` or ``"random"``, with ``"default"``
   being the default and indicating that the paulis should be grouped.

-  The expansion mode:

   .. code:: python

       expansion_mode = "trotter" | "suzuki"

   Two ``string`` values are permitted: ``"trotter"`` (Lloyd's method) or ``"suzuki"`` (for Trotter-Suzuki expansion),
   with  ``"trotter"`` being the default one.

-  The expansion order:

   .. code:: python

       expansion_order = 1 | 2 | ...

   This parameter sets the Trotter-Suzuki expansion order.  A positive ``int`` value is expected.  The default value is ``2``.

-  The number of ancillae:

   .. code:: python 

       num_ancillae = 1 | 2 | ...

   This parameter sets the number of ancillary qubits to be used by QPE.  A positive ``int`` value is expected.
   The default value is ``1``.

.. topic:: Declarative Name

   When referring to QPE declaratively inside QISKit ACQUA, its code ``name``, by which
   QISKit ACQUA dynamically discovers and loads it, is ``QPE``.

.. topic:: Problems Supported

   In QISKit ACQUA, QPE supports the ``energy`` problem.

Iterative Quantum Phase Estimation (IQPE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

IQPE, as its name
suggests, iteratively computes the phase so as to require less qubits.
It takes in the same set of parameters as :ref:`Quantum Phase Estimation (QPE)`, except for the number of
ancillary qubits ``num_ancillae``, which is replaced by
``num_iterations`` (a positive ``int``, also defaulted to ``1``), and for the fact that an `IQFT <./iqfts.html>`__ is not
used for IQPE.

.. note::
    This algorithm **does not** support the ``local_state_vector`` simulator.

For more details, please see `arXiv:quant-ph/0610214 <https://arxiv.org/abs/quant-ph/0610214>`__.

.. topic:: Declarative Name

   When referring to IQPE declaratively inside QISKit ACQUA, its code ``name``, by which
   QISKit ACQUA dynamically discovers and loads it, is ``IQPE``.

.. topic:: Problems Supported

   In QISKit ACQUA, IQPE supports the ``energy`` problem.


Quantum Grover Search
~~~~~~~~~~~~~~~~~~~~~

Grover’s Search is a well known quantum algorithm for searching through
unstructured collection of records for particular targets with quadratic
speedups. All that is needed for carrying out a search is an `oracle <./oracles.html>`__ for
specifying the search criterion, which basically indicates a hit or miss
for any given record. Currently the satisfiability (SAT) oracle
implementation is provided, which takes as input a SAT problem in
`DIMACS CNF
format <http://www.satcompetition.org/2009/format-benchmarks2009.html>`__
and constructs the corresponding quantum circuit.

.. topic:: Declarative Name

   When referring to Quantum Grover Search declaratively inside QISKit ACQUA, its code ``name``, by which
   QISKit ACQUA dynamically discovers and loads it, is ``Grover``.

.. topic:: Problems Supported

   In QISKit ACQUA, Grover supports the ``search`` problem.


Support Vector Machine Quantum Kernel (SVM Q Kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Classification algorithms and methods for machine learning are essential
for pattern recognition and data mining applications. Well known
techniques, such as support vector machines or neural networks, have
blossomed over the last two decades as a result of the spectacular
advances in classical hardware computational capabilities and speed.
This progress in computer power made it possible to apply techniques
theoretically developed towards the middle of the XX century on
classification problems that soon became increasingly challenging.

A key concept in classification methods is that of a kernel. Data cannot
typically be separated by a hyperplane in its original space. A common
technique used to find such a hyperplane consists on applying a
non-linear transformation function to the data. This function is called
a *feature map*, as it transforms the raw features, or measurable
properties, of the phenomenon or subject under study. Classifying in
this new feature space – and, as a matter of fact, also in any other
space, including the raw original one – is nothing more than seeing how
close data points are to each other. This is the same as computing the
inner product for each pair of data in the set. In fact we do not need
to compute the non-linear feature map for each datum, but only the inner
product of each pair of data points in the new feature space. This
collection of inner products is called the *kernel* and it is perfectly
possible to have feature maps that are hard to compute but whose kernels
are not.

The SVM Q Kernel algorithm applies to classification problems that
require a feature map for which computing the kernel is not efficient
classically. This means that the required computational resources are
expected to scale exponentially with the size of the problem.
SVM_QKernel uses a Quantum processor to solve this problem by a direct
estimation of the kernel in the feature space. The method used falls in
the category of what is called *supervised learning*, consisting of a
*training phase* (where the kernel is calculated and the support vectors
obtained) and a *test or classification phase* (where new labeless data
is classified according to the solution found in the training phase).

SVM Q Kernel can be configured with the following parameter:

-  A Boolean indicating whether or not to print additional information when the algorithm is running:

   .. code:: python

       print_info : bool

   A Boolean value is expected.  The default is ``False``.

.. topic:: Declarative Name

   When referring to SVM Q Kernel declaratively inside QISKit ACQUA, its code ``name``, by which
   QISKit ACQUA dynamically discovers and loads it, is ``SVM_QKernel``.

.. topic:: Problems Supported

   In QISKit ACQUA, SVM Q Kernel  supports the ``svm_classification`` problem.

Support Vector Machine Variational (SVM Variational)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just like SVM_Kernel, the SVM_Variational algorithm applies to
classification problems that require a feature map for which computing
the kernel is not efficient classically. SVM_Variational solves such
problems in a quantum processor by variational method that optimizes a
parameterized quantum circuit to provide a solution that cleanly
separates the data.

SVM_Variational can be configured with the following parameters:

-  The depth of the variational circuit to be optimized:

   .. code:: python

       circuit_depth = 3 | 4 | ...

   An integer value greater than or equal to ``3`` is expected.  The default is ``3``.

-  A Boolean indicating whether or not to print additional information when the algorithm is running:

   .. code:: python

       print_info : bool

   A Boolean value is expected.  The default is ``False``.

.. topic:: Declarative Name

   When referring to SVM Variational declaratively inside QISKit ACQUA, its code ``name``, by which
   QISKit ACQUA dynamically discovers and loads it, is ``SVM_Variational``.

.. topic:: Problems Supported

   In QISKit ACQUA, SVM Variational  supports the ``svm_classification`` problem.


Classical Algorithms
--------------------
In this section, we describe the classical algorithms currently available in QISKit ACQUA.
While these algorithm do not use a quantum device or simulator, and rely on
purely classical approaches, they may be useful in the
near term while experimenting with, developing and testing quantum
algorithms.

.. note::
    QISKit ACQUA prevents associating a quantum device or simulator to any experiment that uses a classical
    algorithm.  The ``backend`` section of an experiment to be conducted via a classical algorithm is
    disabled.

Exact Eigensolver
~~~~~~~~~~~~~~~~~

Exact Eigensolver computes up to the first ``k`` eigenvalues of a complex square matrix of dimension ``n x n``,
with ``k`` :math:`leq` ``n``.
It can be configured with the following parameter:

-  The number of eigenvalues to compute:

   .. code:: python

       k = 1 | 2 | ... | n

   An ``int`` value ``k`` in the range :math:`[1,n]` is expected. The default is ``1``.

.. topic:: Declarative Name

   When referring to Exact Eigensolver declaratively inside QISKit ACQUA, its code ``name``, by which
   QISKit ACQUA dynamically discovers and loads it, is ``ExactEigensolver``.

.. topic:: Problems Supported

   In QISKit ACQUA, Exact Eigensolver supports the ``energy``, ``ising`` and ``excited_states``  problems.


CPLEX
~~~~~

This algorithm uses the `IBM ILOG CPLEX Optimization
Studio <https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.studio.help/Optimization_Studio/topics/COS_home.html>`__
which should be installed along with its `Python
API <https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html>`__
setup, for this algorithm to be operational. This algorithm currently
supports computing the energy of an Ising model Hamiltonian.

CPLEX can be configured with the following parameters:

-  A time limit in seconds for the execution:

   .. code:: python

       timelimit = 1 | 2 | ...

   A positive ``int`` val;ue is expected.  The default value is `600`.

-  The number of threads that CPLEX uses:

   .. code:: python

       thread = 0 | 1 | 2 | ...

   A non-negative ``int`` value is expected. Setting ``thread`` to ``0`` lets CPLEX decide the number of threads to allocate, but this may
   not be ideal for small problems.  Any value
   greater than ``0`` specifically sets the thread count.  The default value is ``1``, which is ideal for small problems.

-  Decides what CPLEX reports to the screen and records in a log during mixed integer optimization (MIP).

   .. code:: python

       display = 0 | 1 | 2 | 3 | 4 | 5

   An ``int`` value between ``0`` and ``5`` is expected.
   The amount of information displayed increases with increasing values of this parameter.
   By default, this value is set to ``2``.

.. topic:: Declarative Name

   When referring to CPLEX declaratively inside QISKit ACQUA, its code ``name``, by which
   QISKit ACQUA dynamically discovers and loads it, is ``CPLEX``.

.. topic:: Problems Supported

   In QISKit ACQUA, CPLEX supports the ``ising`` problem.


Support Vector Machine Radial Basis Function Kernel (SVM RBF Kernel)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This algorithm uses a classical approach to experiment with feature map classification
problems.
SVM RBF Kernel can be configured with the following parameter:

-  A Boolean indicating whether or not to print additional information when the algorithm is running:

   .. code:: python

       print_info : bool

   A Boolean value is expected.  The default is ``False``.

.. topic:: Declarative Name

   When referring to SVM RBF Kernel declaratively inside QISKit ACQUA, its code ``name``, by which
   QISKit ACQUA dynamically discovers and loads it, is ``SVM_RBF_Kernel``.

.. topic:: Problems Supported

   In QISKit ACQUA, SVM RBF Kernel  supports the ``svm_classification`` problem.



