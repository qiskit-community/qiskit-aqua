.. _algorithms:

==========
Algorithms
==========

Aqua is an extensible collection of algorithms and utilities for use with quantum computers to
carry out research and investigate how to solve problems using near-term
quantum applications on short depth circuits. The applications can span
different domains. Aqua uses
`Terra <https://www.qiskit.org/terra>`__ for the generation, compilation and execution
of the quantum circuits modeling the specific problems.

The following `quantum algorithms <#quantum-algorithms>`__ are part of Aqua:

-  :ref:`Variational Quantum Eigensolver (VQE)`
-  :ref:`Quantum Approximate Optimization Algorithm (QAOA)`
-  :ref:`Quantum Dynamics`
-  :ref:`Quantum Phase Estimation (QPE)`
-  :ref:`Iterative Quantum Phase Estimation (IQPE)`
-  :ref:`Quantum Grover Search`
-  :ref:`Support Vector Machine Quantum Kernel (SVM Q Kernel)`
-  :ref:`Support Vector Machine Variational (SVM Variational)`

Aqua includes  also some `classical algorithms <#classical-reference-algorithms>`__
for generating reference values. This feature of Aqua may be
useful to quantum algorithm researchers interested in generating, comparing and contrasting
results in the near term while experimenting with, developing and testing
quantum algorithms:

-  :ref:`Exact Eigensolver`
-  :ref:`CPLEX`
-  :ref:`Support Vector Machine Radial Basis Function Kernel (SVM RBF Kernel)`

.. topic:: Extending the Algorithm Library

    Algorithms and many of the components they use have been designed to be
    pluggable. A new algorithm may be developed according to the specific Application Programming Interface (API)
    provided by Aqua, and by simply adding its code to the collection of existing
    algorithms, that new algorithm  will be immediately recognized via dynamic lookup,
    and made available for use within the framework of Aqua.
    Specifically, to develop and deploy any new algorithm, the new algorithm class should derive from the ``QuantumAlgorithm`` class.
    Along with any supporting  module, for immediate dynamic discovery, the new algorithm class
    can simply be installed under its own folder in the ``qiskit_aqua`` directory, just like the
    existing algorithms.  Aqua also allows for
    :ref:`aqua-dynamically-discovered-components`: new algorithms can register themselves
    as Aqua extensions and be dynamically discovered at run time independent of their
    location in the file system.
    This is done in order to encourage researchers and
    developers interested in
    :ref:`aqua-extending` to extend the Aqua framework with their novel research contributions.

.. seealso::

    Section :ref:`aqua-extending` provides more
    details on how to extend Aqua with new components.


.. _quantum-algorithms:

------------------
Quantum Algorithms
------------------

In this section, we describe the quantum algorithms currently available in Aqua.

.. note::

    Aqua requires associating a quantum device or simulator to any experiment that uses a quantum
    algorithm.  This is done by configuring the ``"backend"`` section of the experiment to be run.
    Consult the documentation on the :ref:`aqua-input-file` for more details.

.. _vqe:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Variational Quantum Eigensolver (VQE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`VQE <https://arxiv.org/abs/1304.3061>`__ uses a variational approach
to find the minimum eigenvalue of a Hamiltonian energy problem. It is
configured with a trial wave function, supplied by a variational
form and an optimizer. An initial state may be supplied too.

.. seealso::

    Refer to the documentation of :ref:`variational-forms`, :ref:`optimizers`
    and :ref:`initial-states` for more details.

Additionally, VQE can be configured with the following parameters:

-  A ``str`` value indicating the mode used by the ``Operator`` class for the computation:

   .. code:: python

       operator_mode : "matrix" | "paulis" | "grouped_paulis"

   If no value for ``operator_mode`` is specified, the default is ``"matrix"``.

-  The initial point for the search of the minimum eigenvalue:

   .. code:: python

       initial_point : [float, float, ... , float]

   An optional list of ``float`` values  may be provided as the starting point for the search of the minimum eigenvalue.
   This feature is particularly useful when there are reasons to believe that the
   solution point is close to a particular point, which can then be provided as the preferred initial point.  As an example,
   when building the dissociation profile of a molecule, it is likely that
   using the previous computed optimal solution as the starting initial point for the next interatomic distance is going
   to reduce the number of iterations necessary for the variational algorithm to converge.  Aqua provides
   `a tutorial detailing this use case <https://github.com/Qiskit/aqua-tutorials/blob/master/chemistry/h2_vqe_initial_point.ipynb>`__.
    
   The length of the ``initial_point`` list value must match the number of the parameters expected by the variational form being used.
   If the user does not supply a preferred initial point, then VQE will look to the variational form for a preferred value.
   If the variational form returns ``None``,
   then a random point will be generated within the parameter bounds set, as per above.
   If the variational form provides ``None`` as the lower bound, then VQE
   will default it to :math:`-2\pi`; similarly, if the variational form returns ``None`` as the upper bound, the default value will be :math:`2\pi`.


.. topic:: Declarative Name

   When referring to VQE declaratively inside Aqua, its code ``name``, by which Aqua dynamically discovers and loads it,
   is ``VQE``.

.. topic:: Problems Supported

   In Aqua, VQE supports the ``energy`` and ``ising`` problems.

.. _qaoa:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Quantum Approximate Optimization Algorithm (QAOA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`QAOA <https://arxiv.org/abs/1411.4028>`__ is a well-known algorithm for finding approximate solutions to
combinatorial-optimization problems.
The QAOA implementation in Aqua directly uses `VQE <#variational-quantum-eigensolver-vqe>`__ for its general hybrid optimization structure.
However, unlike VQE, which can be configured with arbitrary variational forms,
QAOA uses its own fine-tuned variational form, which comprises :math:`p` parameterized global :math:`x` rotations and 
:math:`p` different parameterizations of the problem hamiltonian.
As a result, unlike VQE, QAOA does not need to have a variational form specified as an input parameter,
and is configured mainly by a single integer parameter, ``p``,
which dictates the depth of the variational form, and thus affects the approximation quality.
Similar to VQE, an optimizer may also be specified.

.. seealso::

    Consult the documentation on :ref:`optimizers` for more details.

In summary, QAOA can be configured with the following parameters:

-  A ``str`` value indicating the mode used by the ``Operator`` class for the computation:

   .. code:: python

       operator_mode : "matrix" | "paulis" | "grouped_paulis"

   If no value for ``operator_mode`` is specified, the default is ``"matrix"``.

-  A positive ``int`` value configuring the QAOA variational form depth, as discussed above:

   .. code:: python

       p = 1 | 2 | ...

   This has to be a positive ``int`` value.  The default is ``1``.

-  The initial point for the search of the minimum eigenvalue:

   .. code:: python

       initial_point : [float, float, ... , float]

   An optional list of :math:`2p` ``float`` values  may be provided as the starting ``beta`` and ``gamma`` parameters
   (as identically named in the original `QAOA paper <https://arxiv.org/abs/1411.4028>`__) for the QAOA variational form.
   If such list is not provided, QAOA will simply start with the all-zero vector.


.. topic:: Declarative Name

   When referring to QAOA declaratively inside Aqua, its code ``name``,
   by which Aqua dynamically discovers and loads it,
   is ``QAOA``.

.. topic:: Problems Supported

   In Aqua, QAOA supports the ``ising`` problem.

.. _dynamics:

^^^^^^^^^^^^^^^^
Quantum Dynamics
^^^^^^^^^^^^^^^^

Dynamics provides the lower-level building blocks for simulating
universal quantum systems. For any given quantum system that can be
decomposed into local interactions (for example, a global hamiltonian as
the weighted sum of several Pauli spin operators), the local
interactions can then be used to approximate the global quantum system
via, for example, Lloyd’s method or Trotter-Suzuki decomposition.

.. warning::

    This algorithm only supports the local state vector simulator.

Dynamics can be configured with the following parameter settings:

-  Evolution time:

   .. code:: python

       evo_time : float

   A ``float`` value is expected.  The minimum value is ``0.0``.  The default value is ``1.0``.

-  The evolution mode of the computation:

   .. code:: python

       evo_mode = "matrix" | "circuit"

   Two ``str`` values are permitted: ``"matrix"`` or ``"circuit"``, with ``"circuit"`` being the default.

-  The number of time slices:

   .. code:: python

       num_time_slices = 0 | 1 | ...

   This has to be a non-negative ``int`` value.  The default is ``1``.

-  Paulis grouping mode:

   .. code:: python

       paulis_grouping = "default" | "random"

   Two ``str`` values are permitted: ``"default"`` or ``"random"``, with ``"default"`` being the default and indicating
   that the Paulis should be grouped.

-  The expansion mode:

   .. code:: python

       expansion_mode = "trotter" | "suzuki"

   Two ``str`` values are permitted: ``"trotter"`` (Lloyd's method) or ``"suzuki"`` (for Trotter-Suzuki expansion),
   with  ``"trotter"`` being the default one.

-  The expansion order:

   .. code:: python

       expansion_order = 1 | 2 | ...

   This parameter sets the Trotter-Suzuki expansion order.  A positive ``int`` value is expected.  The default value is ``2``.

.. topic:: Declarative Name

   When referring to Quantum Dynamics declaratively inside Aqua, its code ``name``, by which
   Aqua dynamically discovers and loads it, is ``Dynamics``.

.. topic:: Problems Supported

   In Aqua, Quantum Dynamics supports the ``dynamics`` problem.

.. _qpe:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Quantum Phase Estimation (QPE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

QPE (also sometimes abbreviated
as PEA, for *Phase Estimation Algorithm*), takes two quantum registers, *control* and *target*, where the
control consists of several qubits initially put in uniform
superposition, and the target a set of qubits prepared in an eigenstate
(or, oftentimes, a guess of the eigenstate) of the unitary operator of
a quantum system. QPE then evolves the target under the control using
:ref:`Dynamics` on the unitary operator. The information of the
corresponding eigenvalue is then *kicked-back* into the phases of the
control register, which can then be deconvoluted by an Inverse Quantum
Fourier Transform (IQFT), and measured for read-out in binary decimal
format.  QPE also requires a reasonably good estimate of the eigen wave function
to start the process. For example, when estimating molecular ground energies,
the :ref:`Hartree-Fock` method could be used to provide such trial eigen wave
functions.

.. seealso::

    Consult the documentation on :ref:`iqfts` and :ref:`initial-states`
    for more details.

.. warning::

    This algorithm does not support the local state vector simulator.

In addition to requiring an IQFT and an initial state as part of its
configuration, QPE also exposes the following parameter settings:

-  The number of time slices:

   .. code:: python

       num_time_slices = 0 | 1 | ...

   This has to be a non-negative ``int`` value.  The default value is ``1``.

-  Paulis grouping mode:

   .. code:: python

       paulis_grouping = "default" | "random"

   Two string values are permitted: ``"default"`` or ``"random"``, with ``"default"``
   being the default and indicating that the Paulis should be grouped.

-  The expansion mode:

   .. code:: python

       expansion_mode = "trotter" | "suzuki"

   Two ``str`` values are permitted: ``"trotter"`` (Lloyd's method) or ``"suzuki"`` (for Trotter-Suzuki expansion),
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

   When referring to QPE declaratively inside Aqua, its code ``name``, by which
   Aqua dynamically discovers and loads it, is ``QPE``.

.. topic:: Problems Supported

   In Aqua, QPE supports the ``energy`` problem.

.. _iqpe:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Iterative Quantum Phase Estimation (IQPE)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IQPE, as its name
suggests, iteratively computes the phase so as to require fewer qubits.
It takes in the same set of parameters as `QPE <#quantum-phase-estimation-qpe>`__, except for the number of
ancillary qubits ``num_ancillae``, which is replaced by
``num_iterations`` (a positive ``int``, also defaulted to ``1``), and for the fact that an
Inverse Quantum Fourier Transform (IQFT) is not used for IQPE.

.. warning::

    This algorithm does not support the local state vector simulator.

.. seealso::

    For more details, please see `arXiv:quant-ph/0610214 <https://arxiv.org/abs/quant-ph/0610214>`__.

.. topic:: Declarative Name

    When referring to IQPE declaratively inside Aqua, its code ``name``, by which
    Aqua dynamically discovers and loads it, is ``IQPE``.

.. topic:: Problems Supported

    In Aqua, IQPE supports the ``energy`` problem.


.. _grover:

^^^^^^^^^^^^^^^^^^^^^
Quantum Grover Search
^^^^^^^^^^^^^^^^^^^^^

Grover’s Search is a well known quantum algorithm for searching through
unstructured collections of records for particular targets with quadratic
speedups.

Given a set :math:`X` of :math:`N` elements
:math:`X=\{x_1,x_2,\ldots,x_N\}` and a boolean function :math:`f : X \rightarrow \{0,1\}`,
the goal on an *unstructured-search problem* is to find an
element :math:`x^* \in X` such that :math:`f(x^*)=1`.
Unstructured  search  is  often  alternatively  formulated  as  a  database  search  problem, in
which, given a database, the goal is to find in it an item that meets some specification.
The search is called *unstructured* because there are no guarantees as to how the
database is ordered.  On a sorted database, for instance, one could perform
binary  search  to  find  an  element in :math:`\mathbb{O}(\log N)` worst-case time.
Instead, in an unstructured-search problem, there is no  prior knowledge about the contents
of the database.  With classical circuits, there is no alternative but
to perform a linear number of queries to find the target element.
Conversely, Grover’s Search algorithm allows to solve the unstructured-search problem
on a quantum computer in :math:`\mathcal{O}(\sqrt{N})` queries. 

All that is needed for carrying out a search is an oracle from Aqua's :ref:`oracles` library for
specifying the search criterion, which basically indicates a hit or miss
for any given record.  More formally, an *oracle* :math:`O_f` is an object implementing a boolean function
:math:`f` as specified above.  Given an input :math:`x \in X`, :math:`O_f` returns :math:`f(x)`.  The
details of how :math:`O_f` works are unimportant; Grover's search algorithm treats an oracle as a black
box.  Currently, Aqua provides the satisfiability (SAT) oracle
implementation, which takes as input an SAT problem in
`DIMACS CNF
format <http://www.satcompetition.org/2009/format-benchmarks2009.html>`__
and constructs the corresponding quantum circuit.  Oracles are treated as pluggable components
in Aqua; researchers interested in :ref:`aqua-extending` can design and implement new
oracles and extend Aqua's oracle library.

.. topic:: Declarative Name

   When referring to Quantum Grover Search declaratively inside Aqua, its code ``name``, by which
   Aqua dynamically discovers and loads it, is ``Grover``.

.. topic:: Problems Supported

   In Aqua, Grover's Search algorithm supports the ``search`` problem.

.. _svm-q-kernel:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Support Vector Machine Quantum Kernel (SVM Q Kernel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

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
SVM Q Kernel uses a Quantum processor to solve this problem by a direct
estimation of the kernel in the feature space. The method used falls in
the category of what is called *supervised learning*, consisting of a
*training phase* (where the kernel is calculated and the support vectors
obtained) and a *test or classification phase* (where new labelless data
is classified according to the solution found in the training phase).

SVM Q Kernel can be configured with a ``bool`` parameter, indicating
whether or not to print additional information when the algorithm is running:

.. code:: python

    print_info : bool

The default is ``False``.

.. topic:: Declarative Name

   When referring to SVM Q Kernel declaratively inside Aqua, its code ``name``, by which
   Aqua dynamically discovers and loads it, is ``SVM_QKernel``.

.. topic:: Problems Supported

   In Aqua, SVM Q Kernel  supports the ``svm_classification`` problem.

.. _svm-variational:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Support Vector Machine Variational (SVM Variational)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Just like SVM Q Kerne, the SVM Variational algorithm applies to
classification problems that require a feature map for which computing
the kernel is not efficient classically. SVM Variational uses the variational method to solve such
problems in a quantum processor.  Specifically, it optimizes a
parameterized quantum circuit to provide a solution that cleanly
separates the data.

SVM Variational can be configured with the following parameters:

-  The depth of the variational circuit to be optimized:

   .. code:: python

       circuit_depth = 3 | 4 | ...

   An integer value greater than or equal to ``3`` is expected.  The default is ``3``.

-  A Boolean indicating whether or not to print additional information when the algorithm is running:

   .. code:: python

       print_info : bool

   A ``bool`` value is expected.  The default is ``False``.

.. topic:: Declarative Name

   When referring to SVM Variational declaratively inside Aqua, its code ``name``, by which
   Aqua dynamically discovers and loads it, is ``SVM_Variational``.

.. topic:: Problems Supported

   In Aqua, SVM Variational  supports the ``svm_classification`` problem.

.. _classical-reference-algorithms:

------------------------------
Classical Reference Algorithms
------------------------------

In this section, we describe the classical algorithms currently available in Aqua.
While these algorithms do not use a quantum device or simulator, and rely on
purely classical approaches, they may be useful in the
near term to generate reference values while experimenting with, developing and testing quantum
algorithms.

.. warning::

    Aqua prevents associating a quantum device or simulator to any experiment that uses a classical
    algorithm.  The ``"backend"`` section of an experiment to be conducted via a classical algorithm is
    disabled.

.. _exact-eigensolver:

^^^^^^^^^^^^^^^^^
Exact Eigensolver
^^^^^^^^^^^^^^^^^

Exact Eigensolver computes up to the first :math:`k` eigenvalues of a complex square matrix of dimension
:math:`n \times n`, with :math:`k \leq n`.
It can be configured with an ``int`` parameter ``k`` indicating the number of eigenvalues to compute:

.. code:: python

    k = 1 | 2 | ... | n

Specifically, the value of this parameter must be an ``int`` value ``k`` in the range :math:`[1,n]`. The default is ``1``.

.. topic:: Declarative Name

   When referring to Exact Eigensolver declaratively inside Aqua, its code ``name``, by which
   Aqua dynamically discovers and loads it, is ``ExactEigensolver``.

.. topic:: Problems Supported

   In Aqua, Exact Eigensolver supports the ``energy``, ``ising`` and ``excited_states``  problems.

.. _cplex:

^^^^^
CPLEX
^^^^^

This algorithm uses the `IBM ILOG CPLEX Optimization
Studio <https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.studio.help/Optimization_Studio/topics/COS_home.html>`__,
which should be installed along with its `Python API
<https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/set_up/Python_setup.html>`__
for this algorithm to be operational. This algorithm currently
supports computing the energy of an Ising model Hamiltonian.

CPLEX can be configured with the following parameters:

-  A time limit in seconds for the execution:

   .. code:: python

       timelimit = 1 | 2 | ...

   A positive ``int`` value is expected.  The default value is `600`.

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

   When referring to CPLEX declaratively inside Aqua, its code ``name``, by which
   Aqua dynamically discovers and loads it, is ``CPLEX``.

.. topic:: Problems Supported

   In Aqua, CPLEX supports the ``ising`` problem.

.. _avm-rbf-kernel:

^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Support Vector Machine Radial Basis Function Kernel (SVM RBF Kernel)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

SVM RBF Kernel uses a classical approach to experiment with feature map classification
problems.
SVM RBF Kernel can be configured with a ``bool`` parameter,
indicating whether or not to print additional information when the algorithm is running:

.. code:: python

    print_info : bool

The default value for this parameter is ``False``.

.. topic:: Declarative Name

   When referring to SVM RBF Kernel declaratively inside Aqua, its code ``name``, by which
   Aqua dynamically discovers and loads it, is ``SVM_RBF_Kernel``.

.. topic:: Problems Supported

   In Aqua, SVM RBF Kernel  supports the ``svm_classification`` problem.
