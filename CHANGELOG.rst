Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_.

  **Types of changes:**

  - **Added**: for new features.
  - **Changed**: for changes in existing functionality.
  - **Deprecated**: for soon-to-be removed features.
  - **Removed**: for now removed features.
  - **Fixed**: for any bug fixes.
  - **Security**: in case of vulnerabilities.


`UNRELEASED`_
=============

Added
-----

- Relative-Phase Toffoli gates ``rccx`` (with 2 controls) and ``rcccx`` (with 3 controls).


`0.5.0`_ - 2019-05-02
=====================

Added
-----

- Implementation of the HHL algorithm supporting ``LinearSystemInput``.
- Pluggable component ``Eigenvalues`` with variant ``EigQPE``.
- Pluggable component ``Reciprocal`` with variants ``LookupRotation`` and ``LongDivision``.
- Multiple-Controlled U1 and U3 operations ``mcu1`` and ``mcu3``.
- Pluggable component ``QFT`` derived from component ``IQFT``.
- Summarize the transpiled circuits at the DEBUG logging level.
- ``QuantumInstance`` accepts ``basis_gates`` and ``coupling_map`` again.
- Support to use ``cx`` gate for the entangement in ``RY`` and ``RYRZ`` variational form. (``cz`` is the default choice.)
- Support to use arbitrary mixer Hamiltonian in ``QAOA``. This allows to use QAOA in constrained optimization problems [arXiv:1709.03489].
- Added variational algorithm base class ``VQAlgorithm``, implemented by ``VQE`` and ``VQC``.
- Added ``ising/docplex.py`` for automatically generating Ising Hamiltonian from optimization models of DOcplex.
- Added ``'basic-dirty-ancilla'`` mode for ``mct``.
- Added ``mcmt`` for Multi-Controlled, Multi-Target gate.
- Exposed capabilities to generate circuits from logical AND, OR, DNF (disjunctive normal forms), and CNF (conjunctive normal forms) formulae.
- Added the capability to generate circuits from ESOP (exclusive sum of products) formulae with optional optimization based on Quine-McCluskey and ExactCover.
- Added ``LogicalExpressionOracle`` for generating oracle circuits from arbitrary boolean logic expressions (including DIMACS support) with optional optimization capability.
- Added ``TruthTableOracle`` for generating oracle circuits from truth-tables with optional optimization capability.
- Added ``CustomCircuitOracle`` for generating oracle from user specified circuits.
- Added implementation of the Deutsch-Jozsa algorithm.
- Added implementation of the Bernstein-Vazirani algorithm.
- Added implementation of the Simon's algorithm.
- Added implementation of the Shor's algorithm.
- Added optional capability for ``Grover``'s algorithm to take a custom initial state (as opposed to the default uniform superposition)
- Added capability to create a ``Custom`` initial state using existing circuit.
- Added the ADAM (and AMSGRAD) optimization algorithm
- Multivariate distributions added, so uncertainty models now have univariate and multivariate distribution components.
- Added option to include or skip the swaps operations for qft and iqft circuit constructions.
- Added classical linear system solver ``ExactLSsolver``.
- Added parameters ``auto_hermitian`` and ``auto_resize`` to ``HHL`` algorithm to support non-hermititan and non 2**n sized matrices by default.
- Added another feature map, ``RawFeatureVector``, that directly maps feature vectors to qubits' states for classification.
- ``SVM_Classical`` can now load models trained by ``QSVM``.
- Added ``CompleteMeasFitter`` for mitigating measurement error when jobs are run on a real device or noisy simulator.
- Added ``QGAN`` (Quantum Generative Adversarial Network) algorithm, along with neural network components comprising a quantum generator and classical discriminator.

Removed
-------

- ``QuantumInstance`` does not take ``memory`` anymore.
- Moved Command line and GUI interfaces to separate repo (qiskit_aqua_uis).
- Removed the ``SAT``-specific oracle (now supported by ``LogicalExpressionOracle``).


Changed
-------

- Changed the type of ``entanger_map`` used in ``FeatureMap`` and ``VariationalForm`` to list of list.
- Fixed package setup to correctly identify namespace packages using ``setuptools.find_namespace_packages``.
- Changed ``advanced`` mode implementation of ``mct``: using simple ``h`` gates instead of ``ch``, and fixing the old recursion step in ``_multicx``.
- Components ``random_distributions`` renamed to ``uncertainty_models``.
- Reorganized the constructions of various common gates (``ch``, ``cry``, ``mcry``, ``mct``, ``mcu1``, ``mcu3``, ``mcmt``, ``logic_and``, and ``logic_or``) and circuits (``PhaseEstimationCircuit``, ``BooleanLogicCircuits``, ``FourierTransformCircuits``, and ``StateVectorCircuits``) under the ``circuits`` directory.
- Renamed the algorithm ``QSVMVariational`` to ``VQC``, which stands for Variational Quantum Classifier.
- Renamed the algorithm ``QSVMKernel`` to ``QSVM``.
- Renamed the class ``SVMInput`` to ``ClassificationInput``.
- Renamed problem type ``'svm_classification'`` to ``'classification'``

Fixed
-----

- Fixed ``ising/docplex.py`` to correctly multiply constant values in constraints


`0.4.1`_ - 2019-01-09
=====================

Added
-----

- Optimizers now have most relevant options on constructor for ease of programming. Options may still be set via set_options.
- Provider is now explicitly named and the named backend is created from that named provider. Backend being selected from the first of the internally known set of providers is deprecated.
- Improve operation with Aer provider/backends.
- Registration to Aqua of externally provided pluggable algorithms and components altered to setuptools entry point support for plugins. The prior registration mechanism has been removed.
- A flag ``before_04`` in the ``load_from_dict(file)`` method is added to support to load operator in the old format. We encourage to save the operator in the new format from now on.

`0.4.0`_ - 2018-12-19
=====================

Added
-----

- Compatibility with Terra 0.7
- Compatibility with Aer 0.1
- Programmatic APIs for algorithms and components -- each component can now be instantiated and initialized via a single (non-emptY) constructot call
- ``QuantumInstance`` API for algorithm/backend decoupling -- ``QuantumInstance`` encapsulates a backend and its settings
- Updated documentation and Jupyter Notebooks illustrating the new programmatic APIs
- Transparent parallelization for gradient-based optimizers
- Multiple-Controlled-NOT (cnx) operation
- Pluggable algorithmic component ``RandomDistribution``
- Concrete implementations of ``RandomDistribution``: ``BernoulliDistribution``, ``LogNormalDistribution``,
  ``MultivariateDistribution``, ``MultivariateNormalDistribution``, ``MultivariateUniformDistribution``, ``NormalDistribution``,
  ``UniformDistribution``, and ``UnivariateDistribution``
- Pluggable algorithmic component:
- Concrete implementations of ``UncertaintyProblem``: ``FixedIncomeExpectedValue``, ``EuropeanCallExpectedValue``, and
  ``EuropeanCallDelta``
- Amplitude Estimation algorithm
- Qiskit Optimization: New Ising models for optimization problems exact cover, set packing, vertex cover, clique, and graph partition
- Qiskit AI:
   - New feature maps extending the ``FeatureMap`` pluggable interface: ``PauliExpansion`` and ``PauliZExpansion``
   - Training model serialization/deserialization mechanism
- Qiskit Finance:
   - Amplitude estimation for Bernoulli random variable: illustration of amplitude estimation on a single qubit problem
   - Loading of multiple univariate and multivariate random distributions
   - European call option: expected value and delta (using univariate distributions)
   - Fixed income asset pricing: expected value (using multivariate distributions)

Changed
-------

- The pauli string in ``Operator`` class is aligned with Terra 0.7. Now the order of a n-qubit pauli string is ``q_{n-1}...q{0}`` Thus, the (de)serialier (``save_to_dict`` and ``load_from_dict``) in the ``Operator`` class are also changed to adopt the changes of ``Pauli`` class.

Removed
-------

- ``HartreeFock`` component of pluggable type ``InitialState` moved to Qiskit Chemistry
- ``UCCSD`` component of pluggable type ``VariationalForm`` moved to Qiskit Chemistry

`0.3.1`_ - 2018-11-29
=====================

Changed
-------

- Different backends might have different signatures for describing the job completion.

`0.3.0`_ - 2018-10-05
=====================

Added
-----

- Updated for 0.6 Terra
- Enhanced backend settings
- Pluggable multiclass classifier extensions
   - AllPairs
   - OneAgainstAll
   - ErrorCorrectingCode
- Pluggable Feature Maps for QSVM algos
- Pluggable Variation Forms for QSVM.Variational
- SPSA calibration and control variables all configurable
- Step size configurable for optimizers with numerical approximation of the jacobian
- Z2 Symmetry tapering
   - Operator
   - HartreeFock InitialState
   - UCCSD
- UCCSD performance improvements
- Remote device/simulator job auto-recovery
- Algorithm concatenation: VQE->(I)QPE
- Operator improvements
   - Subtraction
   - Negation
   - Scaling

`0.2.0`_ - 2018-07-27
=====================

Added
-----

- Ising model for TSP.
- add summarize circuits.
- Relax max circuits for simulation.
- Added qubit_tapering method.
- multiclass svm (one against all).
- Allow dynamic loading preferences package.module.

Changed
-------

- Changed name from acqua to aqua.
- Move QAOA's variational form to under the algorithm implementation directory.
- Factor out the QAOA variational form.

Fixed
-----

- Operator will crash if the backend is None.
- Fix/max num circuits.
- fix grover for cases that don't need ancillary qubits.
- Fixed validation error for string of numbers.
- fix link to ai and opt notebooks.

`0.1.2`_ - 2018-07-12
=====================

Added
-----

- UI Preferences Page including proxies urls, provider, verify.
- Add help menu with link to documentation.
- Add num_iterations param to grover.
- Graph partition ising model added.
- F2 finite field functions and find_Z2_symmetries function.
- Added packages preferences array for client custom pluggable packages.

Changed
-------

- Clean up use_basis_gates options.
- Change Qiskit registering for Qiskit 0.5.5.

Fixed
-----

- GUI - Windows: new line appears when text view dismissed.
- Update test_grover to account for cases where the groundtruth info is missing.
- Qconfig discovery - Fix permission denied error on list folders.
- UI Fix Popup cut/copy/paste/select all behavior in mac/windows/linux.
- Fix typo grouped paulis.
- Fix numpy argmax usage on potentially complex state vector.
- Fix/use list for paulis and update helper function of ising model.


`0.1.1`_ - 2018-06-13
=====================

Changed
-------

- Changed short and long descriptions in setup.py.


`0.1.0` - 2018-06-13
=====================

Changed
-------

- Changed package name to dashes in setup.py.
- Updated qiskit minimum version in setup.py.
- Fixed links in readme.me.

.. _UNRELEASED: https://github.com/Qiskit/qiskit-aqua/compare/0.5.0...HEAD
.. _0.5.0: https://github.com/Qiskit/qiskit-aqua/compare/0.4.1...0.5.0
.. _0.4.1: https://github.com/Qiskit/qiskit-aqua/compare/0.4.0...0.4.1
.. _0.4.0: https://github.com/Qiskit/qiskit-aqua/compare/0.3.1...0.4.0
.. _0.3.1: https://github.com/Qiskit/qiskit-aqua/compare/0.3.0...0.3.1
.. _0.3.0: https://github.com/Qiskit/qiskit-aqua/compare/0.2.0...0.3.0
.. _0.2.0: https://github.com/Qiskit/qiskit-aqua/compare/0.1.2...0.2.0
.. _0.1.2: https://github.com/Qiskit/qiskit-aqua/compare/0.1.1...0.1.2
.. _0.1.1: https://github.com/Qiskit/qiskit-aqua/compare/0.1.0...0.1.1

.. _Keep a Changelog: http://keepachangelog.com/en/1.0.0/
