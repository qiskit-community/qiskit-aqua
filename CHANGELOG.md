Changelog
=========

All notable changes to this project will be documented in this file.

The format is based on [Keep a
Changelog](http://keepachangelog.com/en/1.0.0/).

> **Types of changes:**
>
> -   **Added**: for new features.
> -   **Changed**: for changes in existing functionality.
> -   **Deprecated**: for soon-to-be removed features.
> -   **Removed**: for now removed features.
> -   **Fixed**: for any bug fixes.
> -   **Security**: in case of vulnerabilities.

[UNRELEASED](https://github.com/Qiskit/qiskit-aqua/compare/0.5.5...HEAD)
========================================================================

Added
-----

-   Relative-Phase Toffoli gates `rccx` (with 2 controls) and `rcccx`
    (with 3 controls).
-   Variational form `RYCRX` 
-   A new `'basic-no-ancilla'` mode to `mct`.
-   Multi-controlled rotation gates `mcrx`, `mcry`, and `mcrz` as a general 
    `u3` gate is not supported by graycode implementation
-   Chemistry: ROHF open-shell support
    - Supported for all drivers: Gaussian16, PyQuante, PySCF and PSI4
    - HartreeFock initial state, UCCSD variational form and two qubit reduction for
      parity mapping now support different alpha and beta particle numbers for open
      shell support
-   Chemistry: UHF open-shell support
    - Supported for all drivers: Gaussian16, PyQuante, PySCF and PSI4
    - QMolecule extended to include integrals, coeffiecients etc for separate beta   
-   Chemistry: QMolecule extended with integrals in atomic orbital basis to facilitate common access
    to these for experimentation
    - Supported for all drivers: Gaussian16, PyQuante, PySCF and PSI4
-   Chemistry: Additional PyQuante and PySCF driver configuration
    - Convergence tolerance and max convergence iteration controls.
    - For PySCF initial guess choice   
-   Chemistry: Processing output added to debug log from PyQuante and PySCF computations (Gaussian16 
    and PSI4 outputs were already added to debug log)
-   Chemistry: Merged qiskit-chemistry to this repo. The old chemistry changelog is at 
    [OLD_CHEMISTRY_CHANGELOG.md](OLD_CHEMISTRY_CHANGELOG.md)
-   Add `MatrixOperator`, `WeightedPauliOperator` and `TPBGroupedPauliOperator` class. (#593)
-   Add `evolution_instruction` function to get registerless instruction of time evolution. (#593)
-   Add `op_converter` module to unified the place in charge of converting different types of operators. (#593)
-   Add `Z2Symmetries` class to encapsulate the Z2 symmetries info and has helper methods for tapering an
    Operator. (#593).

Changed
-------

-   Improve `mct`'s `'basic'` mode by using relative-phase Toffoli gates to build intermediate results.
-   Adapt to Qiskit Terra's newly introduced `Qubit` class.
-   Prevent `QPE/IQPE` from modifying input `Operator`s.
-   The PyEDA dependency was removed; 
    corresponding oracles' underlying logic operations are now handled by SymPy.
-   Refactor the `Operator` class, each representation has its own class `MatrixOperator`, 
    `WeightedPauliOperator` and `TPBGroupedPauliOperator`. (#593)
-   The `power` in `evolution_instruction` was applied on the theta on the CRZ gate directly, 
    the new version repeats the circuits to implement power. (#593)
-   CircuitCache is OFF by default, and it can be set via environment variable now 
    `QISKIT_AQUA_CIRCUIT_CACHE`. (#630)

Fixed
-------

-   A bug where `TruthTableOracle` would build incorrect circuits for truth tables with only a single `1` value. 
-   A bug caused by `PyEDA`'s indeterminism.
-   A bug with `QPE/IQPE`'s translation and stretch computation.
-   Chemistry: Bravyi-Kitaev mapping fixed when num qubits was not a power of 2
-   Setup `initial_layout` in `QuantumInstance` via a list. (#630)

Removed
-------

-   General multi-controlled rotation gate `mcu3` is removed and replaced by 
    multi-controlled rotation gates `mcrx`, `mcry`, and `mcrz` 

Deprecated
----------

-   The `Operator` class is deprecated, in favor of using `MatrixOperator`, 
    `WeightedPauliOperator` and `TPBGroupedPauliOperator`. (#593)

[0.5.5](https://github.com/Qiskit/qiskit-aqua/compare/0.5.4...0.5.5) - 2019-07-26
=================================================================================

Fixed
-----

-   A bug with `docplex.get_qubitops`'s incorrect translation

[0.5.4](https://github.com/Qiskit/qiskit-aqua/compare/0.5.3...0.5.4) - 2019-07-24
=================================================================================

Fixed
-----

-   Fix the bug about manipulating the right operand and rebuild diagonal matrix every time. (#622)
 
[0.5.3](https://github.com/Qiskit/qiskit-aqua/compare/0.5.2...0.5.3) - 2019-07-16
=================================================================================

Fixed
-----

-   Since the syntax inverse() on a gate does not invert a gate now, the bug introduced wrong post rotation for Pauli Y.

[0.5.2](https://github.com/Qiskit/qiskit-aqua/compare/0.5.1...0.5.2) - 2019-06-27
=================================================================================

Changed
-------

-   The pyeda requirement was made optional instead of an install requirement

[0.5.1](https://github.com/Qiskit/qiskit-aqua/compare/0.5.0...0.5.1) - 2019-05-24
=================================================================================

Changed
-------

-   Make torch optional install

[0.5.0](https://github.com/Qiskit/qiskit-aqua/compare/0.4.1...0.5.0) - 2019-05-02
=================================================================================

Added
-----

-   Implementation of the HHL algorithm supporting `LinearSystemInput`.
-   Pluggable component `Eigenvalues` with variant `EigQPE`.
-   Pluggable component `Reciprocal` with variants `LookupRotation` and
    `LongDivision`.
-   Multiple-Controlled U1 and U3 operations `mcu1` and `mcu3`.
-   Pluggable component `QFT` derived from component `IQFT`.
-   Summarize the transpiled circuits at the DEBUG logging level.
-   `QuantumInstance` accepts `basis_gates` and `coupling_map` again.
-   Support to use `cx` gate for the entangement in `RY` and `RYRZ`
    variational form. (`cz` is the default choice.)
-   Support to use arbitrary mixer Hamiltonian in `QAOA`. This allows to
    use QAOA in constrained optimization problems \[arXiv:1709.03489\].
-   Added variational algorithm base class `VQAlgorithm`, implemented by
    `VQE` and `VQC`.
-   Added `ising/docplex.py` for automatically generating Ising
    Hamiltonian from optimization models of DOcplex.
-   Added `'basic-dirty-ancilla'` mode for `mct`.
-   Added `mcmt` for Multi-Controlled, Multi-Target gate.
-   Exposed capabilities to generate circuits from logical AND, OR, DNF
    (disjunctive normal forms), and CNF (conjunctive normal forms)
    formulae.
-   Added the capability to generate circuits from ESOP (exclusive sum
    of products) formulae with optional optimization based on
    Quine-McCluskey and ExactCover.
-   Added `LogicalExpressionOracle` for generating oracle circuits from
    arbitrary boolean logic expressions (including DIMACS support) with
    optional optimization capability.
-   Added `TruthTableOracle` for generating oracle circuits from
    truth-tables with optional optimization capability.
-   Added `CustomCircuitOracle` for generating oracle from user
    specified circuits.
-   Added implementation of the Deutsch-Jozsa algorithm.
-   Added implementation of the Bernstein-Vazirani algorithm.
-   Added implementation of the Simon\'s algorithm.
-   Added implementation of the Shor\'s algorithm.
-   Added optional capability for `Grover`\'s algorithm to take a custom
    initial state (as opposed to the default uniform superposition)
-   Added capability to create a `Custom` initial state using existing
    circuit.
-   Added the ADAM (and AMSGRAD) optimization algorithm
-   Multivariate distributions added, so uncertainty models now have
    univariate and multivariate distribution components.
-   Added option to include or skip the swaps operations for qft and
    iqft circuit constructions.
-   Added classical linear system solver `ExactLSsolver`.
-   Added parameters `auto_hermitian` and `auto_resize` to `HHL`
    algorithm to support non-hermititan and non 2\*\*n sized matrices by
    default.
-   Added another feature map, `RawFeatureVector`, that directly maps
    feature vectors to qubits\' states for classification.
-   `SVM_Classical` can now load models trained by `QSVM`.
-   Added `CompleteMeasFitter` for mitigating measurement error when
    jobs are run on a real device or noisy simulator.
-   Added `QGAN` (Quantum Generative Adversarial Network) algorithm,
    along with neural network components comprising a quantum generator
    and classical discriminator.

Removed
-------

-   `QuantumInstance` does not take `memory` anymore.
-   Moved Command line and GUI interfaces to separate repo
    (qiskit\_aqua\_uis).
-   Removed the `SAT`-specific oracle (now supported by
    `LogicalExpressionOracle`).

Changed
-------

-   Changed the type of `entanger_map` used in `FeatureMap` and
    `VariationalForm` to list of list.
-   Fixed package setup to correctly identify namespace packages using
    `setuptools.find_namespace_packages`.
-   Changed `advanced` mode implementation of `mct`: using simple `h`
    gates instead of `ch`, and fixing the old recursion step in
    `_multicx`.
-   Components `random_distributions` renamed to `uncertainty_models`.
-   Reorganized the constructions of various common gates (`ch`, `cry`,
    `mcry`, `mct`, `mcu1`, `mcu3`, `mcmt`, `logic_and`, and `logic_or`)
    and circuits (`PhaseEstimationCircuit`, `BooleanLogicCircuits`,
    `FourierTransformCircuits`, and `StateVectorCircuits`) under the
    `circuits` directory.
-   Renamed the algorithm `QSVMVariational` to `VQC`, which stands for
    Variational Quantum Classifier.
-   Renamed the algorithm `QSVMKernel` to `QSVM`.
-   Renamed the class `SVMInput` to `ClassificationInput`.
-   Renamed problem type `'svm_classification'` to `'classification'`

Fixed
-----

-   Fixed `ising/docplex.py` to correctly multiply constant values in
    constraints

[0.4.1](https://github.com/Qiskit/qiskit-aqua/compare/0.4.0...0.4.1) - 2019-01-09
=================================================================================

Added
-----

-   Optimizers now have most relevant options on constructor for ease of
    programming. Options may still be set via set\_options.
-   Provider is now explicitly named and the named backend is created
    from that named provider. Backend being selected from the first of
    the internally known set of providers is deprecated.
-   Improve operation with Aer provider/backends.
-   Registration to Aqua of externally provided pluggable algorithms and
    components altered to setuptools entry point support for plugins.
    The prior registration mechanism has been removed.
-   A flag `before_04` in the `load_from_dict(file)` method is added to
    support to load operator in the old format. We encourage to save the
    operator in the new format from now on.

[0.4.0](https://github.com/Qiskit/qiskit-aqua/compare/0.3.1...0.4.0) - 2018-12-19
=================================================================================

Added
-----

-   Compatibility with Terra 0.7
-   Compatibility with Aer 0.1
-   Programmatic APIs for algorithms and components \-- each component
    can now be instantiated and initialized via a single (non-emptY)
    constructot call
-   `QuantumInstance` API for algorithm/backend decoupling \--
    `QuantumInstance` encapsulates a backend and its settings
-   Updated documentation and Jupyter Notebooks illustrating the new
    programmatic APIs
-   Transparent parallelization for gradient-based optimizers
-   Multiple-Controlled-NOT (cnx) operation
-   Pluggable algorithmic component `RandomDistribution`
-   Concrete implementations of `RandomDistribution`:
    `BernoulliDistribution`, `LogNormalDistribution`,
    `MultivariateDistribution`, `MultivariateNormalDistribution`,
    `MultivariateUniformDistribution`, `NormalDistribution`,
    `UniformDistribution`, and `UnivariateDistribution`
-   Pluggable algorithmic component:
-   Concrete implementations of `UncertaintyProblem`:
    `FixedIncomeExpectedValue`, `EuropeanCallExpectedValue`, and
    `EuropeanCallDelta`
-   Amplitude Estimation algorithm
-   Qiskit Optimization: New Ising models for optimization problems
    exact cover, set packing, vertex cover, clique, and graph partition
-   

    Qiskit AI:

    :   -   New feature maps extending the `FeatureMap` pluggable
            interface: `PauliExpansion` and `PauliZExpansion`
        -   Training model serialization/deserialization mechanism

-   

    Qiskit Finance:

    :   -   Amplitude estimation for Bernoulli random variable:
            illustration of amplitude estimation on a single qubit
            problem
        -   Loading of multiple univariate and multivariate random
            distributions
        -   European call option: expected value and delta (using
            univariate distributions)
        -   Fixed income asset pricing: expected value (using
            multivariate distributions)

Changed
-------

-   The pauli string in `Operator` class is aligned with Terra 0.7. Now
    the order of a n-qubit pauli string is `q_{n-1}...q{0}` Thus, the
    (de)serialier (`save_to_dict` and `load_from_dict`) in the
    `Operator` class are also changed to adopt the changes of `Pauli`
    class.

Removed
-------

-   `HartreeFock` component of pluggable type
    [\`InitialState]{.title-ref} moved to Qiskit Chemistry
-   `UCCSD` component of pluggable type `VariationalForm` moved to
    Qiskit Chemistry

[0.3.1](https://github.com/Qiskit/qiskit-aqua/compare/0.3.0...0.3.1) - 2018-11-29
=================================================================================

Changed
-------

-   Different backends might have different signatures for describing
    the job completion.

[0.3.0](https://github.com/Qiskit/qiskit-aqua/compare/0.2.0...0.3.0) - 2018-10-05
=================================================================================

Added
-----

-   Updated for 0.6 Terra
-   Enhanced backend settings
-   

    Pluggable multiclass classifier extensions

    :   -   AllPairs
        -   OneAgainstAll
        -   ErrorCorrectingCode

-   Pluggable Feature Maps for QSVM algos
-   Pluggable Variation Forms for QSVM.Variational
-   SPSA calibration and control variables all configurable
-   Step size configurable for optimizers with numerical approximation
    of the jacobian
-   

    Z2 Symmetry tapering

    :   -   Operator
        -   HartreeFock InitialState
        -   UCCSD

-   UCCSD performance improvements
-   Remote device/simulator job auto-recovery
-   Algorithm concatenation: VQE-\>(I)QPE
-   

    Operator improvements

    :   -   Subtraction
        -   Negation
        -   Scaling

[0.2.0](https://github.com/Qiskit/qiskit-aqua/compare/0.1.2...0.2.0) - 2018-07-27
=================================================================================

Added
-----

-   Ising model for TSP.
-   add summarize circuits.
-   Relax max circuits for simulation.
-   Added qubit\_tapering method.
-   multiclass svm (one against all).
-   Allow dynamic loading preferences package.module.

Changed
-------

-   Changed name from acqua to aqua.
-   Move QAOA\'s variational form to under the algorithm implementation
    directory.
-   Factor out the QAOA variational form.

Fixed
-----

-   Operator will crash if the backend is None.
-   Fix/max num circuits.
-   fix grover for cases that don\'t need ancillary qubits.
-   Fixed validation error for string of numbers.
-   fix link to ai and opt notebooks.

[0.1.2](https://github.com/Qiskit/qiskit-aqua/compare/0.1.1...0.1.2) - 2018-07-12
=================================================================================

Added
-----

-   UI Preferences Page including proxies urls, provider, verify.
-   Add help menu with link to documentation.
-   Add num\_iterations param to grover.
-   Graph partition ising model added.
-   F2 finite field functions and find\_Z2\_symmetries function.
-   Added packages preferences array for client custom pluggable
    packages.

Changed
-------

-   Clean up use\_basis\_gates options.
-   Change Qiskit registering for Qiskit 0.5.5.

Fixed
-----

-   GUI - Windows: new line appears when text view dismissed.
-   Update test\_grover to account for cases where the groundtruth info
    is missing.
-   Qconfig discovery - Fix permission denied error on list folders.
-   UI Fix Popup cut/copy/paste/select all behavior in
    mac/windows/linux.
-   Fix typo grouped paulis.
-   Fix numpy argmax usage on potentially complex state vector.
-   Fix/use list for paulis and update helper function of ising model.

[0.1.1](https://github.com/Qiskit/qiskit-aqua/compare/0.1.0...0.1.1) - 2018-06-13
=================================================================================

Changed
-------

-   Changed short and long descriptions in setup.py.
                                                      
[0.1.0](https://github.com/Qiskit/qiskit-aqua/compare/7e913ef...0.1.0) - 2018-06-13
================================

Changed
-------

-   Changed package name to dashes in setup.py.
-   Updated qiskit minimum version in setup.py.
-   Fixed links in readme.me.
