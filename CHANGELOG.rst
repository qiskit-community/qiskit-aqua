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

- Compatibility with Terra 0.7
- API changes (programmatic approach)
- QuantumInstance API for algorithm/backend decoupling
- Updated documentation and Jupyter Notebooks exhibiting both programmatic and declarative APIs
- Hartree-Fock initial state method and UCCSD variational form moved to Aqua Chemistry
- Transparent parallelization implemented for gradient-based SciPy optimizers
- Amplitude estimation algorithm
- Aqua Optimization: New Ising model for: exact cover, set packing, vertex cover
- Aqua AI: New feature maps extending the FeatureMap pluggable interface: PauliExpansion and PauliZExpansion
- Aqua Finance:
   - Amplitude estimation for Bernoulli random variable: illustration of amplitude estimation on a single qubit problem
   - Loading of multiple univariate and multivariate random distributions
   - European call option: expected value and delta (using univariate distributions)
   - Fixed income asset pricing: expected value (using multivariate distributions)

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

.. _UNRELEASED: https://github.com/Qiskit/aqua/compare/0.3.1...HEAD
.. _0.3.1: https://github.com/Qiskit/aqua/compare/0.3.0...0.3.1
.. _0.3.0: https://github.com/Qiskit/aqua/compare/0.2.0...0.3.0
.. _0.2.0: https://github.com/Qiskit/aqua/compare/0.1.2...0.2.0
.. _0.1.2: https://github.com/Qiskit/aqua/compare/0.1.1...0.1.2
.. _0.1.1: https://github.com/Qiskit/aqua/compare/0.1.0...0.1.1

.. _Keep a Changelog: http://keepachangelog.com/en/1.0.0/
