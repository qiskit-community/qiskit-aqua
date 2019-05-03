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

`0.5.0`_ - 2019-05-02
=====================

Removed
-------

- Moved Command line and GUI interfaces to separate repo (qiskit_aqua_uis)


`0.4.2`_ - 2019-01-09
=====================

Added
-------

- Programming API for drivers simplified. Molecular config is now supplied on constructor
  consistent with other algorithms and components in Aqua.
- UCCSD and HartreeFock components now are registered to Aqua using the setuptools mechanism
  now supported by Aqua.
- Z-Matrix support, as added to PySCF and PyQuante drivers in 0.4.0, now allows dummy atoms to
  included so linear molecules can be specified. Also dummy atoms may be used to simplify
  the Z-Matrix specification of the molecule by appropriate choice of dummy atom(s).
- HartreeFock state now uses a bitordering which is consistent with Qiskit Terra result ordering.

`0.4.1`_ - 2018-12-21
=====================

Changed
-------

- Changed package name and imports to qiskit_chemistry

Fixed
-----

- "ModuleNotFoundError unavailable in python 3.5"


`0.4.0`_ - 2018-12-19
=====================

Added
-----

- Compatibility with Aqua 0.4
- Compatibility with Terra 0.7
- Compatibility with Aer 0.1
- Programmatic APIs for algorithms and components -- each component can now be instantiated and initialized via a single (non-emptY) constructot call
- ``QuantumInstance`` API for algorithm/backend decoupling -- ``QuantumInstance`` encapsulates a backend and its settings
- Updated documentation and Jupyter Notebooks illustrating the new programmatic APIs
- Z-Matrix support for the PySCF & PyQuante classical computational chemistry drivers
- ``HartreeFock`` component of pluggable type ``InitialState` moved from Qiskit Aqua to Qiskit Chemistry
  registers itself at installation time as Aqua algorithmic components for use at run time
- ``UCCSD`` component of pluggable type ``VariationalForm`` moved from Qiskit Aqua to Qiskit Chemistry
  registers itself at installation time as Aqua algorithmic components for use at run time

`0.3.0`_ - 2018-10-05
=====================

Added
-----

- BKSF Mapping
- Operator tapering example

`0.2.0`_ - 2018-07-27
=====================

Added
-----

- Allow dynamic loading preferences package.module.
- Dynamic loading of client preference chemistry operators and drivers.

Changed
-------

- Changed name from acqua to aqua.
- Add version to about dialog

Fixed
-----

- Fixed validation error for string of numbers.
- Fix backend name ui show

`0.1.1`_ - 2018-07-12
=====================

Added
-----

- UI Preferences Page including proxies urls, provider, verify.

Changed
-------

- Remove use_basis_gates flag.
- Change Qiskit registering for Qiskit 0.5.5.
- Changed enable_substitutions to auto_substitutions.

Fixed
-----

- GUI - Windows: new line appears when text view dismissed.
- Catch qconfig.py save error.
- UI Fix Popup cut/copy/paste/select all behavior in mac/windows/linux.
- UI Should truncate debug output for large arrays


`0.1.0` - 2018-06-13
=====================

Changed
-------

- Changed description and change package name to dashes in setup.py.
- Update description and fixed links in readme

.. _UNRELEASED: https://github.com/Qiskit/qiskit-chemistry/compare/0.5.0...HEAD
.. _0.5.0: https://github.com/Qiskit/qiskit-chemistry/compare/0.4.2...0.5.0
.. _0.4.2: https://github.com/Qiskit/qiskit-chemistry/compare/0.4.1...0.4.2
.. _0.4.1: https://github.com/Qiskit/qiskit-chemistry/compare/0.4.0...0.4.1
.. _0.4.0: https://github.com/Qiskit/qiskit-chemistry/compare/0.3.0...0.4.0
.. _0.3.0: https://github.com/Qiskit/qiskit-chemistry/compare/0.2.0...0.3.0
.. _0.2.0: https://github.com/Qiskit/qiskit-chemistry/compare/0.1.1...0.2.0
.. _0.1.1: https://github.com/Qiskit/qiskit-chemistry/compare/0.1.0...0.1.1

.. _Keep a Changelog: http://keepachangelog.com/en/1.0.0/
