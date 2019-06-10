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

[UNRELEASED](https://github.com/Qiskit/qiskit-chemistry/compare/0.5.0...HEAD)
=============================================================================

Added
-----

- ROHF open-shell support
    - Supported for all drivers: Gaussian16, PyQuante, PySCF and PSI4
    - HartreeFock initial state, UCCSD variational form and two qubit reduction for
      parity mapping now support different alpha and beta particle numbers for open
      shell support

- UHF open-shell support
    - Supported for all drivers: Gaussian16, PyQuante, PySCF and PSI4
    - QMolecule extended to include integrals, coeffiecients etc for separate beta
    
- QMolecule extended with integrals in atomic orbital basis to facilitate common access
  to these for experimentation
    - Supported for all drivers: Gaussian16, PyQuante, PySCF and PSI4

- Additional PyQuante and PySCF driver configuration
    - Convergence tolerance and max convergence iteration controls.
    - For PySCF initial guess choice
    
- Processing output added to debug log from PyQuante and PySCF computations (Gaussian16 
  and PSI4 outputs were already added to debug log)

Fixed
-----

- Bravyi-Kitaev mapping fixed when num qubits was not a power of 2


[0.5.0](https://github.com/Qiskit/qiskit-chemistry/compare/0.4.2...0.5.0) - 2019-05-02
======================================================================================

Removed
-------

-   Moved Command line and GUI interfaces to separate repo
    (qiskit\_aqua\_uis)

[0.4.2](https://github.com/Qiskit/qiskit-chemistry/compare/0.4.1...0.4.2) - 2019-01-09
======================================================================================

Added
-----

-   Programming API for drivers simplified. Molecular config is now
    supplied on constructor consistent with other algorithms and
    components in Aqua.
-   UCCSD and HartreeFock components now are registered to Aqua using
    the setuptools mechanism now supported by Aqua.
-   Z-Matrix support, as added to PySCF and PyQuante drivers in 0.4.0,
    now allows dummy atoms to included so linear molecules can be
    specified. Also dummy atoms may be used to simplify the Z-Matrix
    specification of the molecule by appropriate choice of dummy
    atom(s).
-   HartreeFock state now uses a bitordering which is consistent with
    Qiskit Terra result ordering.

[0.4.1](https://github.com/Qiskit/qiskit-chemistry/compare/0.4.0...0.4.1) - 2018-12-21
======================================================================================

Changed
-------

-   Changed package name and imports to qiskit\_chemistry

Fixed
-----

-   \"ModuleNotFoundError unavailable in python 3.5\"

[0.4.0](https://github.com/Qiskit/qiskit-chemistry/compare/0.3.0...0.4.0) - 2018-12-19
======================================================================================

Added
-----

-   Compatibility with Aqua 0.4
-   Compatibility with Terra 0.7
-   Compatibility with Aer 0.1
-   Programmatic APIs for algorithms and components \-- each component
    can now be instantiated and initialized via a single (non-emptY)
    constructot call
-   `QuantumInstance` API for algorithm/backend decoupling \--
    `QuantumInstance` encapsulates a backend and its settings
-   Updated documentation and Jupyter Notebooks illustrating the new
    programmatic APIs
-   Z-Matrix support for the PySCF & PyQuante classical computational
    chemistry drivers
-   `HartreeFock` component of pluggable type
    [\`InitialState]{.title-ref} moved from Qiskit Aqua to Qiskit
    Chemistry registers itself at installation time as Aqua algorithmic
    components for use at run time
-   `UCCSD` component of pluggable type `VariationalForm` moved from
    Qiskit Aqua to Qiskit Chemistry registers itself at installation
    time as Aqua algorithmic components for use at run time

[0.3.0](https://github.com/Qiskit/qiskit-chemistry/compare/0.2.0...0.3.0) - 2018-10-05
======================================================================================

Added
-----

-   BKSF Mapping
-   Operator tapering example

[0.2.0](https://github.com/Qiskit/qiskit-chemistry/compare/0.1.1...0.2.0) - 2018-07-27
======================================================================================

Added
-----

-   Allow dynamic loading preferences package.module.
-   Dynamic loading of client preference chemistry operators and
    drivers.

Changed
-------

-   Changed name from acqua to aqua.
-   Add version to about dialog

Fixed
-----

-   Fixed validation error for string of numbers.
-   Fix backend name ui show

[0.1.1](https://github.com/Qiskit/qiskit-chemistry/compare/0.1.0...0.1.1) - 2018-07-12
======================================================================================

Added
-----

-   UI Preferences Page including proxies urls, provider, verify.

Changed
-------

-   Remove use\_basis\_gates flag.
-   Change Qiskit registering for Qiskit 0.5.5.
-   Changed enable\_substitutions to auto\_substitutions.

Fixed
-----

-   GUI - Windows: new line appears when text view dismissed.
-   Catch qconfig.py save error.
-   UI Fix Popup cut/copy/paste/select all behavior in
    mac/windows/linux.
-   UI Should truncate debug output for large arrays

[0.1.0]{.title-ref} - 2018-06-13
================================

Changed
-------

-   Changed description and change package name to dashes in setup.py.
-   Update description and fixed links in readme
