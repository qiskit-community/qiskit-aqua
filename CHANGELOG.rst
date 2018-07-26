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
- Change QISKit registering for QISKit 0.5.5.

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

.. _UNRELEASED: https://github.com/QISKit/aqua/compare/0.1.2...HEAD
.. _0.1.2: https://github.com/QISKit/aqua/compare/0.1.1...0.1.2
.. _0.1.1: https://github.com/QISKit/aqua/compare/0.1.0...0.1.1

.. _Keep a Changelog: http://keepachangelog.com/en/1.0.0/
