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

`0.1.1`_ - 2018-07-12
=====================

Added
-----

- UI Preferences Page including proxies urls, provider, verify.

Changed
-------

- Remove use_basis_gates flag.
- Change QISKit registering for QISKit 0.5.5.
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

.. _UNRELEASED: https://github.com/QISKit/qiskit-acqua-chemistry/compare/0.1.1...HEAD
.. _0.1.1: https://github.com/QISKit/qiskit-acqua-chemistry/compare/0.1.0...0.1.1

.. _Keep a Changelog: http://keepachangelog.com/en/1.0.0/
