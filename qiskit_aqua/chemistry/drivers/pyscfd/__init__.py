# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
PYSCF Installation
==================
`PySCF <https://github.com/sunqm/pyscf>`__ is an open-source library for computational chemistry.
In order for Qiskit's chemistry module to interface PySCF and execute PySCF to
extract the electronic structure information PySCF must be installed.

According to the `PySCF installation instructions <http://sunqm.github.io/pyscf/install.html>`__,
the preferred installation method is via the pip package management system.  Doing so,
while in the Python virtual environment where Qiskit's chemistry module is also installed, will
automatically make PySCF available to Qiskit at run time.
"""

from .pyscfdriver import PySCFDriver, InitialGuess

__all__ = ['PySCFDriver',
           'InitialGuess']
