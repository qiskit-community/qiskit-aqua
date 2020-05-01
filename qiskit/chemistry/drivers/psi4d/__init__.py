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
PSI4 Installation
=================
`PSI4 <http://www.psicode.org/>`__ is an open-source program for computational chemistry.
In order for Qiskit's chemistry module to interface PSI4, i.e. execute PSI4 to extract
the electronic structure information necessary for the computation of the input to the quantum
algorithm, PSI4 must be `installed <http://www.psicode.org/downloads.html>`__ and discoverable on
the system where Qiskit's chemistry module is also installed.

Therefore, once PSI4 has been installed, the `psi4` executable must be reachable via the system
environment path. For example, on macOS, this can be achieved by adding the following section to
the `.bash_profile` file in the user's home directory:

.. code:: sh

    # PSI4
    alias enable_psi4='export PATH=/Users/username/psi4conda/bin:$PATH'

where `username` should be replaced with the user's account name.
In order for the chemistry module to discover PSI4 at run time, it is then necessary to execute the
`enable_psi4` command before launching Qiskit.

"""

from .psi4driver import PSI4Driver

__all__ = ['PSI4Driver']
