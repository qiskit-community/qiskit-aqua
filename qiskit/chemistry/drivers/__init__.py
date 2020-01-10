# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Chemistry Drivers (:mod:`qiskit.chemistry.drivers`)
=========================================================
Chemistry drivers take a molecule configuration as input, and run classical
software to produce a :class:`QMolecule` containing information the
chemistry stacks needs to produce input for a Quantum Algorithm. Such information
includes one and two-body electronic integrals, dipole integrals, nuclear
repulsion energy and more.

.. currentmodule:: qiskit.chemistry.drivers

Driver Base Class
=================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BaseDriver

Driver Common
=============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   UnitsType
   HFMethodType
   BasisType
   InitialGuess

Drivers
=======
Qiskit Chemistry drivers obtain their information from classical ab-initio programs
or libraries. Several drivers, interfacing to common programs and libraries, are
available. To use the driver its dependent program/library must be installed. See
the relevant installation instructions below for your program/library that you intend
to use.

Note: `PySCF` is automatically installed for `macOS` and `Linux` platforms when Qiskit
is installed. For other platforms again consult the relevant installation instructions below.

.. toctree::
   :maxdepth: 1

   qiskit.chemistry.drivers.gaussiand
   qiskit.chemistry.drivers.psi4d
   qiskit.chemistry.drivers.pyscfd
   qiskit.chemistry.drivers.pyquanted

The :class:`HDF5Driver` reads and writes molecular data from a file and is not dependent
on any external chemistry program/library and needs no special install.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GaussianDriver
   HDF5Driver
   PSI4Driver
   PyQuanteDriver
   PySCFDriver

"""
from ._basedriver import BaseDriver, UnitsType, HFMethodType
from .gaussiand import GaussianDriver
from .hdf5d import HDF5Driver
from .psi4d import PSI4Driver
from .pyquanted import PyQuanteDriver, BasisType
from .pyscfd import PySCFDriver, InitialGuess

__all__ = ['BaseDriver',
           'UnitsType',
           'HFMethodType',
           'GaussianDriver',
           'HDF5Driver',
           'PSI4Driver',
           'BasisType',
           'PyQuanteDriver',
           'PySCFDriver',
           'InitialGuess']
