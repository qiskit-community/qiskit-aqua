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
from ._discover_driver import (DRIVERS_ENTRY_POINT,
                               refresh_drivers,
                               register_driver,
                               deregister_driver,
                               get_driver_class,
                               get_driver_configuration,
                               local_drivers)
from .gaussiand import GaussianDriver
from .hdf5d import HDF5Driver
from .psi4d import PSI4Driver
from .pyquanted import PyQuanteDriver, BasisType
from .pyscfd import PySCFDriver, InitialGuess

__all__ = ['BaseDriver',
           'UnitsType',
           'HFMethodType',
           'DRIVERS_ENTRY_POINT',
           'refresh_drivers',
           'register_driver',
           'deregister_driver',
           'get_driver_class',
           'get_driver_configuration',
           'local_drivers',
           'GaussianDriver',
           'HDF5Driver',
           'PSI4Driver',
           'BasisType',
           'PyQuanteDriver',
           'PySCFDriver',
           'InitialGuess']
