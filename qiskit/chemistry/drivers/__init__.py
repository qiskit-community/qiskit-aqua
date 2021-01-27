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
Chemistry Drivers (:mod:`qiskit.chemistry.drivers`)
===================================================
.. currentmodule:: qiskit.chemistry.drivers

Qiskit's chemistry module requires a computational chemistry program or library, accessed via a
chemistry *driver*, to be installed on the system for the electronic-structure computation of a
given molecule. A driver is created with a molecular configuration, passed in the format compatible
with that particular driver. This allows custom configuration specific to each computational
chemistry program or library to be passed.

The chemistry module thus allows the user to configure a chemistry problem in a way that a chemist
already using the underlying chemistry program or library will be familiar with. The driver is
used to compute some intermediate data, which later will be used to form the input to an Aqua
algorithm.  Such intermediate data, is populated into a :class:`~qiskit.chemistry.QMolecule`
object and includes the following for example:

1. One- and two-body integrals in Molecular Orbital (MO) basis
2. Dipole integrals
3. Molecular orbital coefficients
4. Hartree-Fock energy
5. Nuclear repulsion energy

Once extracted, the structure of this intermediate data is independent of the driver that was
used to compute it.  However the values and level of accuracy of such data will depend on the
underlying chemistry program or library used by the specific driver.

Qiskit's chemistry module offers the option to serialize the Qmolecule data in a binary format known
as `Hierarchical Data Format 5 (HDF5) <https://support.hdfgroup.org/HDF5/>`__.
This is done to allow chemists to reuse the same input data in the future and to enable researchers
to exchange input data with each other --- which is especially useful to researchers who may not
have particular computational chemistry drivers installed on their computers.

Driver Base Class
=================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BaseDriver
   BosonicDriver
   FermionicDriver

Driver Common
=============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Molecule
   HFMethodType
   UnitsType
   BasisType
   InitialGuess

Drivers
=======

The drivers in the chemistry module obtain their information from classical ab-initio programs
or libraries. Several drivers, interfacing to common programs and libraries, are
available. To use the driver its dependent program/library must be installed. See
the relevant installation instructions below for your program/library that you intend
to use.

.. toctree::
   :maxdepth: 1

   qiskit.chemistry.drivers.gaussiand
   qiskit.chemistry.drivers.psi4d
   qiskit.chemistry.drivers.pyquanted
   qiskit.chemistry.drivers.pyscfd

The :class:`HDF5Driver` reads molecular data from a pre-existing HDF5 file, as saved from a
:class:`~qiskit.chemistry.QMolecule`, and is not dependent on any external chemistry
program/library and needs no special install.

The :class:`FCIDumpDriver` likewise reads from a pre-existing file in this case a standard
FCIDump file and again needs no special install.

Fermionic Drivers
=================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GaussianDriver
   PSI4Driver
   PyQuanteDriver
   PySCFDriver
   HDF5Driver
   FCIDumpDriver

Bosonic Drivers
===============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GaussianForcesDriver

General Driver
==============

The :class:`GaussianLogDriver` allows an arbitrary Gaussian Job Control File to be run and
return a :class:`GaussianLogResult` containing the log as well as ready access certain data
of interest that is parsed from the log.

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   GaussianLogDriver
   GaussianLogResult


"""

from .base_driver import BaseDriver
from .molecule import Molecule
from .bosonic_driver import BosonicDriver
from .fermionic_driver import FermionicDriver, HFMethodType
from .units_type import UnitsType
from .fcidumpd import FCIDumpDriver
from .gaussiand import GaussianDriver, GaussianLogDriver, GaussianLogResult, GaussianForcesDriver
from .hdf5d import HDF5Driver
from .psi4d import PSI4Driver
from .pyquanted import PyQuanteDriver, BasisType
from .pyscfd import PySCFDriver, InitialGuess

__all__ = ['HFMethodType',
           'Molecule',
           'BaseDriver',
           'BosonicDriver',
           'FermionicDriver',
           'UnitsType',
           'FCIDumpDriver',
           'GaussianDriver',
           'GaussianForcesDriver',
           'GaussianLogDriver',
           'GaussianLogResult',
           'HDF5Driver',
           'PSI4Driver',
           'BasisType',
           'PyQuanteDriver',
           'PySCFDriver',
           'InitialGuess']
