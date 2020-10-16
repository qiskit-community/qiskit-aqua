# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Qubit Transformations (:mod:`qiskit.chemistry.transformations`)
===============================================================
.. currentmodule:: qiskit.chemistry.transformations

Transformations for both Fermionic and Bosonic operators to qubit operators. Transformation
includes specification of qubit mapping type for example, as well as other options. As part of
the transformation of the main operator other, so called auxiliary operators, may be created to
enable other properties of the result state with the main operator to also be evaluated, such as
total spin for example.

Base Transformation
===================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Transformation

Fermionic Transformation
========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   FermionicTransformation
   FermionicQubitMappingType
   FermionicTransformationType

Bosonic Transformation
======================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BosonicTransformation
   BosonicQubitMappingType
   BosonicTransformationType

"""

from .fermionic_transformation import (FermionicTransformation,
                                       FermionicQubitMappingType,
                                       FermionicTransformationType)
from .bosonic_transformation import (BosonicTransformation,
                                     BosonicQubitMappingType,
                                     BosonicTransformationType)
from .transformation import Transformation

__all__ = [
    'FermionicTransformation',
    'FermionicQubitMappingType',
    'FermionicTransformationType',
    'BosonicTransformation',
    'BosonicQubitMappingType',
    'BosonicTransformationType',
    'Transformation'
]
