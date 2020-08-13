# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Optimization converters (:mod:`qiskit.optimization.converters`)
===============================================================

.. currentmodule:: qiskit.optimization.converters

This is selection of converters having encode, decode functionality to go between different
forms.

Base class for converters
=========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QuadraticProgramConverter

Converters
==========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   InequalityToEquality
   IntegerToBinary
   LinearEqualityToPenalty
   QuadraticProgramToQubo

"""

from .inequality_to_equality import InequalityToEquality
from .integer_to_binary import IntegerToBinary
from .linear_equality_to_penalty import LinearEqualityToPenalty
from .quadratic_program_converter import QuadraticProgramConverter
from .quadratic_program_to_qubo import QuadraticProgramToQubo

__all__ = [
    "InequalityToEquality",
    "IntegerToBinary",
    "LinearEqualityToPenalty",
    "QuadraticProgramConverter",
    "QuadraticProgramToQubo",
]
