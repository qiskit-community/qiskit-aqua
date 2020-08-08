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

Converters
==========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   InequalityToEquality
   IntegerToBinary
   QuadraticProgramToQubo
   LinearEqualityToPenalty
   QuadraticProgramToIsing
   IsingToQuadraticProgram

"""

# opt problem dependency
from .integer_to_binary import IntegerToBinary
from .inequality_to_equality import InequalityToEquality
from .linear_equality_to_penalty import LinearEqualityToPenalty
from .quadratic_program_to_qubo import QuadraticProgramToQubo
from .quadratic_program_to_ising import QuadraticProgramToIsing
from .ising_to_quadratic_program import IsingToQuadraticProgram


__all__ = [
    "InequalityToEquality",
    "IntegerToBinary",
    "QuadraticProgramToQubo",
    "LinearEqualityToPenalty",
    "QuadraticProgramToIsing",
    "IsingToQuadraticProgram"
]
