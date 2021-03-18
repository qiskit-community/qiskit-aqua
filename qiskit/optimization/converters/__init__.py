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

This is a set of converters having `convert` functionality to go between different representations
of a given :class:`~qiskit.optimization.problems.QuadraticProgram` and to `interpret` a given
result for the problem, based on the original problem before conversion, to return an appropriate
:class:`~qiskit.optimization.algorithms.OptimizationResult`.

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

from .integer_to_binary import IntegerToBinary
from .inequality_to_equality import InequalityToEquality
from .linear_equality_to_penalty import LinearEqualityToPenalty
from .quadratic_program_to_qubo import QuadraticProgramToQubo
from .quadratic_program_converter import QuadraticProgramConverter

__all__ = [
    "InequalityToEquality",
    "IntegerToBinary",
    "LinearEqualityToPenalty",
    "QuadraticProgramConverter",
    "QuadraticProgramToQubo",
]
