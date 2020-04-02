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
========================================================
Optimization stack for Aqua (:mod:`qiskit.optimization.converters`)
========================================================

.. currentmodule:: qiskit.optimization.converters

Structures for converting optimization problems
==========

"""

from .inequality_to_equality_converter import InequalityToEqualityConverter
from .integer_to_binary_converter import IntegerToBinaryConverter
from .optimization_problem_to_negative_value_oracle import OptimizationProblemToNegativeValueOracle
from .optimization_problem_to_operator import OptimizationProblemToOperator
from .optimization_problem_to_qubo import OptimizationProblemToQubo
from .penalize_linear_equality_constraints import PenalizeLinearEqualityConstraints

__all__ = [
    "InequalityToEqualityConverter",
    "IntegerToBinaryConverter",
    "OptimizationProblemToNegativeValueOracle",
    "OptimizationProblemToOperator",
    "OptimizationProblemToQubo",
    "PenalizeLinearEqualityConstraints",
]
