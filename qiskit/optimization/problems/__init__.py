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
Optimization stack for Aqua (:mod:`qiskit.optimization`)
========================================================

.. currentmodule:: qiskit.optimization.problems

Structures for defining an optimization problem and its solution
==========

.. autosummary::
   :toctree:

   OptimizationProblem
   VariablesInterface
   ObjectiveInterface
   LinearConstraintInterface
   QuadraticConstraintInterface

N.B. Additional classes LinearConstraintInterface, QuadraticConstraintInterface,
ObjectiveInterface, and VariablesInterface
are not to be instantiated directly. Objects of those types are available within
an instantiated OptimizationProblem.

"""

from qiskit.optimization.problems.optimization_problem import OptimizationProblem
from qiskit.optimization.problems.linear_constraint import LinearConstraintInterface
from qiskit.optimization.problems.objective import ObjSense, ObjectiveInterface

__all__ = ["OptimizationProblem", "LinearConstraintInterface", "ObjSense", "ObjectiveInterface"]
