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
=================================================================
Optimization stack for Aqua (:mod:`qiskit.optimization.problems`)
=================================================================

.. currentmodule:: qiskit.optimization.problems

Structures for defining an optimization problem and its solution
==========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   LinearConstraintInterface
   ObjSense
   ObjectiveInterface
   OptimizationProblem
   QuadraticConstraintInterface
   VariablesInterface

N.B. Additional classes LinearConstraintInterface, QuadraticConstraintInterface,
ObjectiveInterface, and VariablesInterface
are not to be instantiated directly. Objects of those types are available within
an instantiated OptimizationProblem.

"""

from .linear_constraint import LinearConstraintInterface
from .objective import ObjSense, ObjectiveInterface
from .optimization_problem import OptimizationProblem
from .quadratic_constraint import QuadraticConstraintInterface
from .variables import VariablesInterface

__all__ = ['LinearConstraintInterface',
           'ObjSense',
           'ObjectiveInterface',
           'OptimizationProblem',
           'QuadraticConstraintInterface',
           'VariablesInterface'
           ]
