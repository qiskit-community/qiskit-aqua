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
====================================================================
Optimization application stack for Aqua (:mod:`qiskit.optimization`)
====================================================================
This is the finance domain logic....

.. currentmodule:: qiskit.optimization

Submodules
==========

.. autosummary::
   :toctree:

   OptimizationProblem

"""

from qiskit.optimization.infinity import infinity  # must be at the top of the file
from qiskit.optimization.utils import QiskitOptimizationError
from qiskit.optimization.problems.linear_constraint import LinearConstraintInterface
from qiskit.optimization.problems.objective import ObjSense, ObjectiveInterface
from qiskit.optimization.problems.optimization_problem import OptimizationProblem
from ._logging import (get_qiskit_optimization_logging,
                       set_qiskit_optimization_logging)

__all__ = ["OptimizationProblem", "QiskitOptimizationError", "LinearConstraintInterface",
           "ObjSense", "ObjectiveInterface", "infinity", 'get_qiskit_optimization_logging',
           'set_qiskit_optimization_logging']
