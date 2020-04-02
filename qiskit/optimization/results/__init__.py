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
Optimization Results Objects (:mod:`qiskit.optimization.results`)
Results objects for optimization problems

.. currentmodule:: qiskit.optimization.results

Results Objects
=========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   grover_optimization_results

========================================================
Optimization stack for Aqua (:mod:`qiskit.optimization.results`)
========================================================

.. currentmodule:: qiskit.optimization.results

Structures for defining a solution with metrics of its quality etc
==========

.. autosummary::
   :toctree:

   SolutionStatus
   QualityMetrics

"""

from .quality_metrics import QualityMetrics
from .solution import SolutionInterface
from .solution_status import SolutionStatus
from .optimization_result import OptimizationResult
from .grover_optimization_results import GroverOptimizationResults

__all__ = ["SolutionStatus", "QualityMetrics", "SolutionStatus", "OptimizationResult",
           "GroverOptimizationResults"]
