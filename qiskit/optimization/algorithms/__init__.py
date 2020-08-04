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
Optimization algorithms (:mod:`qiskit.optimization.algorithms`)
===============================================================

Algorithms for optimization problems.

.. currentmodule:: qiskit.optimization.algorithms

Base class
==========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   OptimizationAlgorithm
   MultiStartOptimizer

Algorithms and results
======================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ADMMOptimizationResult
   ADMMOptimizer
   ADMMParameters
   ADMMState
   CobylaOptimizer
   CplexOptimizer
   GroverOptimizationRawResult
   GroverOptimizer
   MinimumEigenOptimizerResult
   MinimumEigenOptimizer
   OptimizationResult
   OptimizationResultStatus
   RecursiveMinimumEigenOptimizer
   SlsqpOptimizer

"""

from .admm_optimizer import ADMMOptimizer, ADMMOptimizationResult, ADMMState, ADMMParameters
from .cobyla_optimizer import CobylaOptimizer
from .cplex_optimizer import CplexOptimizer
from .grover_optimizer import GroverOptimizer, GroverOptimizationRawResult
from .minimum_eigen_optimizer import MinimumEigenOptimizer, MinimumEigenOptimizerResult
from .multistart_optimizer import MultiStartOptimizer
from .optimization_algorithm import (OptimizationAlgorithm, OptimizationResult,
                                     OptimizationResultStatus)
from .recursive_minimum_eigen_optimizer import RecursiveMinimumEigenOptimizer
from .slsqp_optimizer import SlsqpOptimizer

__all__ = ["ADMMOptimizer", "OptimizationAlgorithm", "OptimizationResult", "CplexOptimizer",
           "CobylaOptimizer", "MinimumEigenOptimizer", "RecursiveMinimumEigenOptimizer",
           "GroverOptimizer", "GroverOptimizationRawResult", "SlsqpOptimizer"]
