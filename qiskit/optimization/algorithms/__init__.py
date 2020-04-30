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

Algorithms
==========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ADMMOptimizer
   CobylaOptimizer
   CplexOptimizer
   GroverOptimizer
   MinimumEigenOptimizer
   RecursiveMinimumEigenOptimizer

"""

from .optimization_algorithm import OptimizationResult
from .optimization_algorithm import OptimizationAlgorithm
from .admm_optimizer import ADMMOptimizer
from .cplex_optimizer import CplexOptimizer
from .cobyla_optimizer import CobylaOptimizer
from .minimum_eigen_optimizer import MinimumEigenOptimizer
from .recursive_minimum_eigen_optimizer import RecursiveMinimumEigenOptimizer
from .grover_optimizer import GroverOptimizer, GroverOptimizationResults

__all__ = ["ADMMOptimizer", "OptimizationAlgorithm", "OptimizationResult", "CplexOptimizer",
           "CobylaOptimizer", "MinimumEigenOptimizer", "RecursiveMinimumEigenOptimizer",
           "GroverOptimizer", "GroverOptimizationResults"]
