# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Optimization Converters (:mod:`qiskit.optimization.converters`)
====================================================================
Converters for optimization problems

.. currentmodule:: qiskit.optimization.converters

Converters for Operators and Oracles
=========================================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   optimization_problem_to_negative_value_oracle

Utilities/Results Objects
=========================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   grover_optimization_results
   portfolio_util

"""

from .optimization_problem_to_negative_value_oracle import OptimizationProblemToNegativeValueOracle
