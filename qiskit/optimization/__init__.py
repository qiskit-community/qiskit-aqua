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

"""

from .utils import QiskitOptimizationError
from .problems import OptimizationProblem
from ._logging import (get_qiskit_optimization_logging,
                       set_qiskit_optimization_logging)
from .infinity import infinity
from .util import get_qubo_solutions

__all__ = ['OptimizationProblem',
           'QiskitOptimizationError',
           'get_qiskit_optimization_logging',
           'set_qiskit_optimization_logging',
           'infinity',
           'get_qubo_solutions'
           ]
