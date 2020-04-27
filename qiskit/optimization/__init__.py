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

.. currentmodule:: qiskit.optimization

Contents
========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    QuadraticProgram
    QiskitOptimizationError
    INFINITY

"""

from .infinity import INFINITY  # must be at the top of the file
from .exceptions import QiskitOptimizationError
from .problems import QuadraticProgram
from ._logging import (get_qiskit_optimization_logging,
                       set_qiskit_optimization_logging)

__all__ = ['QuadraticProgram',
           'QiskitOptimizationError',
           'get_qiskit_optimization_logging',
           'set_qiskit_optimization_logging',
           'INFINITY'
           ]
