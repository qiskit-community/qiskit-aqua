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

.. currentmodule:: qiskit.optimization.utils

Utility classes and functions
==========

.. autosummary::
   :toctree:

   QiskitOptimizationError
   BaseInterface
   SparsePair
   SparseTriple

N.B. Utility functions in .aux are intended for internal use.

"""

from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError
from qiskit.optimization.utils.base import BaseInterface
from qiskit.optimization.utils.eigenvector_to_solutions import eigenvector_to_solutions

__all__ = ["QiskitOptimizationError", "BaseInterface", "eigenvector_to_solutions"]
