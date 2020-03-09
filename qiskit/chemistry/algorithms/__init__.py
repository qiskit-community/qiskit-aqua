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
Chemistry specific Aqua algorithms (:mod:`qiskit.chemistry.algorithms`)
=======================================================================
These are chemistry specific Aqua algorithms where they inherit from
:class:`QuantumAlgorithm`. As they rely on chemistry specific knowledge
and/or functions they live here rather than in Aqua.

.. currentmodule:: qiskit.chemistry.algorithms

Chemistry Quantum Algorithms
============================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QEomVQE
   VQEAdapt

Chemistry Classical Algorithms
==============================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   QEomEE

"""

from .eigen_solvers import QEomVQE, QEomEE
from .minimum_eigen_solvers import VQEAdapt, VQEAdaptResult

__all__ = [
    'QEomVQE',
    'QEomEE',
    'VQEAdapt',
    'VQEAdaptResult',
]
