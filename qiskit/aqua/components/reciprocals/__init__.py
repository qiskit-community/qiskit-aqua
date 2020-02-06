# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Reciprocals (:mod:`qiskit.aqua.components.reciprocals`)
===================================================================
A component for computing a reciprocal, mainly used by :class:`HHL` algorithm

.. currentmodule:: qiskit.aqua.components.reciprocals

Reciprocal Base Class
=====================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   Reciprocal

Reciprocals
===========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   LookupRotation
   LongDivision

"""

from .reciprocal import Reciprocal
from .lookup_rotation import LookupRotation
from .long_division import LongDivision

__all__ = ['LookupRotation', 'LongDivision', 'Reciprocal']
