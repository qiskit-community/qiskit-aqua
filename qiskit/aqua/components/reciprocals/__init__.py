# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
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
Reciprocals are components to invert a fixed-point number prepared in a quantum register.
They were designed to be used in the context of a larger algorithm such as
:class:`~qiskit.aqua.algorithms.HHL`.

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
