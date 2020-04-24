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
List Operators (:mod:`qiskit.aqua.operators.list_ops`)
==============================================================
List Operators are classes for storing and manipulating lists of Operators, State functions,
or Measurements, often including rules for lazy evaluation of the list members together at some
later point. For example, a ``SummedOp`` includes an addition rule, so once the Operators
are evaluated against some bitstring to produce a list of results, we know to add up that list to
produce the final result of the ``SummedOp``'s evaluation. While the combination function is
defined over classical values, it should be understood as the operation by which each Operators'
underlying function is combined to form the underlying Operator function of the ``ListOp``. In
this way, the ``ListOps`` are the basis for constructing large and sophisticated Operators,
State Functions, and Measurements in Aqua.

.. currentmodule:: qiskit.aqua.operators.list_ops

List Operators
===============

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   ListOp
   ComposedOp
   SummedOp
   TensoredOp

"""

from .list_op import ListOp
from .summed_op import SummedOp
from .composed_op import ComposedOp
from .tensored_op import TensoredOp

__all__ = ['ListOp',
           'SummedOp',
           'TensoredOp',
           'ComposedOp']
