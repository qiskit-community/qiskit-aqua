# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Operators (:mod:`qiskit.aqua.operators`)
========================================
Operators

.. currentmodule:: qiskit.aqua.operators

Operators
=========

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

   BaseOperator
   WeightedPauliOperator
   TPBGroupedWeightedPauliOperator
   MatrixOperator

Operator support
================

.. autosummary::
   :toctree: ../stubs/
   :nosignatures:

    evolution_instruction
    suzuki_expansion_slice_pauli_list
    pauli_measurement
    measure_pauli_z
    covariance
    row_echelon_F2
    kernel_F2
    commutator
    check_commutativity
    PauliGraph
    Z2Symmetries

"""
from qiskit.quantum_info import Pauli
from qiskit.extensions.standard import CnotGate, SGate, TGate, HGate, SwapGate

from qiskit.aqua.operators.legacy.common import (evolution_instruction, suzuki_expansion_slice_pauli_list, pauli_measurement,
                                                 measure_pauli_z, covariance, row_echelon_F2,
                                                 kernel_F2, commutator, check_commutativity)
from qiskit.aqua.operators.legacy.pauli_graph import PauliGraph
from qiskit.aqua.operators.legacy.base_operator import BaseOperator
from qiskit.aqua.operators.legacy.weighted_pauli_operator import WeightedPauliOperator, Z2Symmetries
from qiskit.aqua.operators.legacy.tpb_grouped_weighted_pauli_operator import TPBGroupedWeightedPauliOperator
from qiskit.aqua.operators.legacy.matrix_operator import MatrixOperator

# New Operators
from .operator_base import OperatorBase
from .op_kron import OpKron
from .op_composition import OpComposition
from .op_vec import OpVec
from .op_primitive import OpPrimitive
from .op_sum import OpSum
from .states import StateFn

# Paulis
X = OpPrimitive(Pauli.from_label('X'))
Y = OpPrimitive(Pauli.from_label('Y'))
Z = OpPrimitive(Pauli.from_label('Z'))
I = OpPrimitive(Pauli.from_label('I'))

# Clifford+T
CX = OpPrimitive(CnotGate())
S = OpPrimitive(SGate())
H = OpPrimitive(HGate())
T = OpPrimitive(TGate())
Swap = OpPrimitive(SwapGate())

Zero = StateFn('0')
One = StateFn('1')
Plus = H.compose(Zero)
Minus = H.compose(One)
# TODO figure out what to do about gate/pauli overlap, especially I and Id

__all__ = [
    'evolution_instruction',
    'suzuki_expansion_slice_pauli_list',
    'pauli_measurement',
    'measure_pauli_z',
    'covariance',
    'row_echelon_F2',
    'kernel_F2',
    'commutator',
    'check_commutativity',
    'PauliGraph',
    'BaseOperator',
    'WeightedPauliOperator',
    'Z2Symmetries',
    'TPBGroupedWeightedPauliOperator',
    'MatrixOperator',
    'OperatorBase'
    'OpPrimitive'
    'OpSum',
    'OpKron',
    'OpComposition',
    'OpVec'
]
