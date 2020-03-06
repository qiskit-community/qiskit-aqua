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

   LegacyBaseOperator
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

from qiskit.aqua.operators.legacy.common import (evolution_instruction, suzuki_expansion_slice_pauli_list, pauli_measurement,
                                                 measure_pauli_z, covariance, row_echelon_F2,
                                                 kernel_F2, commutator, check_commutativity)
from qiskit.aqua.operators.legacy import (LegacyBaseOperator, WeightedPauliOperator, Z2Symmetries,
                                          TPBGroupedWeightedPauliOperator, MatrixOperator, PauliGraph)

# New Operators
from .operator_base import OperatorBase

from qiskit.aqua.operators.operator_primitives import OpPrimitive, OpPauli, OpMatrix, OpCircuit
from qiskit.aqua.operators.state_functions import StateFn, StateFnDict, StateFnVector, StateFnCircuit, StateFnOperator
from qiskit.aqua.operators.operator_combos import OpVec, OpSum, OpComposition, OpKron

from qiskit.quantum_info import Pauli
from qiskit.extensions.standard import CXGate, SGate, TGate, HGate, SwapGate

# Paulis
X = OpPrimitive(Pauli.from_label('X'))
Y = OpPrimitive(Pauli.from_label('Y'))
Z = OpPrimitive(Pauli.from_label('Z'))
I = OpPrimitive(Pauli.from_label('I'))

# Clifford+T
CX = OpPrimitive(CXGate())
S = OpPrimitive(SGate())
H = OpPrimitive(HGate())
T = OpPrimitive(TGate())
Swap = OpPrimitive(SwapGate())

Zero = StateFn('0')
One = StateFn('1')
Plus = H.compose(Zero)
Minus = H.compose(One)

from qiskit.aqua.operators.converters import (ConverterBase, PauliChangeOfBasis, PaulitoInstruction, ToMatrixOp,
                                              DicttoCircuitSum)
from qiskit.aqua.operators.expectation_values import (ExpectationBase, PauliExpectation, MatrixExpectation,
                                                      AerPauliExpectation)
from qiskit.aqua.operators.circuit_samplers import CircuitSampler, LocalSimulatorSampler, IBMQSampler


__all__ = [
    # Common
    'evolution_instruction', 'suzuki_expansion_slice_pauli_list', 'pauli_measurement', 'measure_pauli_z',
    'covariance', 'row_echelon_F2', 'kernel_F2', 'commutator', 'check_commutativity',
    # Legacy
    'PauliGraph', 'LegacyBaseOperator', 'WeightedPauliOperator', 'Z2Symmetries', 'TPBGroupedWeightedPauliOperator',
    'MatrixOperator',
    # New
    'OperatorBase'
    'OpPrimitive', 'OpPauli', 'OpMatrix', 'OpCircuit',
    'StateFn', 'StateFnDict', 'StateFnVector', 'StateFnCircuit', 'StateFnOperator',
    'OpVec', 'OpSum', 'OpComposition', 'OpKron',
    # Singletons
    'X', 'Y', 'Z', 'I', 'CX', 'S', 'H', 'T', 'Swap', 'Zero', 'One', 'Plus', 'Minus'
]
