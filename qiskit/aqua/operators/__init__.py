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

from .legacy.common import (evolution_instruction,
                            suzuki_expansion_slice_pauli_list,
                            pauli_measurement,
                            measure_pauli_z, covariance, row_echelon_F2,
                            kernel_F2, commutator, check_commutativity)
from .legacy import (LegacyBaseOperator, WeightedPauliOperator, Z2Symmetries,
                     TPBGroupedWeightedPauliOperator, MatrixOperator,
                     PauliGraph, op_converter)

# New Operators
from .operator_base import OperatorBase
from .primitive_operators import PrimitiveOp, PauliOp, MatrixOp, CircuitOp
from .state_functions import (StateFn, DictStateFn, VectorStateFn,
                              CircuitStateFn, OperatorStateFn)
from .combo_operators import ListOp, SummedOp, ComposedOp, TensoredOp
from .converters import (ConverterBase, PauliBasisChange, PauliToInstruction,
                         DictToCircuitSum, AbelianGrouper)
from .expectation_values import (ExpectationBase, PauliExpectation, MatrixExpectation,
                                 AerPauliExpectation)
from .circuit_samplers import CircuitSampler, LocalSimulatorSampler, IBMQSampler
from .evolutions import (EvolutionBase, EvolvedOp, PauliTrotterEvolution, TrotterizationBase,
                         Trotter, Suzuki, QDrift)

# Singletons
from .operator_globals import X, Y, Z, I, CX, S, H, T, Swap, Zero, One, Plus, Minus

__all__ = [
    # Common
    'evolution_instruction', 'suzuki_expansion_slice_pauli_list',
    'pauli_measurement', 'measure_pauli_z',
    'covariance', 'row_echelon_F2', 'kernel_F2', 'commutator', 'check_commutativity',
    # Legacy
    'PauliGraph', 'LegacyBaseOperator', 'WeightedPauliOperator',
    'Z2Symmetries', 'TPBGroupedWeightedPauliOperator',
    'MatrixOperator',
    # New
    'OperatorBase',
    'PrimitiveOp', 'PauliOp', 'MatrixOp', 'CircuitOp',
    'StateFn', 'DictStateFn', 'VectorStateFn', 'CircuitStateFn', 'OperatorStateFn',
    'ListOp', 'SummedOp', 'ComposedOp', 'TensoredOp',
    # Singletons
    'X', 'Y', 'Z', 'I', 'CX', 'S', 'H', 'T', 'Swap', 'Zero', 'One', 'Plus', 'Minus'
]
