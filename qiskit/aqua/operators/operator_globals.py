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
Operator Globals
"""

from qiskit.quantum_info import Pauli
from qiskit.extensions.standard import CXGate, SGate, TGate, HGate, SwapGate

from .operator_primitives import OpPrimitive
from .state_functions import StateFn

# Singletons

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
