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

""" Expectation Algorithm Base """

import logging
import numpy as np
from functools import partial, reduce

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.extensions.standard import XGate, YGate, ZGate, IdGate

from qiskit.aqua.operators import OpPrimitive
from .converter_base import ConverterBase

logger = logging.getLogger(__name__)
_pauli_to_gate_mapping = {'X': XGate(), 'Y': YGate(), 'Z': ZGate(), 'I': IdGate()}


class PaulitoInstruction(ConverterBase):

    def __init__(self, delete_ids=False):
        self._delete_ids = delete_ids

    def convert(self, pauli, traverse=False):

        if isinstance(operator, Pauli):
            pauli = operator
            coeff = 1.0
        elif hasattr(operator, 'primitive') and isinstance(operator.primitive, Pauli):
            pauli = operator.primitive
            coeff = operator.coeff
        # TODO allow parameterized OpVec to be returned to save circuit copying.
        elif isinstance(operator, OpVec) and self._traverse and 'Pauli' in operator.get_primitives():
            return operator.traverse(self.convert)
        else:
            raise TypeError('PauliChangeOfBasis can only accept OperatorBase objects or '
                            'Paulis, not {}'.format(type(operator)))

        return OpPrimitive(self.convert_pauli(operator), coeff=coeff)

    def convert_pauli(self, pauli):
        # Note: Reversing endian-ness!!
        qc = QuantumCircuit(len(pauli))
        for q, p in enumerate(reversed(pauli.to_label())):
            gate = _pauli_to_gate_mapping[p]
            if not (self._delete_ids and p == 'I'):
                qc.append(gate, qargs=[q])
        return qc.to_instruction()