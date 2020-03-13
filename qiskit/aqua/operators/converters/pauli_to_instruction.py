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

""" Expectation Algorithm Base """

import logging

from qiskit import QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.extensions.standard import XGate, YGate, ZGate, IGate

from qiskit.aqua.operators import OpPrimitive, OpVec
from .converter_base import ConverterBase

logger = logging.getLogger(__name__)
_pauli_to_gate_mapping = {'X': XGate(), 'Y': YGate(), 'Z': ZGate(), 'I': IGate()}


class PaulitoInstruction(ConverterBase):

    def __init__(self, traverse=True, delete_identities=False):
        self._traverse = traverse
        self._delete_identities = delete_identities

    def convert(self, operator):

        if isinstance(operator, Pauli):
            coeff = 1.0
        elif isinstance(operator, OpPrimitive) and isinstance(operator.primitive, Pauli):
            operator = operator.primitive
            coeff = operator.coeff
        # TODO allow parameterized OpVec to be returned to save circuit copying.
        elif isinstance(operator, OpVec) and self._traverse and \
                'Pauli' in operator.get_primitives():
            return operator.traverse(self.convert)
        else:
            raise TypeError('PauliToInstruction can only accept OperatorBase objects or '
                            'Paulis, not {}'.format(type(operator)))

        return OpPrimitive(self.convert_pauli(operator), coeff=coeff)

    def convert_pauli(self, pauli):
        # Note: Reversing endianness!!
        qc = QuantumCircuit(len(pauli))
        for q, p in enumerate(reversed(pauli.to_label())):
            gate = _pauli_to_gate_mapping[p]
            if not (self._delete_identities and p == 'I'):
                qc.append(gate, qargs=[q])
        return qc.to_instruction()
