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

""" Weighted Pauli Operator """

import logging
import numpy as np
import copy

from qiskit import QuantumCircuit, Instruction
from qiskit.quantum_info import Pauli, Operator

from .operator_base import OperatorBase
from .op_sum import OpSum

logger = logging.getLogger(__name__)


class OpPrimitive(OperatorBase):
    """ Class for Wrapping Operator Primitives """

    def __init__(self, primitive, name=None, coeff=1.0):
        """
        Args:
            primtive (Gate, Pauli, [[complex]], np.ndarray, QuantumCircuit, Instruction): The operator primitive being
            wrapped.
            name (str, optional): the name of operator.
        """
        if isinstance(primitive, QuantumCircuit):
            primitive = primitive.to_instruction()
        elif isinstance(primitive, (list, np.ndarray)):
            primitive = Operator(primitive)
        self._primitive = primitive
        self._name = name
        self._coeff = coeff

    @property
    def primitive(self):
        return self._primitive

    @property
    def coeff(self):
        return self._coeff

    def add(self, other, inplace=True):
        """ Addition """
        if self.equals(other):
            return OpPrimitive(self.primitive, coeff=self.coeff + other.coeff)
        try:
            return self.primitive + other.primitive
        except(NotImplementedError):
            return OpSum([self.primitive, other.primitive])

    def neg(self):
        """ Negate """
        return self.mul(-1.0)

    def equals(self, other):
        """ Evaluate Equality """
        if not isinstance(self.primitive, type(other.primitive)) \
                or not self.coeff == other.coeff:
            return False
        return self.primitive == other.primitive
        # Will return NotImplementedError if not supported

    def mul(self, scalar):
        """ Scalar multiply """
        return OpPrimitive(self.primitive, coeff=self.coeff * scalar)

    # TODO change to *other to handle lists?
    def kron(self, other):
        """ Kron """
        # TODO accept primitives in addition to OpPrimitive?
        try:
            if not isinstance(self.primitive, type(other.primitive)):
                return OpKron([self.primitive, other.primitive])
            elif isinstance(self.primitive, Pauli):
                return OpPrimitive(self.primitive.kron(other.primitive), coeff=self.coeff * other.coeff)
                # TODO double check coeffs logic for paulis
            elif isinstance(self.primitive, Operator):
                return OpPrimitive(self.primitive.tensor(other.primitive), coeff=self.coeff * other.coeff)
            elif isinstance(self.primitive, Instruction):
                new_qc = QuantumCircuit(inst1.num_qubits+inst2.num_qubits)
                new_qc.append(inst1, new_qc.qubits[0:inst1.num_qubits])
                new_qc.append(inst2, new_qc.qubits[inst1.num_qubits:])
                # TODO fix because converting to dag just to append is nuts
                return OpPrimitive(new_qc.decompose().to_instruction(), coeff=self.coeff * other.coeff)
            else:
                return OpKron([self.primitive, other.primitive])
        except(NotImplementedError):
            return OpKron([self.primitive, other.primitive])

    def kronpower(self, other):
        """ Kron with Self Multiple Times """
        if not isinstance(other, int):
            raise TypeError('Kronpower can only take int arguments')
        temp = OpPrimitive(self.primitive, coeff=self.coeff)
        for i in range(other):
            temp = temp.kron(self)
        return temp

    def compose(self, other):
        """ Operator Composition (Circuit-style, left to right) """
        raise NotImplementedError

    def dot(self, other):
        """ Operator Composition (Linear algebra-style, right to left) """
        raise NotImplementedError

    def power(self, other):
        """ Compose with Self Multiple Times """
        raise NotImplementedError

    def __str__(self):
        """Overload str() """
        return str(self.primitive)

    def print_details(self):
        """ print details """
        raise NotImplementedError


X = OpPrimitive(Pauli.from_label('X'))
Y = OpPrimitive(Pauli.from_label('Y'))
Z = OpPrimitive(Pauli.from_label('Z'))
I = OpPrimitive(Pauli.from_label('I'))
