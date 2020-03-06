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

"""Fixed Value Comparator."""

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.aqua.utils.circuit_factory import CircuitFactory
from qiskit.aqua.circuits.gates import logical_or  # pylint: disable=unused-import

# pylint: disable=invalid-name


class FixedValueComparator(QuantumCircuit, CircuitFactory):
    r"""
    Fixed Value Comparator.

    Operator compares basis states \|i>_n against a classically
    given fixed value L and flips a target qubit if i >= L (or < depending on parameters):

        \|i>_n\|0> --> \|i>_n\|1> if i >= L else \|i>\|0>

    Operator is based on two's complement implementation of binary
    subtraction but only uses carry bits and no actual result bits.
    If the most significant carry bit (= results bit) is 1, the ">="
    condition is True otherwise it is False.
    """

    def __init__(self, num_state_qubits, value, geq=True, i_state=None, i_target=None):
        """

        Args:
            num_state_qubits (int): number of state qubits, the target qubit comes on top of this
            value (int): fixed value to compare with
            geq (Optional(bool)): evaluate ">=" condition of "<" condition
            i_state (Optional(Union(list, numpy.ndarray))): indices of state qubits in
                given list of qubits / register,
                if None, i_state = list(range(num_state_qubits)) is used
            i_target (Optional(int)): index of target qubit in given list
                of qubits / register, if None, i_target = num_state_qubits is used
        """
        # state (num_state_qubits) + ancillas (num_state_qubits - 1) + compare qubit (1)
        qr_state = QuantumRegister(num_state_qubits, 'state')
        qr_result = QuantumRegister(1, 'result')
        super().__init__(qr_state, qr_result)
        if num_state_qubits > 1:
            qr_ancilla = QuantumRegister(num_state_qubits - 1, 'ancilla')
            self.add_register(qr_ancilla)
        else:
            qr_ancilla = None

        self._num_state_qubits = num_state_qubits
        self._value = value
        self._geq = geq

        # get indices
        self.i_state = None
        if i_state is not None:
            self.i_state = i_state
        else:
            self.i_state = list(range(num_state_qubits))

        self.i_target = None
        if i_target is not None:
            self.i_target = i_target
        else:
            self.i_target = num_state_qubits

        self._build(qr_state, qr_result, qr_ancilla)

    @property
    def num_state_qubits(self):
        """ returns num state qubits """
        return self._num_state_qubits

    @property
    def value(self):
        """ returns value """
        return self._value

    def required_ancillas(self):
        return self._num_state_qubits - 1

    def required_ancillas_controlled(self):
        return self._num_state_qubits - 1

    def _get_twos_complement(self):
        """
        Returns the 2's complement of value as array

        Returns:
             list: two's complement
        """

        twos_complement = pow(2, self.num_state_qubits) - int(np.ceil(self.value))
        twos_complement = '{0:b}'.format(twos_complement).rjust(self.num_state_qubits, '0')
        twos_complement = \
            [1 if twos_complement[i] == '1' else 0 for i in reversed(range(len(twos_complement)))]
        return twos_complement

    def _build(self,
               qr_state: QuantumRegister,
               qr_result: QuantumRegister,
               qr_ancilla: QuantumRegister) -> None:
        """Build the comparator circuit.

        Args:
            qr_state: The register containing the qubit state.
            qr_result: The register containing the single qubit, which will contain the result.
            qr_ancilla: The register containing the ancilla qubits.
        """

        if self.value <= 0:  # condition always satisfied for non-positive values
            if self._geq:  # otherwise the condition is never satisfied
                self.x(qr_result)
        # condition never satisfied for values larger than or equal to 2^n
        elif self.value < pow(2, self.num_state_qubits):

            if self.num_state_qubits > 1:

                tc = self._get_twos_complement()
                for i in range(self.num_state_qubits):
                    if i == 0:
                        if tc[i] == 1:
                            self.cx(qr_state[i], qr_ancilla[i])
                    elif i < self.num_state_qubits-1:
                        if tc[i] == 1:
                            self.OR([qr_state[i], qr_ancilla[i-1]], qr_ancilla[i], None)
                        else:
                            self.ccx(qr_state[i], qr_ancilla[i-1], qr_ancilla[i])
                    else:
                        if tc[i] == 1:
                            # OR needs the result argument as qubit not register, thus
                            # access the index [0]
                            self.OR([qr_state[i], qr_ancilla[i-1]], qr_result[0], None)
                        else:
                            self.ccx(qr_state[i], qr_ancilla[i-1], qr_result)

                # flip result bit if geq flag is false
                if not self._geq:
                    self.x(qr_result[0])

                # uncompute ancillas state
                for i in reversed(range(self.num_state_qubits-1)):
                    if i == 0:
                        if tc[i] == 1:
                            self.cx(qr_state[i], qr_ancilla[i])
                    else:
                        if tc[i] == 1:
                            self.OR([qr_state[i], qr_ancilla[i - 1]], qr_ancilla[i], None)
                        else:
                            self.ccx(qr_state[i], qr_ancilla[i - 1], qr_ancilla[i])
            else:

                # num_state_qubits == 1 and value == 1:
                self.cx(qr_state[0], qr_result)

                # flip result bit if geq flag is false
                if not self._geq:
                    self.x(qr_result)

        else:
            if not self._geq:  # otherwise the condition is never satisfied
                self.x(qr_result)

    def build(self, qc, q, q_ancillas=None, params=None):
        instr = self.to_instruction()
        qr = [qi for qi in q]  # pylint:disable=unnecessary-comprehension
        if q_ancillas:
            qr += [qi for qi in q_ancillas]  # pylint:disable=unnecessary-comprehension
        qc.append(instr, qr)
