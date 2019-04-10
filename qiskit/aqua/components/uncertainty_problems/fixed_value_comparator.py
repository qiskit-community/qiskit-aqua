# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from qiskit.aqua.utils.circuit_factory import CircuitFactory
import numpy as np


class FixedValueComparator(CircuitFactory):
    """
    Fixed Value Comparator.

    Operator compares basis states |i>_n against a classically given fixed value L and flips a target qubit if i >= L (or < depending on parameters):

        |i>_n|0> --> |i>_n|1> if i >= L else |i>|0>

    Operator is based on two's complement implementation of binary subtraction but only uses carry bits and no actual result bits.
    If the most significant carry bit (= results bit) is 1, the "">=" condition is True otherwise it is False.
    """

    def __init__(self, num_state_qubits, value, geq=True, i_state=None, i_target=None):
        """
        Initializes the fixed value comparator
        :param num_target_qubits: total number of target qubits (n-1 state qubits and 1 result qubit)
        :param value: fixed value to compare with
        :param geq: evaluate ">=" condition or "<" condition
        """
        super().__init__(num_state_qubits + 1)
        self._num_state_qubits = num_state_qubits
        self._value = value
        self._geq = geq

        # get indices
        self.i_state = None
        if i_state is not None:
            self.i_state = i_state
        else:
            self.i_state = range(num_state_qubits)

        self.i_target = None
        if i_target is not None:
            self.i_target = i_target
        else:
            self.i_target = num_state_qubits

    @property
    def num_state_qubits(self):
        return self._num_state_qubits

    @property
    def value(self):
        return self._value

    def required_ancillas(self):
        return self._num_state_qubits - 1

    def required_ancillas_controlled(self):
        return self._num_state_qubits - 1

    def _get_twos_complement(self):
        """
        Returns the 2's complement of value as array
        :return: two's complement
        """
        twos_complement = pow(2, self.num_state_qubits) - int(np.ceil(self.value))
        twos_complement = '{0:b}'.format(twos_complement).rjust(self.num_state_qubits, '0')
        twos_complement = [1 if twos_complement[i] == '1' else 0 for i in reversed(range(len(twos_complement)))]
        return twos_complement

    @staticmethod
    def _or(qc, a, b, c):
        """
        Applies or logical: c = a or b
        :param a: input qubit 1
        :param b: input qubit 2
        :param c: result qubit
        """
        qc.x(a)
        qc.x(b)
        qc.x(c)
        qc.ccx(a, b, c)
        qc.x(a)
        qc.x(b)

    def build(self, qc, q, q_ancillas=None, params=None):

        # get parameters
        i_state = self.i_state
        i_target = self.i_target
        if params is not None:
            uncompute = params.get('uncompute', True)
        else:
            uncompute = True

        # get qubits
        q_result = q[i_target]
        q_state = [q[i] for i in i_state]

        if self.value <= 0:  # condition always satisfied for non-positive values
            if self._geq:  # otherwise the condition is never satisfied
                qc.x(q_result)
        elif self.value < pow(2, self.num_state_qubits):  # condition never satisfied for values larger than or equal to 2^n

            tc = self._get_twos_complement()
            for i in range(self.num_state_qubits):
                if i == 0:
                    if tc[i] == 1:
                        qc.cx(q_state[i], q_ancillas[i])
                elif i < self.num_state_qubits-1:
                    if tc[i] == 1:
                        self._or(qc, q_state[i], q_ancillas[i-1], q_ancillas[i])
                    else:
                        qc.ccx(q_state[i], q_ancillas[i-1], q_ancillas[i])
                else:
                    if tc[i] == 1:
                        self._or(qc, q_state[i], q_ancillas[i-1], q_result)
                    else:
                        qc.ccx(q_state[i], q_ancillas[i-1], q_result)

            # flip result bit if geq flag is false
            if not self._geq:
                qc.x(q_result)

            # uncompute ancillas state
            if uncompute:
                for i in reversed(range(self.num_state_qubits-1)):
                    if i == 0:
                        if tc[i] == 1:
                            qc.cx(q_state[i], q_ancillas[i])
                    else:
                        if tc[i] == 1:
                            self._or(qc, q_state[i], q_ancillas[i - 1], q_ancillas[i])
                        else:
                            qc.ccx(q_state[i], q_ancillas[i - 1], q_ancillas[i])

        else:
            if not self._geq:  # otherwise the condition is never satisfied
                qc.x(q_result)
