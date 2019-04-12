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
from qiskit.aqua.circuits.gates import mct
import numpy as np


class WeightedSumOperator(CircuitFactory):
    """
    Adds q^T * w to separate register for non-negative integer weights w
    """

    def __init__(self, num_state_qubits, weights):

        self._weights = weights

        # check weights
        for i, w in enumerate(weights):
            if not np.isclose(w, np.round(w)):
                print('Warning: non-integer weights are rounded to the nearest integer! (%s, %s)' % (i, w))

        self._num_state_qubits = num_state_qubits
        self._num_sum_qubits = self.get_required_sum_qubits(weights)
        self._num_carry_qubits = self.num_sum_qubits - 1

        num_target_qubits = num_state_qubits + self.num_sum_qubits
        super().__init__(num_target_qubits)

        self.i_state = list(range(num_state_qubits))
        self.i_sum = list(range(max(self.i_state)+1, max(self.i_state)+self.num_sum_qubits+1))

    @staticmethod
    def get_required_sum_qubits(weights):
        return int(np.floor(np.log2(sum(weights))) + 1)

    @property
    def weights(self):
        return self._weights

    @property
    def num_state_qubits(self):
        return self._num_state_qubits

    @property
    def num_sum_qubits(self):
        return self._num_sum_qubits

    @property
    def num_carry_qubits(self):
        return self._num_carry_qubits

    def required_ancillas(self):
        # includes one ancilla qubit for 3-controlled not gates
        return self.num_carry_qubits + 1  # TODO: validate when the +1 is needed and make a case distinction

    def required_ancillas_controlled(self):
        return self.required_ancillas()

    def build(self, qc, q, q_ancillas=None, params=None):

        # get indices for state and target qubits
        i_state = self.i_state
        i_sum = self.i_sum

        # set indices for sum and carry qubits (from ancilla register)
        i_carry = range(self.num_carry_qubits)
        i_control = self.num_carry_qubits

        # loop over state qubits and corresponding weights
        for i, w in enumerate(self.weights):

            # only act if non-trivial weight
            if w > 0:

                # get state control qubit
                q_state = q[i_state[i]]

                # get bit representation of current weight
                w_bits = "{0:b}".format(int(w)).rjust(self.num_sum_qubits, '0')[::-1]

                # loop over bits of current weight and add them to sum and carry registers
                for j, b in enumerate(w_bits):
                    if b == '1':
                        if self.num_sum_qubits == 1:
                            qc.cx(q_state, q[i_sum[j]])
                        elif j == 0:
                            # compute (q_sum[0] + 1) into (q_sum[0], q_carry[0]) - controlled by q_state[i]
                            qc.ccx(q_state, q[i_sum[j]], q_ancillas[i_carry[j]])
                            qc.cx(q_state, q[i_sum[j]])
                        elif j == self.num_sum_qubits-1:
                            # compute (q_sum[j] + q_carry[j-1] + 1) into (q_sum[j]) - controlled by q_state[i] / last qubit, no carry needed by construction
                            qc.cx(q_state, q[i_sum[j]])
                            qc.ccx(q_state, q_ancillas[i_carry[j - 1]], q[i_sum[j]])
                        else:
                            # compute (q_sum[j] + q_carry[j-1] + 1) into (q_sum[j], q_carry[j]) - controlled by q_state[i]
                            qc.x(q[i_sum[j]])
                            qc.x(q_ancillas[i_carry[j-1]])
                            # multi_toffoli_q(qc, q_controls=[q_state, q[i_sum[j]], q_ancillas[i_carry[j-1]]], q_target=q_ancillas[i_carry[j]], q_ancillas=q_ancillas[i_control])
                            qc.mct([q_state, q[i_sum[j]], q_ancillas[i_carry[j-1]]], q_ancillas[i_carry[j]], [q_ancillas[i_control]])
                            qc.cx(q_state, q_ancillas[i_carry[j]])
                            qc.x(q[i_sum[j]])
                            qc.x(q_ancillas[i_carry[j-1]])
                            qc.cx(q_state, q[i_sum[j]])
                            qc.ccx(q_state, q_ancillas[i_carry[j-1]], q[i_sum[j]])
                    else:
                        if self.num_sum_qubits == 1:
                            pass  # nothing to do, since nothing to add
                        elif j == 0:
                            pass  # nothing to do, since nothing to add
                        elif j == self.num_sum_qubits-1:
                            # compute (q_sum[j] + q_carry[j-1]) into (q_sum[j]) - controlled by q_state[i] / last qubit, no carry needed by construction
                            qc.ccx(q_state, q_ancillas[i_carry[j - 1]], q[i_sum[j]])
                        else:
                            # compute (q_sum[j] + q_carry[j-1]) into (q_sum[j], q_carry[j]) - controlled by q_state[i]
                            # multi_toffoli_q(qc, q_controls=[q_state, q[i_sum[j]], q_ancillas[i_carry[j-1]]], q_target=q_ancillas[i_carry[j]], q_ancillas=q_ancillas[i_control])
                            qc.mct([q_state, q[i_sum[j]], q_ancillas[i_carry[j-1]]], q_ancillas[i_carry[j]], [q_ancillas[i_control]])
                            qc.ccx(q_state, q_ancillas[i_carry[j-1]], q[i_sum[j]])

                # uncompute carry qubits
                for j in reversed(range(len(w_bits))):
                    b = w_bits[j]
                    if b == '1':
                        if self.num_sum_qubits == 1:
                            pass
                        elif j == 0:
                            qc.x(q[i_sum[j]])
                            qc.ccx(q_state, q[i_sum[j]], q_ancillas[i_carry[j]])
                            qc.x(q[i_sum[j]])
                        elif j == self.num_sum_qubits - 1:
                            pass
                        else:
                            qc.x(q_ancillas[i_carry[j - 1]])
                            # multi_toffoli_q(qc, q_controls=[q_state, q[i_sum[j]], q_ancillas[i_carry[j - 1]]], q_target=q_ancillas[i_carry[j]], q_ancillas=q_ancillas[i_control])
                            qc.mct([q_state, q[i_sum[j]], q_ancillas[i_carry[j - 1]]], q_ancillas[i_carry[j]], [q_ancillas[i_control]])
                            qc.cx(q_state, q_ancillas[i_carry[j]])
                            qc.x(q_ancillas[i_carry[j - 1]])
                    else:
                        if self.num_sum_qubits == 1:
                            pass
                        elif j == 0:
                            pass
                        elif j == self.num_sum_qubits - 1:
                            pass
                        else:
                            # compute (q_sum[j] + q_carry[j-1]) into (q_sum[j], q_carry[j]) - controlled by q_state[i]
                            qc.x(q[i_sum[j]])
                            # multi_toffoli_q(qc, q_controls=[q_state, q[i_sum[j]], q_ancillas[i_carry[j - 1]]], q_target=q_ancillas[i_carry[j]], q_ancillas=q_ancillas[i_control])
                            qc.mct([q_state, q[i_sum[j]], q_ancillas[i_carry[j - 1]]], q_ancillas[i_carry[j]], [q_ancillas[i_control]])
                            qc.x(q[i_sum[j]])
