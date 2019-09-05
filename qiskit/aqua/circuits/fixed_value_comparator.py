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

from qiskit.aqua.utils.circuit_factory import CircuitFactory
from qiskit.aqua.circuits.gates import logical_or  # pylint: disable=unused-import

# pylint: disable=invalid-name


class FixedValueComparator(CircuitFactory):
    """
    Fixed Value Comparator.

    Operator compares basis states |i>_n against a classically
    given fixed value L and flips a target qubit if i >= L (or < depending on parameters):

        |i>_n|0> --> |i>_n|1> if i >= L else |i>|0>

    Operator is based on two's complement implementation of binary
    subtraction but only uses carry bits and no actual result bits.
    If the most significant carry bit (= results bit) is 1, the "">="
    condition is True otherwise it is False.
    """

    def __init__(self, num_state_qubits, value, geq=True, i_state=None, i_target=None):
        """
        Constructor.

        Initializes the fixed value comparator

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
        super().__init__(num_state_qubits + 1)
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
        Returns: two's complement
        """

        twos_complement = pow(2, self.num_state_qubits) - int(np.ceil(self.value))
        twos_complement = '{0:b}'.format(twos_complement).rjust(self.num_state_qubits, '0')
        twos_complement = \
            [1 if twos_complement[i] == '1' else 0 for i in reversed(range(len(twos_complement)))]
        return twos_complement

    def build(self, qc, q, q_ancillas=None, params=None):

        # get parameters
        i_state = self.i_state
        i_target = self.i_target

        # get qubits
        q_result = q[i_target]
        q_state = [q[i] for i in i_state]

        if self.value <= 0:  # condition always satisfied for non-positive values
            if self._geq:  # otherwise the condition is never satisfied
                qc.x(q_result)
        # condition never satisfied for values larger than or equal to 2^n
        elif self.value < pow(2, self.num_state_qubits):

            if self.num_state_qubits > 1:

                tc = self._get_twos_complement()
                for i in range(self.num_state_qubits):
                    if i == 0:
                        if tc[i] == 1:
                            qc.cx(q_state[i], q_ancillas[i])
                    elif i < self.num_state_qubits-1:
                        if tc[i] == 1:
                            qc.OR([q_state[i], q_ancillas[i-1]], q_ancillas[i], None)
                        else:
                            qc.ccx(q_state[i], q_ancillas[i-1], q_ancillas[i])
                    else:
                        if tc[i] == 1:
                            qc.OR([q_state[i], q_ancillas[i-1]], q_result, None)
                        else:
                            qc.ccx(q_state[i], q_ancillas[i-1], q_result)

                # flip result bit if geq flag is false
                if not self._geq:
                    qc.x(q_result)

                # uncompute ancillas state
                for i in reversed(range(self.num_state_qubits-1)):
                    if i == 0:
                        if tc[i] == 1:
                            qc.cx(q_state[i], q_ancillas[i])
                    else:
                        if tc[i] == 1:
                            qc.OR([q_state[i], q_ancillas[i - 1]], q_ancillas[i], None)
                        else:
                            qc.ccx(q_state[i], q_ancillas[i - 1], q_ancillas[i])
            else:

                # num_state_qubits == 1 and value == 1:
                qc.cx(q_state[0], q_result)

                # flip result bit if geq flag is false
                if not self._geq:
                    qc.x(q_result)

        else:
            if not self._geq:  # otherwise the condition is never satisfied
                qc.x(q_result)
