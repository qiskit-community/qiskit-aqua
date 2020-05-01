# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Fixed Value Comparator."""

import warnings
import numpy as np

from qiskit.circuit.library import IntegerComparator
from qiskit.aqua.utils.circuit_factory import CircuitFactory


class FixedValueComparator(CircuitFactory):
    r"""*DEPRECATED.* Fixed Value Comparator

    .. deprecated:: 0.7.0
       Use Terra's qiskit.circuit.library.IntegerComparator instead.

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
        warnings.warn('The qiskit.aqua.circuits.FixedValueComparator object is deprecated and will '
                      'be removed no earlier than 3 months after the 0.7.0 release of Qiskit Aqua. '
                      'You should use qiskit.circuit.library.IntegerComparator instead.',
                      DeprecationWarning, stacklevel=2)

        super().__init__(num_state_qubits + 1)
        self._comparator_circuit = IntegerComparator(value=value,
                                                     num_state_qubits=num_state_qubits,
                                                     geq=geq)

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
        """ returns num state qubits """
        return self._comparator_circuit._num_state_qubits

    @property
    def value(self):
        """ returns value """
        return self._comparator_circuit._value

    def required_ancillas(self):
        return self.num_state_qubits - 1

    def required_ancillas_controlled(self):
        return self.num_state_qubits - 1

    def _get_twos_complement(self):
        """Returns the 2's complement of value as array

        Returns:
             list: two's complement
        """

        twos_complement = pow(2, self.num_state_qubits) - int(np.ceil(self.value))
        twos_complement = '{0:b}'.format(twos_complement).rjust(self.num_state_qubits, '0')
        twos_complement = \
            [1 if twos_complement[i] == '1' else 0 for i in reversed(range(len(twos_complement)))]
        return twos_complement

    def build(self, qc, q, q_ancillas=None, params=None):
        instr = self._comparator_circuit.to_instruction()
        qr = [q[i] for i in self.i_state] + [q[self.i_target]]
        if q_ancillas:
            # pylint:disable=unnecessary-comprehension
            qr += [qi for qi in q_ancillas[:self.required_ancillas()]]
        qc.append(instr, qr)
