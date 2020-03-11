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

""" S0 factory """

from qiskit.aqua.utils import CircuitFactory

# pylint: disable=invalid-name


class S0Factory(CircuitFactory):
    """ S0 factory """
    # pylint: disable=useless-super-delegation
    def __init__(self, num_target_qubits):
        super().__init__(num_target_qubits)

    def required_ancillas(self):
        """ required ancillas """
        if self.num_target_qubits == 1:
            return 0
        else:
            return max(1, self._num_target_qubits - 1)

    def required_ancillas_controlled(self):
        """ requires ancillas controlled """
        if self.num_target_qubits == 1:
            return 0
        else:
            return self._num_target_qubits

    def build(self, qc, q, q_ancillas=None, params=None):
        """ build """
        if self.num_target_qubits == 1:
            qc.z(q[0])
        else:
            for q_ in q:
                qc.x(q_)
            qc.x(q_ancillas[0])
            qc.h(q_ancillas[0])
            q_controls = [q[i] for i in range(len(q))]
            q_ancillas_ = [q_ancillas[i] for i in range(len(q_ancillas))]
            qc.mct(q_controls, q_ancillas_[0], q_ancillas_[1:])
            qc.h(q_ancillas[0])
            qc.x(q_ancillas[0])
            for q_ in q:
                qc.x(q_)
