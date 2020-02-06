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

""" SPsi0Factory """

from qiskit.aqua.utils import CircuitFactory


class SPsi0Factory(CircuitFactory):
    """ SPsi0Factory """
    def __init__(self, num_target_qubits, i_objective):
        super().__init__(num_target_qubits)
        self.i_objective = i_objective

    def required_ancillas(self):
        if self.num_target_qubits == 1:
            return 0
        else:
            return 1

    def required_ancillas_controlled(self):
        if self.num_target_qubits == 1:
            return 0
        else:
            return 1

    def build(self, qc, q, q_ancillas=None, params=None):
        if self.num_target_qubits == 1:
            qc.z(q[0])
        else:
            qc.x(q_ancillas[0])
            qc.h(q_ancillas[0])
            qc.x(q[self.i_objective])
            qc.cx(q[self.i_objective], q_ancillas[0])
            qc.x(q[self.i_objective])
            qc.h(q_ancillas[0])
            qc.x(q_ancillas[0])
