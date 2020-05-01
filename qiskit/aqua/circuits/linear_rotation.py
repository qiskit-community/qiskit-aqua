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

"""Linearly-controlled X, Y or Z rotation."""

import numpy as np

from qiskit.aqua.utils import CircuitFactory
from qiskit.aqua.circuits.gates import cry  # pylint: disable=unused-import


class LinearRotation(CircuitFactory):
    """
    Linearly-controlled X, Y or Z rotation.
    For a register of state qubits |x> and a target qubit |0> this operator acts as:

        |x>|0> --> |x>( cos(slope * x + offset)|0> + sin(slope * x + offset)|1> )

    """

    def __init__(self, slope, offset, num_state_qubits, basis='Y', i_state=None, i_target=None):
        """
        Constructor.

        Construct linear rotation circuit factory
        Args:
            slope (float): slope of the controlled rotation
            offset (float): offset of the controlled rotation
            num_state_qubits (int): number of qubits representing the state
            basis (str): type of Pauli rotation ('X', 'Y', 'Z')
            i_state (Optional(Union(list, numpy.ndarray))): indices of the state qubits
                    (least significant to most significant)
            i_target (Optional(int)): index of target qubit
        Raises:
            ValueError: invalid input
        """

        super().__init__(num_state_qubits + 1)

        # store parameters
        self.num_control_qubits = num_state_qubits
        self.slope = slope
        self.offset = offset
        self.basis = basis

        if self.basis not in ['X', 'Y', 'Z']:
            raise ValueError('Basis must be X, Y or Z')

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

    def build(self, qc, q, q_ancillas=None, params=None):

        # get indices
        i_state = self.i_state
        i_target = self.i_target

        # apply linear rotation
        if not np.isclose(self.offset / 4 / np.pi % 1, 0):
            if self.basis == 'X':
                qc.rx(self.offset, q[i_target])
            elif self.basis == 'Y':
                qc.ry(self.offset, q[i_target])
            elif self.basis == 'Z':
                qc.rz(self.offset, q[i_target])
        for i, j in enumerate(i_state):
            theta = self.slope * pow(2, i)
            if not np.isclose(theta / 4 / np.pi % 1, 0):
                if self.basis == 'X':
                    qc.crx(self.slope * pow(2, i), q[j], q[i_target])
                elif self.basis == 'Y':
                    qc.cry(self.slope * pow(2, i), q[j], q[i_target])
                elif self.basis == 'Z':
                    qc.crz(self.slope * pow(2, i), q[j], q[i_target])
