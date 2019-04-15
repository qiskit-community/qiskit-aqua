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
from qiskit.aqua.utils import CircuitFactory
from qiskit.aqua.circuits.gates import cry
import numpy as np


class LinearYRotation(CircuitFactory):
    """
    Linearly-controlled Y rotation.
    For a register of state qubits |x> and a target qubit |0> this operator acts as:

        |x>|0> --> |x>( cos(slope * x + offset)|0> + sin(slope * x + offset)|1> )

    """

    def __init__(self, slope, offset, num_state_qubits, i_state=None, i_target=None):
        """
        Construct linear Y rotation circuit factory
        :param slope: slope of the controlled rotation
        :param offset: offset of the controlled rotation
        :param num_state_qubits: number of qubits representing the state
        :param i_state: indices of the state qubits (least significant to most significant)
        :param i_target: index of target qubit
        """

        super().__init__(num_state_qubits + 1)

        # store parameters
        self.num_control_qubits = num_state_qubits
        self.slope = slope
        self.offset = offset

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

    def build(self, qc, q, q_ancillas=None):

        # get indices
        i_state = self.i_state
        i_target = self.i_target

        # apply linear rotation
        if not np.isclose(self.offset / 4 / np.pi % 1, 0):
            qc.ry(self.offset, q[i_target])
        for i, j in enumerate(i_state):
            theta = self.slope * pow(2, i)
            if not np.isclose(theta / 4 / np.pi % 1, 0):
                qc.cry(self.slope * pow(2, i), q[j], q[i_target])
