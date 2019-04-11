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
from qiskit.aqua.circuits.fixed_value_comparator import FixedValueComparator as Comparator
from qiskit.aqua.circuits.linear_y_rotation import LinearYRotation as LinRY
import numpy as np


class PiecewiseLinearYRotation(CircuitFactory):
    """
    Piecewise-linearly-controlled Y rotation.
    For a piecewise linear (not necessarily continuous) function f(x), a state qubit register |x> and a target qubit |0>, this operator acts as

        |x>|0> --> |x> ( cos( f(x) )|0> + sin( f(x) )|1> )

    The function f(x) is defined through breakpoints, slopes and offsets as follows.
    Suppose the breakpoints { x_0, ..., x_J } are a subset of [0,  2^n-1], where n is the number of state qubits.
    Further on, denote the corresponding slopes and offsets by a_j, b_j respectively.
    Then f(x) is defined as:

        x < x_0            --> f(x) = 0
        x_j <= x < x_{j+1} --> f(x) = a_j * (x - x_j) + b_j

    where we implicitly assume x_{J+1} = 2^n.
    """

    def __init__(self, breakpoints, slopes, offsets, num_state_qubits, i_state=None, i_target=None):
        """
        Construct piecewise-linearly-controlled Y-rotation
        :param breakpoints: breakpoints to define piecewise-linear function
        :param slopes: slopes for different segments of piecewise-linear function
        :param offsets: offsets for different segments of piecewise-linear function
        :param num_state_qubits: number of qubits representing the state
        :param i_state: indices of qubits representing the state
        :param i_target: index of target qubit
        """

        super().__init__(num_state_qubits + 1)

        # store parameters
        self.num_state_qubits = num_state_qubits
        self.breakpoints = breakpoints
        self.slopes = slopes
        self.offsets = offsets

        # map slopes and offsets
        self.mapped_slopes = np.zeros(len(breakpoints))
        self.mapped_offsets = np.zeros(len(breakpoints))
        self.mapped_slopes[0] = self.slopes[0]
        self.mapped_offsets[0] = self.offsets[0] - self.slopes[0] * self.breakpoints[0]
        sum_mapped_slopes = 0
        sum_mapped_offsets = 0
        for i in range(1, len(breakpoints)):
            sum_mapped_slopes += self.mapped_slopes[i - 1]
            sum_mapped_offsets += self.mapped_offsets[i - 1]

            self.mapped_slopes[i] = self.slopes[i] - sum_mapped_slopes
            self.mapped_offsets[i] = self.offsets[i] - self.slopes[i] * self.breakpoints[i] - sum_mapped_offsets

        # check whether 0 is contained in breakpoints
        self.contains_zero_breakpoint = np.isclose(0, self.breakpoints[0])

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

    def evaluate(self, x):

        y = (x >= self.breakpoints[0]) * (x * self.mapped_slopes[0] + self.mapped_offsets[0])
        for i in range(1, len(self.breakpoints)):
            y = y + (x >= self.breakpoints[i]) * (x * self.mapped_slopes[i] + self.mapped_offsets[i])

        return y

    def required_ancillas(self):

        num_ancillas = self.num_state_qubits - 1 + len(self.breakpoints)
        if self.contains_zero_breakpoint:
            num_ancillas -= 1
        return num_ancillas

    def build(self, qc, q, q_ancillas=None, params=None):

        # get parameters
        i_state = self.i_state
        i_target = self.i_target
        if params is not None:
            uncompute = params.get('uncompute', True)
        else:
            uncompute = True

        # apply comparators and controlled linear rotations
        for i, bp in enumerate(self.breakpoints):

            if i == 0 and self.contains_zero_breakpoint:

                # apply rotation
                lin_ry = LinRY(self.mapped_slopes[i], self.mapped_offsets[i], self.num_state_qubits, i_state=i_state, i_target=i_target)
                lin_ry.build(qc, q)

            elif self.contains_zero_breakpoint:

                # apply comparator
                comp = Comparator(self.num_state_qubits, bp)
                q_ = [q[i] for i in range(self.num_state_qubits)]  # map register to list
                q_ = q_ + [q_ancillas[i - 1]]  # add ancilla as compare qubit
                q_ancillas_ = [q_ancillas[j] for j in range(i, len(q_ancillas))]  # take remaining ancillas as ancilla register (list)
                comp.build(qc, q_, q_ancillas_)

                # apply controlled rotation
                lin_ry = LinRY(self.mapped_slopes[i], self.mapped_offsets[i], self.num_state_qubits, i_state=i_state, i_target=i_target)
                lin_ry.build_controlled(qc, q, q_ancillas[i - 1], params={'use_basis_gates': False})

                # uncompute comparator
                if uncompute:
                    comp.build_inverse(qc, q_, q_ancillas_)

            else:

                # apply comparator
                comp = Comparator(self.num_state_qubits, bp)
                q_ = [q[i] for i in range(self.num_state_qubits)]  # map register to list
                q_ = q_ + [q_ancillas[i]]  # add ancilla as compare qubit
                q_ancillas_ = [q_ancillas[j] for j in range(i + 1, len(q_ancillas))]  # take remaining ancillas as ancilla register (list)
                comp.build(qc, q_, q_ancillas_)

                # apply controlled rotation
                lin_ry = LinRY(self.mapped_slopes[i], self.mapped_offsets[i], self.num_state_qubits, i_state=i_state, i_target=i_target)
                lin_ry.build_controlled(qc, q, q_ancillas[i], params={'use_basis_gates': False})

                # uncompute comparator
                if uncompute:
                    comp.build_inverse(qc, q_, q_ancillas_)
