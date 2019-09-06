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

"""Piecewise-linearly-controlled rotation."""

import numpy as np

from qiskit.aqua.utils import CircuitFactory
from qiskit.aqua.circuits.fixed_value_comparator import FixedValueComparator as Comparator
from qiskit.aqua.circuits.linear_rotation import LinearRotation as LinR

# pylint: disable=invalid-name


class PiecewiseLinearRotation(CircuitFactory):
    """
    Piecewise-linearly-controlled rotation.
    For a piecewise linear (not necessarily continuous) function f(x).
    The function f(x) is defined through breakpoints, slopes and offsets as follows.
    Suppose the breakpoints { x_0, ..., x_J } are a subset of [0,  2^n-1], where
    n is the number of state qubits.
    Further on, denote the corresponding slopes and offsets by a_j, b_j respectively.
    Then f(x) is defined as:

        x < x_0            --> f(x) = 0
        x_j <= x < x_{j+1} --> f(x) = a_j * (x - x_j) + b_j

    where we implicitly assume x_{J+1} = 2^n.
    """

    def __init__(self, breakpoints, slopes, offsets, num_state_qubits,
                 basis='Y', i_state=None, i_target=None):
        """
        Constructor.

        Construct piecewise-linearly-controlled Y-rotation.

        Args:
            breakpoints (Union(list, numpy.ndarray)): breakpoints to define
                                                        piecewise-linear function
            slopes (Union(list, numpy.ndarray)): slopes for different segments of
                                                        piecewise-linear function
            offsets (Union(list, numpy.ndarray)): offsets for different segments of
                                                        piecewise-linear function
            num_state_qubits (int): number of qubits representing the state
            basis (Optional(str)): type of Pauli rotation ('X', 'Y', 'Z')
            i_state (Optional(Union(list, numpy.ndarray))): indices of qubits representing
                            the state, set to range(num_state_qubits) if None
            i_target (Optional(int)): index of target qubit, set to num_state_qubits if None
        """

        super().__init__(num_state_qubits + 1)

        # store parameters
        self.num_state_qubits = num_state_qubits
        self.breakpoints = breakpoints
        self.slopes = slopes
        self.offsets = offsets
        self.basis = basis

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
            self.mapped_offsets[i] = \
                self.offsets[i] - self.slopes[i] * self.breakpoints[i] - sum_mapped_offsets

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
        """
        Classically evaluate the piecewise linear rotation
        Args:
            x (float): value to be evaluated at
        Returns:
            float: value of piecewise linear function at x
        """

        y = (x >= self.breakpoints[0]) * (x * self.mapped_slopes[0] + self.mapped_offsets[0])
        for i in range(1, len(self.breakpoints)):
            y = y + (x >= self.breakpoints[i]) * (x * self.mapped_slopes[i] +
                                                  self.mapped_offsets[i])

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

        # apply comparators and controlled linear rotations
        for i, bp in enumerate(self.breakpoints):

            if i == 0 and self.contains_zero_breakpoint:

                # apply rotation
                lin_r = LinR(self.mapped_slopes[i], self.mapped_offsets[i],
                             self.num_state_qubits, basis=self.basis,
                             i_state=i_state, i_target=i_target)
                lin_r.build(qc, q)

            elif self.contains_zero_breakpoint:

                # apply comparator
                comp = Comparator(self.num_state_qubits, bp)
                q_ = [q[i] for i in range(self.num_state_qubits)]  # map register to list
                q_ = q_ + [q_ancillas[i - 1]]  # add ancilla as compare qubit
                # take remaining ancillas as ancilla register (list)
                q_ancillas_ = [q_ancillas[j] for j in range(i, len(q_ancillas))]
                comp.build(qc, q_, q_ancillas_)

                # apply controlled rotation
                lin_r = LinR(self.mapped_slopes[i], self.mapped_offsets[i],
                             self.num_state_qubits, basis=self.basis,
                             i_state=i_state, i_target=i_target)
                lin_r.build_controlled(qc, q, q_ancillas[i - 1], use_basis_gates=False)

                # uncompute comparator
                comp.build_inverse(qc, q_, q_ancillas_)

            else:

                # apply comparator
                comp = Comparator(self.num_state_qubits, bp)
                q_ = [q[i] for i in range(self.num_state_qubits)]  # map register to list
                q_ = q_ + [q_ancillas[i]]  # add ancilla as compare qubit
                # take remaining ancillas as ancilla register (list)
                q_ancillas_ = [q_ancillas[j] for j in range(i + 1, len(q_ancillas))]
                comp.build(qc, q_, q_ancillas_)

                # apply controlled rotation
                lin_r = LinR(self.mapped_slopes[i],
                             self.mapped_offsets[i], self.num_state_qubits, basis=self.basis,
                             i_state=i_state, i_target=i_target)
                lin_r.build_controlled(qc, q, q_ancillas[i], use_basis_gates=False)

                # uncompute comparator
                comp.build_inverse(qc, q_, q_ancillas_)
