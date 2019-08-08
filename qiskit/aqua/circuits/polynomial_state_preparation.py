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
from qiskit.aqua.utils import CircuitFactory
from qiskit.aqua.circuits.polynomial_rotation import PolynomialRotation
import numpy as np


class PolynomialStatePreparation(CircuitFactory):
    """
    Approximation to polynomial state preparation.
    For a polynomial p(x), a basis state |i> and a target qubit |0> this operator acts as:
        |i>|0> --> |i>( cos(scale * p(i))|0> + sin(scale * p(i))|1> )
    
    'scale' depends on the desired accouracy of the approximation. For an accuracy epsilon, 'scale' is a constant such that 
        sin(scale * p(i))/scale = p(i) + epsilon
    """

    def __init__(self, px, num_state_qubits, eps):
        """
        Constructor.
        Prepare an approximation to a state with amplitudes specified by a polynomial.
        Args:
            px (list): coefficients of the polynomial, px[i] is the coefficient of x^i
            num_state_qubits (int): number of qubits representing the state
            basis (str): type of Pauli rotation ('X', 'Y', 'Z')
            eps (float): accuracy of the approximation
        """
        super().__init__(num_state_qubits+1)

        # Store parameters
        self.num_state_qubits = num_state_qubits
        self.scale = np.sqrt(eps/2)
        self.px = self.scale * px
        self.degree = len(px) - 1

    def get_scale(self):
        """ returns the scaling """
        return self.scale

    def required_ancillas(self):
        return max(1, self.degree - 1) 

    def required_ancillas_controlled(self):
        return max(1, self.degree) 

    def build(self, qc, q, q_target, q_ancillas=None, reverse=0):
        """
        Args:
            qc : quantum circuit
            q : list of qubits (has to be same length as self.num_state_qubits)
            q_target : qubit to be rotated. The algorithm is successful when this qubit is in the |1> state
            q_ancillas : list of ancilla qubits (or None if none needed)
            reverse: if 1, apply with reversed list of qubits (i.e. q_n as q_0, q_n-1 as q_1, etc).
        """

        PolynomialRotation(self.px, self.num_state_qubits).build(qc, q, q_target, q_ancillas, reverse)
                    