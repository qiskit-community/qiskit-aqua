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
from qiskit.aqua.circuits.gates import mcry
from sympy.ntheory.multinomial import multinomial_coefficients
from itertools import product
import numpy as np


class PolynomialState(CircuitFactory):
    """
    Polynomial state preparation.
    For a polynomial p(x), a basis state |i> and a target qubit |0> this operator acts as:
        |i>|0> --> |i>( cos(scale * p(i))|0> + sin(scale * p(i))|1> )
    
    Let n be the number of qubits representing the state, d the degree of p(x) and q_i the qubits, 
    where q_0 is the least significant qubit. Then for
        x = \sum_{i=0}^{n-1} 2^{i}*q_i,
    we can write
        p(x) = \sum_{j=0}^{j=d} px[j]*(q_0 + 2*q_1 + ... + 2^{n-1}*q_n-1)^{j}.
    The expression above is used to obtain the list of controls and rotation angles for the circuit.
    """

    def __init__(self, px, num_state_qubits, eps=1): 
        """
        Constructor.
        Prepare an approximation to a state with amplitudes specified by a polynomial.
        Args:
            px (list): coefficients of the polynomial, px[i] is the coefficient of x^i
            num_target_qubits (int): number of qubits representing the state
            eps (float): accuracy of the approximation. The default is to rescale p(x) by 1.
        """
        super().__init__(num_state_qubits+1)

        if eps <= 0:
            raise ValueError('The accuracy should be positive.')

        # Store parameters
        self.num_state_qubits = num_state_qubits
        self.scale = np.sqrt(eps/2)
        self.px = self.scale*px
        self.degree = len(px) - 1

    @property
    def get_scale(self):
        return self.scale

    def required_ancillas(self):
        return max(1, self.degree - 1) 

    def required_ancillas_controlled(self):
        return max(1, self.degree) 

    def _get_controls(self):
        """
        The list of controls is the list of all monomials of the polynomial, where the qubits are the variables.
        """
        t = [0] * (self.num_state_qubits - 1) + [1]
        cdict = {tuple(t): 0}
        clist = list(product([0,1], repeat=self.num_state_qubits))
        index = 0
        while index < len(clist):
            tsum = 0
            i = clist[index]
            for j in i:
                tsum = tsum + j
            if tsum > self.degree:
                clist.remove(i)
            else:
                index = index+1
        clist.remove(tuple([0]*self.num_state_qubits))
        # For now set all angles to 0
        for i in clist:
            cdict[i] = 0
        return cdict
    
    def _get_thetas(self, cdict):
        """
        Compute the coefficient of each monomial. This will be the argument for the controlled y-rotation.
        """
        for j in range(1,len(self.px)):
            # List of multinomial coefficients
            mlist = multinomial_coefficients(self.num_state_qubits, j)
            # Add angles
            for m in mlist:
                temp_t = []
                powers = 1
                # Get controls
                for k in range(0, len(m)):
                    if m[k] > 0:
                        temp_t.append(1)
                        powers *= 2**(k*m[k])
                    else:
                        temp_t.append(0)
                temp_t = tuple(temp_t)
                # Add angle
                cdict[temp_t] += self.px[j]*mlist[m]*powers
        return cdict

    def build(self, qc, q, q_target, q_ancillas=None, reverse=0):
        """
        Args:
            qc : quantum circuit
            q : list of qubits (has to be same length as self.num_state_qubits)
            q_target : qubit to be rotated. The algorithm is successful when this qubit is in the |1> state
            q_ancillas : list of ancilla qubits (or None if none needed)
            reverse: if 1, apply with reversed list of qubits (i.e. q_n as q_0, q_n-1 as q_1, etc).
        """

        # Dictionary of controls for the rotation gates as a tuple and their respective angles
        cdict = self._get_controls() 
        cdict = self._get_thetas(cdict) 

        qc.ry(2*self.px[0], q_target)
        for c in cdict:
            q_controls = []
            if reverse == 1:
                for i in range(0,len(c)):
                    if c[i]>0:
                        q_controls.append(q[q.size-i-1])
            else:
                for i in range(0,len(c)):
                    if c[i]>0:
                        q_controls.append(q[i])
            # Apply controlled y-rotation
            if len(q_controls) > 1:
                qc.mcry(2*cdict[c], q_controls, q_target, q_ancillas)
            elif len(q_controls) == 1:
                qc.cu3(2*cdict[c], 0, 0, q_controls[0], q_target)
                    
                            
                    