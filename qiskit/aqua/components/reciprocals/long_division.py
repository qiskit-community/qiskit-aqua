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


"""The Long Division Rotation for Reciprocals.

It finds the reciprocal with long division method and rotates the ancillary
qubit by C/lambda. This is a first order approximation of arcsin(C/lambda).
"""

from typing import Optional
import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.aqua.components.reciprocals import Reciprocal

# pylint: disable=invalid-name


class LongDivision(Reciprocal):

    """
    The Long Division Rotation for Reciprocals.

    This method calculates inverse of eigenvalues using binary long division and performs the
    corresponding rotation. Long division is implemented as a sequence of subtraction (utilizing
    ripple carry adder module) and bit shifting. The method allows for adjusting of the reciprocal
    precision by changing number of iterations. The method was optimized for register conventions
    used in HHL algorithm (i.e. eigenvalues rescaled to values between 0 and 1).

    The rotation value is always scaled down additionally to the normal scale parameter by 0.5 to
    get the angle into the linear part of the arcsin(x).

    It finds the reciprocal with long division method and rotates the ancillary
    qubit by C/lambda. This is a first order approximation of arcsin(C/lambda).
    """

    def __init__(
            self,
            scale: float = 0,
            precision: Optional[int] = None,
            negative_evals: bool = False,
            evo_time: Optional[float] = None,
            lambda_min: Optional[float] = None) -> None:
        r"""
        Args:
            scale: The scale of rotation angle, corresponds to HHL constant C. This parameter is
                used to scale the reciprocals such that for a scale C, the rotation is performed
                by an angle :math:`\arcsin{\frac{C}{\lambda}}`. If neither the `scale` nor the
                `evo_time` and `lambda_min` parameters are specified, the smallest resolvable
                Eigenvalue is used.
            precision: Number of qubits that defines the precision of long division. The parameter
                sets minimum desired bit precision for the reciprocal. Due to shifting some of
                reciprocals, however, are effectively estimated with higher than this minimum
                specified precision.
            negative_evals: Indicate if negative eigenvalues need to be handled
            evo_time: The evolution time.  This parameter scales the Eigenvalues in the phase
                estimation onto the range (0,1] ( (-0.5,0.5] for negative Eigenvalues ).
            lambda_min: The smallest expected eigenvalue
        """

        super().__init__()
        self._negative_evals = negative_evals
        self._scale = scale
        self._precision = precision
        self._evo_time = evo_time
        self._lambda_min = lambda_min
        self._circuit = None
        self._ev = None
        self._rec = None
        self._anc = None
        self._reg_size = 0
        self._neg_offset = 0
        self._n = 0
        self._num_ancillae = None
        self._a = None
        self._b0 = None
        self._anc1 = None
        self._z = None
        self._c = None

    def sv_to_resvec(self, statevector, num_q):
        half = int(len(statevector) / 2)
        sv_good = statevector[half:]
        vec = np.array([])
        for i in range(2 ** num_q):
            vec = np.append(vec, sum(x for x in sv_good[i::2 ** num_q]))
        return vec

    def _ld_circuit(self):

        def subtract(a, b, b0, c, z, r, rj, n):
            qc = QuantumCircuit(a, b0, b, c, z, r)
            qc2 = QuantumCircuit(a, b0, b, c, z, r)

            def subtract_in(qc, a, b, b0, c, z, r, n):  # pylint: disable=unused-argument
                """subtraction realized with ripple carry adder"""

                def maj(p, a, b, c):
                    p.cx(c, b)
                    p.cx(c, a)
                    p.ccx(a, b, c)

                def uma(p, a, b, c):
                    p.ccx(a, b, c)
                    p.cx(c, a)
                    p.cx(a, b)

                for i in range(n):
                    qc.x(a[i])
                maj(qc, c[0], a[0], b[n - 2])

                for i in range(n - 2):
                    maj(qc, b[n - 2 - i + self._neg_offset],
                        a[i + 1], b[n - 3 - i + self._neg_offset])

                maj(qc, b[self._neg_offset + 0], a[n - 1], b0[0])
                qc.cx(a[n - 1], z[0])
                uma(qc, b[self._neg_offset + 0], a[n - 1], b0[0])

                for i in range(2, n):
                    uma(qc, b[self._neg_offset + i - 1], a[n - i], b[self._neg_offset + i - 2])

                uma(qc, c[0], a[0], b[n - 2 + self._neg_offset])

                for i in range(n):
                    qc.x(a[i])

                qc.x(z[0])

            def u_maj(p, a, b, c, r):
                p.ccx(c, r, b)
                p.ccx(c, r, a)
                p.mct([r, a, b], c, None, mode='noancilla')

            def u_uma(p, a, b, c, r):
                p.mct([r, a, b], c, None, mode='noancilla')
                p.ccx(c, r, a)
                p.ccx(a, r, b)

            def unsubtract(qc2, a, b, b0, c, z, r, n):
                """controlled inverse subtraction to uncompute the registers(when
                the result of the subtraction is negative)"""

                for i in range(n):
                    qc2.cx(r, a[i])
                u_maj(qc2, c[0], a[0], b[n - 2], r)

                for i in range(n - 2):
                    u_maj(qc2, b[n - 2 - i + self._neg_offset],
                          a[i + 1], b[n - 3 - i + self._neg_offset], r)

                u_maj(qc2, b[self._neg_offset + 0], a[n - 1], b0[0], r)
                qc2.ccx(a[n - 1], r, z[0])
                u_uma(qc2, b[self._neg_offset + 0], a[n - 1], b0[0], r)

                for i in range(2, n):
                    u_uma(qc2, b[self._neg_offset + i - 1],
                          a[n - i], b[self._neg_offset + i - 2], r)

                u_uma(qc2, c[0], a[0], b[n - 2 + self._neg_offset], r)

                for i in range(n):
                    qc2.cx(r, a[i])

                un_qc = qc2.mirror()
                un_qc.cx(r, z[0])
                return un_qc

            # assembling circuit for controlled subtraction
            subtract_in(qc, a, b, b0, c, z, r[rj], n)
            qc.x(a[n - 1])
            qc.cx(a[n - 1], r[rj])
            qc.x(a[n - 1])

            qc.x(r[rj])
            qc += unsubtract(qc2, a, b, b0, c, z, r[rj], n)
            qc.x(r[rj])

            return qc

        def shift_to_one(qc, b, anc, n):
            """controlled bit shifting for the initial alignment of the most
            significant bits """

            for i in range(n - 2):            # set all the anc1 qubits to 1
                qc.x(anc[i])

            for j2 in range(n - 2):           # if msb is 1, change ancilla j2 to 0
                qc.cx(b[0 + self._neg_offset], anc[j2])
                for i in np.arange(0, n - 2):
                    i = int(i)              # which activates shifting with the 2 Toffoli gates
                    qc.ccx(anc[j2], b[i + 1 + self._neg_offset], b[i + self._neg_offset])
                    qc.ccx(anc[j2], b[i + self._neg_offset], b[i + 1 + self._neg_offset])

            for i in range(n - 2):            # negate all the ancilla
                qc.x(anc[i])

        def shift_one_left(qc, b, n):
            for i in np.arange(n - 1, 0, -1):
                i = int(i)
                qc.cx(b[i - 1], b[i])
                qc.cx(b[i], b[i - 1])

        def shift_one_leftc(qc, b, ctrl, n):
            for i in np.arange(n - 2, 0, -1):
                i = int(i)
                qc.ccx(ctrl, b[i - 1], b[i])
                qc.ccx(ctrl, b[i], b[i - 1])
            return qc

        def shift_one_rightc(qc, b, ctrl, n):
            for i in np.arange(0, n - 1):
                i = int(i)
                qc.ccx(ctrl, b[n - 2 - i + self._neg_offset], b[n - 1 - i + self._neg_offset])
                qc.ccx(ctrl, b[n - 1 - i + self._neg_offset], b[n - 2 - i + self._neg_offset])

        # executing long division:
        self._circuit.x(self._a[self._n - 2])
        # initial alignment of most significant bits
        shift_to_one(self._circuit, self._ev, self._anc1, self._n)

        for rj in range(self._precision):  # iterated subtraction and shifting
            self._circuit += subtract(self._a, self._ev, self._b0, self._c,
                                      self._z, self._rec, rj, self._n)
            shift_one_left(self._circuit, self._a, self._n)

        for ish in range(self._n - 2):  # unshifting due to initial alignment
            shift_one_leftc(self._circuit, self._rec, self._anc1[ish],
                            self._precision + self._num_ancillae)
            self._circuit.x(self._anc1[ish])
            shift_one_rightc(self._circuit, self._ev, self._anc1[ish], self._num_ancillae)
            self._circuit.x(self._anc1[ish])

    def _rotation(self):
        qc = self._circuit
        rec_reg = self._rec
        ancilla = self._anc

        if self._negative_evals:
            for i in range(0, self._precision + self._num_ancillae):
                qc.cu3(self._scale * 2 ** (-i), 0, 0, rec_reg[i], ancilla)
            qc.cu3(2 * np.pi, 0, 0, self._ev[0], ancilla)  # correcting the sign
        else:
            for i in range(0, self._precision + self._num_ancillae):
                qc.cu3(self._scale * 2 ** (-i), 0, 0, rec_reg[i], ancilla)

        self._circuit = qc
        self._rec = rec_reg
        self._anc = ancilla

    def construct_circuit(self, mode, register=None, circuit=None):
        """Construct the Long Division Rotation circuit.

        Args:
            mode (str): construction mode, 'matrix' not supported
            register (QuantumRegister): input register, typically output register of Eigenvalues
            circuit (QuantumCircuit): Quantum Circuit or None
        Returns:
            QuantumCircuit: containing the Long Division Rotation circuit.
        Raises:
            NotImplementedError: mode not supported
        """

        if mode == 'matrix':
            raise NotImplementedError('The matrix mode is not supported.')
        self._ev = register

        if self._scale == 0:
            self._scale = 2**-len(register)

        if self._negative_evals:
            self._neg_offset = 1

        self._num_ancillae = len(self._ev) - self._neg_offset
        if self._num_ancillae < 3:
            self._num_ancillae = 3
        if self._negative_evals is True:
            if self._num_ancillae < 4:
                self._num_ancillae = 4

        self._n = self._num_ancillae + 1

        if self._precision is None:
            self._precision = self._num_ancillae

        self._a = QuantumRegister(self._n, 'one')  # register storing 1
        self._b0 = QuantumRegister(1, 'b0')  # extension of b - required by subtraction
        # ancilla for the initial shifting
        self._anc1 = QuantumRegister(self._num_ancillae - 1, 'algn_anc')
        self._z = QuantumRegister(1, 'z')  # subtraction overflow
        self._c = QuantumRegister(1, 'c')  # carry
        # reciprocal result
        self._rec = QuantumRegister(self._precision + self._num_ancillae, 'res')
        self._anc = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(self._a, self._b0, self._ev, self._anc1, self._c,
                            self._z, self._rec, self._anc)

        self._circuit = qc
        self._ld_circuit()
        self._rotation()

        return self._circuit
