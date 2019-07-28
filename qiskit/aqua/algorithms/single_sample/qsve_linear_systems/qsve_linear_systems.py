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

# Imports
from copy import deepcopy
import numpy as np
from qiskit.aqua.components.qsve import QSVE
from qiskit.aqua.circuits.gates.multi_control_rotation_gates import mcry
from qiskit import QuantumRegister


class LinearSystemSolverQSVE:
    """Quantum algorithm for solving linear systems of equations based on quantum singular value estimation (QSVE)."""
    def __init__(self, Amatrix, bvector, precision=3, cval=0.5):
        """Initializes a LinearSystemSolver.

        Args:
            Amatrix : numpy.ndarray
                Matrix in the linear system Ax = b.

            bvector : numpy.ndarray
                Vector in the linear system Ax = b.

            precision : int
                Number of bits of precision to use in the QSVE subroutine.
        """
        self._matrix = deepcopy(Amatrix)
        self._vector = deepcopy(bvector)
        self._precision = precision
        self._cval = cval
        self._qsve = QSVE(Amatrix, singular_vector=bvector, nprecision_bits=precision)

    @property
    def matrix(self):
        return self._matrix

    @property
    def vector(self):
        return self._vector

    def classical_solution(self, normalized=True):
        """Returns the solution of the linear system found classically.

        Args:
            normalized : bool (default value = True)
                If True, the classical solution is normalized (by the L2 norm), else it is un-normalized.
        """
        xclassical = np.linalg.solve(self._matrix, self._vector)
        if normalized:
            return xclassical / np.linalg.norm(xclassical, ord=2)
        return xclassical

    def _hhl_rotation(self, circuit, eval_register, ancilla_qubit, constant=0.5):
        """Adds the gates for the HHL rotation to perform the transformation


            sum_j beta_j |lambda_j>  ---->  sum_j beta_j / lambda_j |lambda_j>

        Args:

        """
        # The number of controls is the number of qubits in the eval_register
        ncontrols = len(eval_register)

        for ii in range(2**ncontrols):
            # Get the bitstring for this index
            bitstring = np.binary_repr(ii, ncontrols)

            # Do the initial sequence of NOT gates to get controls/anti-controls correct
            for (ind, bit) in enumerate(bitstring):
                if bit == "0":
                    circuit.x(eval_register[ind])

            # Determine the theta value in this amplitude
            theta = self._qsve.binary_decimal_to_float(bitstring)

            # Compute the eigenvalue in this register
            eigenvalue = self._qsve.matrix_norm() * np.cos(np.pi * theta)

            # TODO: Is this correct to do?
            if np.isclose(eigenvalue, 0.0):
                continue

            # Determine the angle of rotation for the Y-rotation
            angle = 2 * np.arccos(constant / eigenvalue)

            # Do the controlled Y-rotation
            mcry(circuit, angle, eval_register, ancilla_qubit, None, mode="noancilla")

            # Do the final sequence of NOT gates to get controls/anti-controls correct
            for (ind, bit) in enumerate(bitstring):
                if bit == "0":
                    circuit.x(eval_register[ind])

    def create_circuit(self, return_registers=False):
        """Creates the circuit that solves the linear system Ax = b."""
        # Get the QSVE circuit
        circuit, qpe_register, row_register, col_register = self._qsve.create_circuit(return_registers=True)

        # Make a copy to take the inverse of later
        qsve_circuit = deepcopy(circuit)

        # Add the ancilla register (of one qubit) for the HHL rotation
        ancilla = QuantumRegister(1, name="anc")
        circuit.add_register(ancilla)

        # Do the HHL rotation
        self._hhl_rotation(circuit, qpe_register, ancilla[0], self._cval)

        # Add the inverse QSVE circuit (without the initial data loading subroutines)
        circuit += self._qsve.create_circuit(initial_loads=False).inverse()

        if return_registers:
            return circuit, qpe_register, row_register, col_register, ancilla
        return circuit

    def _run(self, simulator, shots):
        """Runs the quantum circuit and returns the measurement counts."""
        pass

    def quantum_solution(self):
        pass

    def compute_expectation(self, observable):
        pass


if __name__ == "__main__":
    mat = np.identity(32)
    vec = np.zeros(len(mat))
    vec[0] = 1

    solver = LinearSystemSolverQSVE(mat, vec, precision=3)

    circ = solver.create_circuit()

    print(circ.count_ops())
