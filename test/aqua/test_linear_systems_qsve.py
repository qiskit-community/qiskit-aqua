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
import numpy as np
import unittest
# TODO: Change back to this before PR: from test.aqua.common import QiskitAquaTestCase
from common import QiskitAquaTestCase
from qiskit import QuantumRegister, QuantumCircuit, BasicAer, execute
from qiskit.aqua.algorithms.single_sample.qsve_linear_systems import LinearSystemSolverQSVE
import parameterized
from qiskit.ignis.verification import tomography


class TestLinearSystemSolverQSVE(QiskitAquaTestCase):
    @parameterized.expand([[[0, 1]], [[1, 0]], [[1, 0.1]], [[1, 1]], [[1, 10]]])
    def test_identity2(self, vector):
        # Define the linear system
        vector = np.array(vector)
        matrix = np.identity(2)
        system = LinearSystemSolverQSVE(matrix, vector, nprecision_bits=5, cval=0.1)

        # Get the circuit and registers
        circuit, _, _, col_register, _ = system.create_circuit(return_registers=True)

        # Test and debug
        print("Current vector is:", vector)
        print("Doing state tomography on col_register, which has {} qubit(s).".format(len(col_register)))
        tomo_circuits = tomography.state_tomography_circuits(circuit, col_register)
        print("There are {} tomography circuits. Running them now...".format(len(tomo_circuits)))
        job = execute(tomo_circuits, BasicAer.get_backend("qasm_simulator"), shots=10000)
        print("Finished running tomography circuits. Now fitting density matrix.")
        fitter = tomography.StateTomographyFitter(job.result(), tomo_circuits)
        rho = fitter.fit()
        print("Density matrix:\n", rho)

        expected = np.outer(vector, vector)
        expected /= np.trace(expected)

        print("Expected density matrix:\n", expected)

        self.assertTrue(np.allclose(rho, expected, atol=1e-2))


if __name__ == "__main__":
    unittest.main()
