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

"""Test phase estimation"""

import unittest
from test.aqua import QiskitAquaTestCase
import numpy as np
from qiskit.aqua.algorithms.phase_estimators import PhaseEstimator, HamiltonianPE
from qiskit.aqua.algorithms.phase_estimators.hamiltonian_pe import TempPauliEvolve
import qiskit
from qiskit.aqua.operators import (H, X, Y, Z)


class TestHamiltonianPE(QiskitAquaTestCase):
    """Tests for obtaining eigenvalues from phase estimation"""

    # pylint: disable=invalid-name
    def hamiltonian_pe(self, hamiltonian, state_preparation=None, num_evaluation_qubits=10,
                       backend=qiskit.Aer.get_backend('qasm_simulator')):
        """Run HamiltonianPE and return result with all  phases."""
        qi = qiskit.aqua.QuantumInstance(backend=backend, shots=100000)
        phase_est = HamiltonianPE(
            num_evaluation_qubits=num_evaluation_qubits,
            hamiltonian=hamiltonian, quantum_instance=qi,
            state_preparation=state_preparation, evolution=TempPauliEvolve())
        result = phase_est.run()
        return result

    # pylint: disable=invalid-name
    def test_pauli_sum_1(self):
        """Two eigenvalues from Pauli sum with X, Z"""
        a1 = 0.5
        a2 = 1.0
        hamiltonian = (a1 * X) + (a2 * Z)
        state_preparation = H.to_circuit()
        result = self.hamiltonian_pe(hamiltonian, state_preparation)
        phase_dict = result.filter_phases(0.162, as_float=True)
        phases = list(phase_dict.keys())
        self.assertAlmostEqual(phases[0], 1.119, delta=0.001)
        self.assertAlmostEqual(phases[1], -1.119, delta=0.001)

    # pylint: disable=invalid-name
    def test_pauli_sum_2(self):
        """Two eigenvalues from Pauli sum with X, Y, Z"""
        a1 = 0.5
        a2 = 1.0
        a3 = 1.0
        hamiltonian = (a1 * X) + (a2 * Y) + (a3 * Z)
        state_preparation = None
        result = self.hamiltonian_pe(hamiltonian, state_preparation)
        phase_dict = result.filter_phases(0.1, as_float=True)
        phases = list(phase_dict.keys())
        self.assertAlmostEqual(phases[0], 1.5, delta=0.001)
        self.assertAlmostEqual(phases[1], -1.5, delta=0.001)

    # pylint: disable=invalid-name
    def test_from_bound(self):
        """HamiltonianPE with bound"""
        a1 = 0.5
        a2 = 1.0
        a3 = 1.0
        hamiltonian = (a1 * X) + (a2 * Y) + (a3 * Z)
        state_preparation = None
        bound = 1.2 * sum([abs(hamiltonian.coeff * pauli.coeff) for pauli in hamiltonian])
        backend = qiskit.Aer.get_backend('qasm_simulator')
        qi = qiskit.aqua.QuantumInstance(backend=backend, shots=100000)
        phase_est = HamiltonianPE(num_evaluation_qubits=8,
                                  hamiltonian=hamiltonian,
                                  bound=bound,
                                  quantum_instance=qi,
                                  state_preparation=state_preparation,
                                  evolution=TempPauliEvolve())
        result = phase_est.run()
        phases = result.filter_phases()
        self.assertEqual(len(phases), 2)
        self.assertEqual(list(phases.keys()), [1.5, -1.5])
        phases = result.filter_phases(scaled=False)
        self.assertEqual(list(phases.keys()), [0.25, 0.75])
        self.assertEqual(result.single_phase(), -1.5)
        self.assertEqual(result.single_phase(scaled=False), 0.75)


class TestPhaseEstimator(QiskitAquaTestCase):
    """Evolution tests."""

    # pylint: disable=invalid-name
    def one_phase(self, unitary_circuit, state_preparation=None, n_eval_qubits=8,
                  backend=qiskit.Aer.get_backend('qasm_simulator')):
        """Run phase estimation with operator, eigenvalue pair `unitary_circuit`,
        `state_preparation`. Return the bit string with the largest amplitude.
        """
        qi = qiskit.aqua.QuantumInstance(backend=backend, shots=100000)
        p_est = PhaseEstimator(num_evaluation_qubits=n_eval_qubits,
                               unitary=unitary_circuit,
                               quantum_instance=qi,
                               state_preparation=state_preparation)
        result = p_est.run()
        phase = result.single_phase()
        return phase

    def test_qpe_Z0(self):
        """eigenproblem Z, |0>"""

        unitary_circuit = Z.to_circuit()
        state_preparation = None  # prepare |0>
        phase = self.one_phase(unitary_circuit, state_preparation)
        self.assertEqual(phase, 0.0)

    def test_qpe_Z0_statevector(self):
        """eigenproblem Z, |0>, statevector simulator"""

        unitary_circuit = Z.to_circuit()
        state_preparation = None  # prepare |0>
        phase = self.one_phase(unitary_circuit, state_preparation,
                               backend=qiskit.Aer.get_backend('statevector_simulator'))
        self.assertEqual(phase, 0.0)

    def test_qpe_Z1(self):
        """eigenproblem Z, |1>"""
        unitary_circuit = Z.to_circuit()
        state_preparation = X.to_circuit()  # prepare |1>
        phase = self.one_phase(unitary_circuit, state_preparation)
        self.assertEqual(phase, 0.5)

    def test_qpe_Xplus(self):
        """eigenproblem X, |+>"""
        unitary_circuit = X.to_circuit()
        state_preparation = H.to_circuit()  # prepare |+>
        phase = self.one_phase(unitary_circuit, state_preparation)
        self.assertEqual(phase, 0.0)

    def test_qpe_Xminus(self):
        """eigenproblem X, |->"""
        unitary_circuit = X.to_circuit()
        state_preparation = X.to_circuit()
        state_preparation.append(H.to_circuit(), [0])  # prepare |->
        phase = self.one_phase(unitary_circuit, state_preparation)
        self.assertEqual(phase, 0.5)

    def phase_estimation(self, unitary_circuit, state_preparation=None, num_evaluation_qubits=8,
                         backend=qiskit.Aer.get_backend('qasm_simulator')):
        """Run phase estimation with operator, eigenvalue pair `unitary_circuit`,
        `state_preparation`. Return all results
        """
        qi = qiskit.aqua.QuantumInstance(backend=backend, shots=100000)
        phase_est = PhaseEstimator(num_evaluation_qubits=num_evaluation_qubits,
                                   unitary=unitary_circuit,
                                   quantum_instance=qi,
                                   state_preparation=state_preparation)
        result = phase_est.run()
        return result

    def test_qpe_Zplus(self):
        """superposition eigenproblem Z, |+>"""
        unitary_circuit = Z.to_circuit()
        state_preparation = H.to_circuit()  # prepare |+>
        result = self.phase_estimation(
            unitary_circuit, state_preparation,
            backend=qiskit.Aer.get_backend('statevector_simulator'))
        phases = result.filter_phases(1e-15, as_float=True)
        self.assertEqual(list(phases.keys()), [0.0, 0.5])
        np.testing.assert_allclose(list(phases.values()), [0.5, 0.5])

    def test_qpe_Zplus_strings(self):
        """superposition eigenproblem Z, |+>, bitstrings"""
        unitary_circuit = Z.to_circuit()
        state_preparation = H.to_circuit()  # prepare |+>
        result = self.phase_estimation(
            unitary_circuit, state_preparation,
            backend=qiskit.Aer.get_backend('statevector_simulator'))
        phases = result.filter_phases(1e-15, as_float=False)
        self.assertEqual(list(phases.keys()), ['00000000', '10000000'])


if __name__ == '__main__':
    unittest.main()
