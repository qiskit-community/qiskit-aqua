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

""" Test VQE """

import unittest
from test.aqua.aqua_test_case import QiskitAquaTestCase
from qiskit import BasicAer

from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.aqua.components.variational_forms import RY, RYRZ
from qiskit.aqua.components.optimizers import L_BFGS_B, SPSA, SLSQP
from qiskit.aqua.components.initial_states import Zero
from qiskit.aqua.algorithms import VQE, NumPyMinimumEigensolver


class TestComputeMinEigenvalue(QiskitAquaTestCase):
    """ Test VQE """

    def setUp(self):
        super().setUp()
        self.seed = 50
        aqua_globals.random_seed = self.seed
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        self.qubit_op = WeightedPauliOperator.from_dict(pauli_dict)

    def test_vqe(self):
        """ VQE test """
        quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                           basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                                           coupling_map=[[0, 1]],
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)

        vqe = VQE(var_form=RYRZ(self.qubit_op.num_qubits),
                  optimizer=L_BFGS_B(),
                  quantum_instance=quantum_instance)
        output = vqe.compute_minimum_eigenvalue(self.qubit_op)
        self.assertAlmostEqual(output.eigenvalue, -1.85727503)

    def test_vqe_qasm(self):
        """ VQE QASM test """
        backend = BasicAer.get_backend('qasm_simulator')
        num_qubits = self.qubit_op.num_qubits
        var_form = RY(num_qubits, num_qubits)
        optimizer = SPSA(max_trials=300, last_avg=5)
        quantum_instance = QuantumInstance(backend, shots=10000,
                                           seed_simulator=self.seed,
                                           seed_transpiler=self.seed)
        vqe = VQE(var_form=var_form,
                  optimizer=optimizer,
                  max_evals_grouped=1,
                  quantum_instance=quantum_instance)

        output = vqe.compute_minimum_eigenvalue(self.qubit_op)
        self.assertAlmostEqual(output.eigenvalue, -1.85727503, places=1)

    def test_ee(self):
        """ EE test """
        dummy_operator = MatrixOperator([[1]])
        ee = NumPyMinimumEigensolver()
        output = ee.compute_minimum_eigenvalue(self.qubit_op)

        self.assertAlmostEqual(output.eigenvalue, -1.85727503)


if __name__ == '__main__':
    unittest.main()
