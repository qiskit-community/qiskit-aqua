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

"""Tests for the VQE algorithm."""

import unittest
import warnings
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, unpack, data
from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit.library import TwoLocal, EfficientSU2, RealAmplitudes

from qiskit.aqua import QuantumInstance, aqua_globals, AquaError
from qiskit.aqua.operators import WeightedPauliOperator, PrimitiveOp
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.optimizers import L_BFGS_B, COBYLA, SPSA, SLSQP
from qiskit.aqua.algorithms import VQE


@ddt
class TestVQE(QiskitAquaTestCase):
    """Test the VQE algorithm."""

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
        self.qubit_op = WeightedPauliOperator.from_dict(pauli_dict).to_opflow()

    def test_vqe_deprecated_varforms(self):
        """Test the VQE on the deprecated variational forms."""
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        wavefunction = RYRZ(self.qubit_op.num_qubits)
        vqe = VQE(self.qubit_op, wavefunction, L_BFGS_B())
        warnings.filterwarnings('always', category=DeprecationWarning)

        result = vqe.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                                         coupling_map=[[0, 1]],
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503)
        np.testing.assert_array_almost_equal(result.eigenvalue.real, -1.85727503, 5)
        self.assertEqual(len(result.optimal_point), 16)
        self.assertIsNotNone(result.cost_function_evals)
        self.assertIsNotNone(result.optimizer_time)

    def test_vqe_plain_circuit(self):
        """Test the VQE on a plain QuantumCircuit object."""
        num_qubits = self.qubit_op.num_qubits
        wavefunction = QuantumCircuit(num_qubits).combine(EfficientSU2(num_qubits))
        vqe = VQE(self.qubit_op, wavefunction, L_BFGS_B())
        result = vqe.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                                         coupling_map=[[0, 1]],
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503)
        np.testing.assert_array_almost_equal(result.eigenvalue.real, -1.85727503, 5)
        self.assertEqual(len(result.optimal_point), 16)
        self.assertIsNotNone(result.cost_function_evals)
        self.assertIsNotNone(result.optimizer_time)

    def test_vqe(self):
        """Test the VQE on statevector simulation."""
        wavefunction = EfficientSU2()
        vqe = VQE(self.qubit_op, wavefunction, L_BFGS_B())

        result = vqe.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         basis_gates=['u1', 'u2', 'u3', 'cx', 'id'],
                                         coupling_map=[[0, 1]],
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503)
        np.testing.assert_array_almost_equal(result.eigenvalue.real, -1.85727503, 5)
        self.assertEqual(len(result.optimal_point), 16)
        self.assertIsNotNone(result.cost_function_evals)
        self.assertIsNotNone(result.optimizer_time)

    def test_vqe_no_varform_params(self):
        """Test specifying a variational form with no parameters raises an error."""
        circuit = QuantumCircuit(self.qubit_op.num_qubits)
        vqe = VQE(self.qubit_op, circuit)
        with self.assertRaises(RuntimeError):
            vqe.run(BasicAer.get_backend('statevector_simulator'))

    @data(
        (SLSQP, 5, 4),
        (SLSQP, 5, 1),
        (SPSA, 3, 2),  # max_evals_grouped=n is considered as max_evals_grouped=2 if n>2
        (SPSA, 3, 1)
    )
    @unpack
    def test_vqe_optimizers(self, optimizer_cls, places, max_evals_grouped):
        """Test VQE on different optimizers."""
        result = VQE(self.qubit_op,
                     TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cz'),
                     optimizer_cls(),
                     max_evals_grouped=max_evals_grouped).run(
                         QuantumInstance(BasicAer.get_backend('statevector_simulator'), shots=1,
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))

        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=places)

    def test_vqe_qasm(self):
        """Test VQE on the QASM simulator."""
        backend = BasicAer.get_backend('qasm_simulator')
        optimizer = SPSA(max_trials=300, last_avg=5)
        wavefunction = RealAmplitudes()
        vqe = VQE(self.qubit_op, wavefunction, optimizer, max_evals_grouped=1)

        # TODO benchmark this later.
        quantum_instance = QuantumInstance(backend, shots=1000,
                                           seed_simulator=self.seed,
                                           seed_transpiler=self.seed)
        result = vqe.run(quantum_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.86823, places=2)

    def test_vqe_statevector_snapshot_mode(self):
        """Test VQE in Aer's statevector_simulator snapshot mode."""
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('statevector_simulator')
        wavefunction = RealAmplitudes()
        optimizer = L_BFGS_B()

        vqe = VQE(self.qubit_op, wavefunction, optimizer, max_evals_grouped=1)
        quantum_instance = QuantumInstance(backend,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqe.run(quantum_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=6)

    def test_vqe_qasm_snapshot_mode(self):
        """Test VQE Aer's qasm_simulator snapshot mode."""
        try:
            # pylint: disable=import-outside-toplevel
            from qiskit import Aer
        except Exception as ex:  # pylint: disable=broad-except
            self.skipTest("Aer doesn't appear to be installed. Error: '{}'".format(str(ex)))
            return
        backend = Aer.get_backend('qasm_simulator')
        optimizer = L_BFGS_B()
        wavefunction = RealAmplitudes()

        vqe = VQE(self.qubit_op, wavefunction, optimizer, max_evals_grouped=1)

        quantum_instance = QuantumInstance(backend, shots=1,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqe.run(quantum_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=6)

    def test_vqe_callback(self):
        """Test the callback function of VQE."""
        history = {'eval_count': [], 'parameters': [], 'mean': [], 'std': []}

        def store_intermediate_result(eval_count, parameters, mean, std):
            history['eval_count'].append(eval_count)
            history['parameters'].append(parameters)
            history['mean'].append(mean)
            history['std'].append(std)

        backend = BasicAer.get_backend('qasm_simulator')
        optimizer = COBYLA(maxiter=3)
        wavefunction = RealAmplitudes()

        vqe = VQE(self.qubit_op, wavefunction, optimizer, callback=store_intermediate_result)

        quantum_instance = QuantumInstance(backend,
                                           seed_transpiler=50,
                                           shots=1024,
                                           seed_simulator=50)
        vqe.run(quantum_instance)

        self.assertTrue(all(isinstance(count, int) for count in history['eval_count']))
        self.assertTrue(all(isinstance(mean, float) for mean in history['mean']))
        self.assertTrue(all(isinstance(std, float) for std in history['std']))
        for params in history['parameters']:
            self.assertTrue(all(isinstance(param, float) for param in params))

    def test_vqe_reuse(self):
        """Test reuse of the VQE."""
        vqe = VQE()
        with self.assertRaises(AquaError):
            _ = vqe.run()

        var_form = EfficientSU2()
        vqe.var_form = var_form
        with self.assertRaises(AquaError):
            _ = vqe.run()

        vqe.operator = self.qubit_op
        with self.assertRaises(AquaError):
            _ = vqe.run()

        qinst = QuantumInstance(BasicAer.get_backend('statevector_simulator'))
        vqe.quantum_instance = qinst
        result = vqe.run()
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=5)

        operator = PrimitiveOp(np.array([[1, 0, 0, 0],
                                         [0, -1, 0, 0],
                                         [0, 0, 2, 0],
                                         [0, 0, 0, 3]]))
        vqe.operator = operator
        result = vqe.run()
        self.assertAlmostEqual(result.eigenvalue.real, -1.0, places=5)

    def test_vqe_mes(self):
        """Test the minimum eigen solver interface of the VQE."""
        ansatz = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cz')
        vqe = VQE(var_form=ansatz, optimizer=COBYLA())
        vqe.set_backend(BasicAer.get_backend('statevector_simulator'))
        result = vqe.compute_minimum_eigenvalue(self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503, places=5)

    @unittest.skip(reason="IBMQ testing not available in general.")
    def test_ibmq_vqe(self):
        """Test VQE on IBMQ."""
        from qiskit import IBMQ
        provider = IBMQ.load_account()
        backend = provider.get_backend('ibmq_qasm_simulator')
        ansatz = TwoLocal(rotation_blocks=['ry', 'rz'], entanglement_blocks='cz')

        opt = SLSQP(maxiter=1)
        opt.set_max_evals_grouped(100)
        vqe = VQE(self.qubit_op, ansatz, SLSQP(maxiter=2))

        result = vqe.run(backend)
        print(result)
        self.assertAlmostEqual(result.eigenvalue.real, -1.85727503)
        np.testing.assert_array_almost_equal(result.eigenvalue.real, -1.85727503, 5)
        self.assertEqual(len(result.optimal_point), 16)
        self.assertIsNotNone(result.cost_function_evals)
        self.assertIsNotNone(result.optimizer_time)


if __name__ == '__main__':
    unittest.main()
