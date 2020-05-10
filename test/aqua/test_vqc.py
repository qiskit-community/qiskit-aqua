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

""" Test VQC """

import os
import unittest
import warnings
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, data
from qiskit import BasicAer
from qiskit.circuit import ParameterVector, QuantumCircuit, Parameter
from qiskit.circuit.library import TwoLocal, ZZFeatureMap, EfficientSU2
from qiskit.aqua import QuantumInstance, aqua_globals, AquaError
from qiskit.aqua.algorithms import VQC
from qiskit.aqua.components.optimizers import SPSA, COBYLA
from qiskit.aqua.components.feature_maps import SecondOrderExpansion, RawFeatureVector
from qiskit.aqua.components.variational_forms import RYRZ
from qiskit.aqua.components.optimizers import L_BFGS_B
from qiskit.ml.datasets import wine, ad_hoc_data


@ddt
class TestVQC(QiskitAquaTestCase):
    """Test the VQC algorithm."""

    def setUp(self):
        super().setUp()
        # self.seed = 1376
        self.seed = 1378
        aqua_globals.random_seed = self.seed
        self.training_data = {'A': np.asarray([[2.95309709, 2.51327412], [3.14159265, 4.08407045]]),
                              'B': np.asarray([[4.08407045, 2.26194671], [4.46106157, 2.38761042]])}
        self.testing_data = {'A': np.asarray([[3.83274304, 2.45044227]]),
                             'B': np.asarray([[3.89557489, 0.31415927]])}
        self.ref_opt_params = np.array([-3.08190321, -0.03132918, 2.87252832, -4.18715976,
                                        7.36305705, 8.04016843, 1.09389159, -0.52016933, 0.97312367,
                                        -1.314532, 4.76849427, 7.51815271, 1.19324999, -1.48298067,
                                        -7.89851594, 4.00974338])

        self.ref_train_loss = 0.61108566
        self.ref_prediction_a_probs = [[0.62988281, 0.37011719]]

        self.ref_prediction_a_label = [0]

        library_ryrz = EfficientSU2(2)
        self.ryrz_wavefunction = library_ryrz

        library_circuit = ZZFeatureMap(2, reps=2)
        self.data_preparation = library_circuit

    def test_vqc(self):
        """ vqc test """
        aqua_globals.random_seed = self.seed
        optimizer = SPSA(max_trials=10, save_steps=1, c0=4.0, c1=0.1, c2=0.602, c3=0.101, c4=0.0,
                         skip_calibration=True)
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction

        # set up algorithm
        vqc = VQC(optimizer, data_preparation, wavefunction, self.training_data, self.testing_data)

        quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                           shots=1024,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqc.run(quantum_instance)

        np.testing.assert_array_almost_equal(result['opt_params'], self.ref_opt_params, decimal=8)
        np.testing.assert_array_almost_equal(result['training_loss'], self.ref_train_loss,
                                             decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

    def test_vqc_plain_circuit(self):
        """Test VQC on a plain circuit object."""
        aqua_globals.random_seed = self.seed
        optimizer = SPSA(max_trials=10, save_steps=1, c0=4.0, c1=0.1, c2=0.602, c3=0.101, c4=0.0,
                         skip_calibration=True)
        data_preparation = QuantumCircuit(2).compose(self.data_preparation)
        wavefunction = QuantumCircuit(2).compose(self.ryrz_wavefunction)

        # set up algorithm
        vqc = VQC(optimizer, data_preparation, wavefunction, self.training_data, self.testing_data)

        quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                           shots=1024,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqc.run(quantum_instance)

        np.testing.assert_array_almost_equal(result['opt_params'], self.ref_opt_params, decimal=8)
        np.testing.assert_array_almost_equal(result['training_loss'], self.ref_train_loss,
                                             decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

    def test_vqc_on_deprecated_components(self):
        """Test VQC on a VariationalForm and FeatureMap."""
        aqua_globals.random_seed = 1376
        ref_opt_params = np.array([10.03814083, -12.22048954, -7.58026833, -2.42392954,
                                   12.91555293, 13.44064652, -2.89951454, -10.20639406,
                                   0.81414546, -1.00551752, -4.7988307, 14.00831419,
                                   8.26008064, -7.07543736, 11.43368677, -5.74857438])
        ref_train_loss = 0.69366523
        optimizer = SPSA(max_trials=10, save_steps=1, c0=4.0, c1=0.1, c2=0.602, c3=0.101, c4=0.0,
                         skip_calibration=True)
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        data_preparation = SecondOrderExpansion(2)
        wavefunction = RYRZ(2)

        # set up algorithm
        vqc = VQC(optimizer, data_preparation, wavefunction, self.training_data, self.testing_data)
        warnings.filterwarnings('always', category=DeprecationWarning)

        quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                           shots=1024,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqc.run(quantum_instance)

        np.testing.assert_array_almost_equal(result['opt_params'], ref_opt_params, decimal=8)
        np.testing.assert_array_almost_equal(result['training_loss'], ref_train_loss, decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

    def test_vqc_with_max_evals_grouped(self):
        """ vqc with max evals grouped test """
        aqua_globals.random_seed = self.seed
        optimizer = SPSA(max_trials=10, save_steps=1,
                         c0=4.0, c1=0.1, c2=0.602, c3=0.101, c4=0.0, skip_calibration=True)
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction

        # set up algorithm
        vqc = VQC(optimizer, data_preparation, wavefunction, self.training_data, self.testing_data,
                  max_evals_grouped=2)

        quantum_instance = QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                           shots=1024,
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqc.run(quantum_instance)
        np.testing.assert_array_almost_equal(result['opt_params'],
                                             self.ref_opt_params, decimal=8)
        np.testing.assert_array_almost_equal(result['training_loss'],
                                             self.ref_train_loss, decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

    def test_vqc_statevector(self):
        """ vqc statevector test """
        aqua_globals.random_seed = 2
        optimizer = COBYLA(maxiter=500)
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction

        # set up algorithm
        vqc = VQC(optimizer, data_preparation, wavefunction, self.training_data, self.testing_data)

        quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                           seed_simulator=aqua_globals.random_seed,
                                           seed_transpiler=aqua_globals.random_seed)
        result = vqc.run(quantum_instance)
        self.assertEqual(result['testing_accuracy'], 0.5)
        ref_train_loss = 0.1082656
        np.testing.assert_array_almost_equal(result['training_loss'], ref_train_loss, decimal=4)

    # we use the ad_hoc dataset (see the end of this file) to test the accuracy.
    def test_vqc_minibatching_no_gradient_support(self):
        """ vqc minibatching with no gradient support test """
        n_dim = 2  # dimension of each data point
        seed = 1024
        aqua_globals.random_seed = seed
        _, training_input, test_input, _ = ad_hoc_data(training_size=6,
                                                       test_size=3,
                                                       n=n_dim,
                                                       gap=0.3,
                                                       plot_data=False)
        backend = BasicAer.get_backend('statevector_simulator')
        optimizer = COBYLA(maxiter=40)
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction

        # set up algorithm
        vqc = VQC(optimizer, data_preparation, wavefunction, training_input, test_input,
                  minibatch_size=2)

        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed,
                                           optimization_level=0)
        result = vqc.run(quantum_instance)
        self.log.debug(result['testing_accuracy'])
        self.assertGreaterEqual(result['testing_accuracy'], 0.5)

    def test_vqc_minibatching_with_gradient_support(self):
        """ vqc minibatching with gradient support test """
        n_dim = 2  # dimension of each data point
        seed = 1024
        aqua_globals.random_seed = seed
        _, training_input, test_input, _ = ad_hoc_data(training_size=4,
                                                       test_size=2,
                                                       n=n_dim,
                                                       gap=0.3,
                                                       plot_data=False)
        backend = BasicAer.get_backend('statevector_simulator')
        optimizer = L_BFGS_B(maxfun=30)

        # set up data encoding circuit
        data_preparation = self.data_preparation

        # set up wavefunction
        wavefunction = TwoLocal(2, ['ry', 'rz'], 'cz', reps=1, insert_barriers=True)

        # set up algorithm
        vqc = VQC(optimizer, data_preparation, wavefunction, training_input, test_input,
                  minibatch_size=2)

        quantum_instance = QuantumInstance(backend, seed_simulator=seed, seed_transpiler=seed)
        result = vqc.run(quantum_instance)
        vqc_accuracy = 0.75
        self.log.debug(result['testing_accuracy'])
        self.assertAlmostEqual(result['testing_accuracy'], vqc_accuracy, places=3)

    def test_save_and_load_model(self):
        """ save and load model test """
        aqua_globals.random_seed = self.seed
        backend = BasicAer.get_backend('qasm_simulator')

        optimizer = SPSA(max_trials=10, save_steps=1, c0=4.0, skip_calibration=True)
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction

        # set up algorithm
        vqc = VQC(optimizer, data_preparation, wavefunction, self.training_data, self.testing_data)

        quantum_instance = QuantumInstance(backend,
                                           shots=1024,
                                           seed_simulator=self.seed,
                                           seed_transpiler=self.seed)
        result = vqc.run(quantum_instance)

        np.testing.assert_array_almost_equal(result['opt_params'],
                                             self.ref_opt_params, decimal=4)
        np.testing.assert_array_almost_equal(result['training_loss'],
                                             self.ref_train_loss, decimal=8)

        self.assertEqual(1.0, result['testing_accuracy'])

        file_path = self.get_resource_path('vqc_test.npz')
        vqc.save_model(file_path)

        self.assertTrue(os.path.exists(file_path))

        loaded_vqc = VQC(optimizer, data_preparation, wavefunction, self.training_data, None)

        loaded_vqc.load_model(file_path)

        np.testing.assert_array_almost_equal(
            loaded_vqc.ret['opt_params'], self.ref_opt_params, decimal=4)

        loaded_test_acc = loaded_vqc.test(vqc.test_dataset[0],
                                          vqc.test_dataset[1],
                                          quantum_instance)
        self.assertEqual(result['testing_accuracy'], loaded_test_acc)

        predicted_probs, predicted_labels = loaded_vqc.predict(self.testing_data['A'],
                                                               quantum_instance)
        np.testing.assert_array_almost_equal(predicted_probs,
                                             self.ref_prediction_a_probs,
                                             decimal=8)
        np.testing.assert_array_equal(predicted_labels, self.ref_prediction_a_label)

        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:  # pylint: disable=broad-except
                pass

    def test_vqc_callback(self):
        """ vqc callback test """
        history = {'eval_count': [], 'parameters': [], 'cost': [], 'batch_index': []}

        def store_intermediate_result(eval_count, parameters, cost, batch_index):
            history['eval_count'].append(eval_count)
            history['parameters'].append(parameters)
            history['cost'].append(cost)
            history['batch_index'].append(batch_index)

        aqua_globals.random_seed = self.seed
        backend = BasicAer.get_backend('qasm_simulator')

        optimizer = COBYLA(maxiter=3)
        data_preparation = self.data_preparation
        wavefunction = self.ryrz_wavefunction

        # set up algorithm
        vqc = VQC(optimizer, data_preparation, wavefunction, self.training_data, self.testing_data,
                  callback=store_intermediate_result)

        quantum_instance = QuantumInstance(backend,
                                           shots=1024,
                                           seed_simulator=self.seed,
                                           seed_transpiler=self.seed)
        vqc.run(quantum_instance)

        self.assertTrue(all(isinstance(count, int) for count in history['eval_count']))
        self.assertTrue(all(isinstance(cost, float) for cost in history['cost']))
        self.assertTrue(all(isinstance(index, int) for index in history['batch_index']))
        for params in history['parameters']:
            self.assertTrue(all(isinstance(param, float) for param in params))

    def test_same_parameter_names_raises(self):
        """Test that the varform and feature map can have parameters with the same name."""
        var_form = QuantumCircuit(1)
        var_form.ry(Parameter('a'), 0)
        feature_map = QuantumCircuit(1)
        feature_map.rz(Parameter('a'), 0)
        optimizer = SPSA()
        vqc = VQC(optimizer, feature_map, var_form, self.training_data, self.testing_data)

        with self.assertRaises(AquaError):
            _ = vqc.run(BasicAer.get_backend('statevector_simulator'))

    def test_feature_map_without_parameters_warns(self):
        """Test that specifying a feature map with 0 parameters raises a warning."""
        var_form = QuantumCircuit(1)
        var_form.ry(Parameter('a'), 0)
        feature_map = QuantumCircuit(1)
        optimizer = SPSA()
        with self.assertWarns(UserWarning):
            vqc = VQC(optimizer, feature_map, var_form, self.training_data, self.testing_data)
            # _ = vqc.run(BasicAer.get_backend('statevector_simulator'))

    def test_vqc_on_wine(self):
        """Test VQE on the wine test using circuits as feature map and variational form."""
        feature_dim = 4  # dimension of each data point
        training_dataset_size = 6
        testing_dataset_size = 3

        _, training_input, test_input, _ = wine(training_size=training_dataset_size,
                                                test_size=testing_dataset_size,
                                                n=feature_dim,
                                                plot_data=False)
        aqua_globals.random_seed = self.seed
        data_preparation = ZZFeatureMap(feature_dim)
        wavefunction = TwoLocal(feature_dim, rotation_blocks=['ry', 'rz'],
                                entanglement_blocks='cz', reps=1)

        vqc = VQC(COBYLA(maxiter=100), data_preparation, wavefunction, training_input, test_input)

        result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         shots=1024,
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.log.debug(result['testing_accuracy'])

        self.assertLess(result['testing_accuracy'], 0.6)

    def test_vqc_with_raw_feature_vector_on_wine(self):
        """ vqc with raw features vector on wine test """
        feature_dim = 4  # dimension of each data point
        training_dataset_size = 8
        testing_dataset_size = 4

        _, training_input, test_input, _ = wine(training_size=training_dataset_size,
                                                test_size=testing_dataset_size,
                                                n=feature_dim,
                                                plot_data=False)
        aqua_globals.random_seed = self.seed
        feature_map = RawFeatureVector(feature_dimension=feature_dim)
        vqc = VQC(COBYLA(maxiter=100),
                  feature_map,
                  TwoLocal(feature_map.num_qubits, ['ry', 'rz'], 'cz', reps=3),
                  training_input,
                  test_input)
        result = vqc.run(QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         shots=1024,
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.log.debug(result['testing_accuracy'])

        self.assertGreater(result['testing_accuracy'], 0.8)


if __name__ == '__main__':
    unittest.main()
