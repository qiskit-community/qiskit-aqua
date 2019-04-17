# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest

import numpy as np

# from parameterized import parameterized

from qiskit.aqua.components.optimizers import ADAM
from qiskit.aqua.components.uncertainty_models import UniformDistribution, UnivariateVariationalDistribution
from qiskit.aqua.components.variational_forms import RY

from qiskit.aqua.algorithms.adaptive.qgan.qgan import QGAN

from qiskit.aqua import aqua_globals, QuantumInstance

from qiskit import Aer

from test.common import QiskitAquaTestCase


class TestQGAN(QiskitAquaTestCase):
    # @parameterized.expand([
    #     'qasm_simulator',
    #     'statevector_simulator'
    # ])
    # def setUp(self, simulator):
    def setUp(self):
        super().setUp()
        # Number training data samples
        N = 1000
        # Load data samples from log-normal distribution with mean=1 and standard deviation=1
        mu = 1
        sigma = 1
        real_data = np.random.lognormal(mean=mu, sigma=sigma, size=N)
        # Set the data resolution
        # Set upper and lower data values as list of k min/max data values [[min_0,max_0],...,[min_k-1,max_k-1]]
        bounds = np.array([0., 3.])
        # Set number of qubits per data dimension as list of k qubit values[#q_0,...,#q_k-1]
        num_qubits = [2]
        # Set number of training epochs
        num_epochs = 10
        # Batch size
        batch_size = 100

        # Initialize qGAN
        self.qgan = QGAN(real_data, bounds, num_qubits, batch_size, num_epochs, snapshot_dir=None)

        # Set quantum instance to run the quantum generator
        simulator = 'statevector_simulator'
        backend = Aer.get_backend(simulator)
        self.quantum_instance = QuantumInstance(backend=backend, shots=batch_size, coupling_map=None,
                                                circuit_caching=False)
        self.qgan.set_quantum_instance(self.quantum_instance)

        # Set entangler map
        entangler_map = [[0, 1]]

        # Set variational form
        var_form = RY(int(np.sum(num_qubits)), depth=1, entangler_map=entangler_map, entanglement_gate='cz')
        # Set generator's initial parameters
        init_params = aqua_globals.random.rand(var_form._num_parameters) * 2 * 1e-2
        # Set an initial state for the generator circuit
        init_dist = UniformDistribution(np.sum(num_qubits), low=bounds[0], high=bounds[1])
        # Set generator circuit
        g_circuit = UnivariateVariationalDistribution(np.sum(num_qubits), var_form, init_params,
                                                      initial_distribution=init_dist, low=bounds[0], high=bounds[1])
        # Set generator optimizer
        g_optimizer = ADAM(maxiter=1, tol=1e-6, lr=1e-5, beta_1=0.9, beta_2=0.99, noise_factor=1e-6,
                           eps=1e-10, amsgrad=True)
        # Set quantum generator
        self.qgan.set_generator(generator_circuit=g_circuit, generator_optimizer=g_optimizer)

    def test_sample_generation(self):
        self.qgan.generator.get_samples(self.quantum_instance)

    def test_qgan_training(self):
        self.qgan.run()


if __name__ == '__main__':
    unittest.main()
