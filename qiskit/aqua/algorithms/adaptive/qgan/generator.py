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

from __future__ import absolute_import, division, print_function
import numpy as np
from copy import deepcopy

import sys
if sys.version_info < (3, 5):
    raise Exception('Please use Python version 3.5 or greater.')
sys.path.append('..')
sys.path.append('../../quantum_risk_analysis/uncertainty_models')

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.aqua import aqua_globals
from qiskit.aqua.components.optimizers import ADAM
from qiskit.aqua.components.uncertainty_models import UniformDistribution, MultivariateUniformDistribution
from qiskit.aqua.components.uncertainty_models import UnivariateVariationalDistribution, \
    MultivariateVariationalDistribution
from qiskit.aqua.components.variational_forms import RY
from qiskit.aqua import AquaError




class Generator:
    """
    Generator
    """

    def __init__(self, bounds, num_qubits, data_grid, generator_circuit=None, init_params=None,
                 optimizer=None):
        '''  Initialize the generator network.

            Arguments
            ---------
            :param bounds: list of k min/max data values [[min_1,max_1],...,[min_k,max_k]], given input data dim k
            :param num_qubits: list/numpy array, list of k numbers of qubits to determine representation resolution,
            i.e. n qubits enable the representation of 2**n values [n_1,..., n_k]
            :param data_grid: array, describing the data resolution of the qgan, i.e. the representable values v
            :param generator_circuit: UnivariateVariationalDistribution for univariate data/
            MultivariateVariationalDistribution for multivariate data, Quantum circuit to implement the generator.
            :param init_params: 1D numpy array or list, Initialization for the generator's parameters.
        '''

        self._num_qubits = num_qubits
        self._data_grid = data_grid
        self.generator_circuit = generator_circuit
        if self.generator_circuit is None:
            entangler_map = []
            if np.sum(num_qubits) > 2:
                for i in range(int(np.sum(num_qubits))):
                    entangler_map.append([i, int(np.mod(i + 1, np.sum(num_qubits)))])
            else:
                if np.sum(num_qubits) > 1:
                    entangler_map.append([0, 1])
            var_form = RY(int(np.sum(num_qubits)), depth=1, entangler_map=entangler_map, entanglement_gate='cz')
            if init_params is None:
                init_params = aqua_globals.random.rand(var_form._num_parameters) * 2 * 1e-2

            if len(num_qubits)>1:
                num_qubits = list(map(int, num_qubits))
                low=bounds[:, 0].tolist()
                high = bounds[:, 1].tolist()
                init_dist = MultivariateUniformDistribution(num_qubits, low=low, high=high)
                self.generator_circuit = MultivariateVariationalDistribution(num_qubits, var_form, init_params,
                                initial_distribution=init_dist, low=low, high=high)
            else:
                init_dist = UniformDistribution(np.sum(num_qubits), low=bounds[0], high=bounds[1])
                self.generator_circuit = UnivariateVariationalDistribution(np.sum(num_qubits), var_form, init_params,
                                                        initial_distribution=init_dist, low=bounds[0], high=bounds[1])

        if len(num_qubits)>1:
            if isinstance(self.generator_circuit, MultivariateVariationalDistribution):
                pass
            else:
                raise AquaError('Set multivariate variational distribution to represent multivariate data')
        else:
            if isinstance(self.generator_circuit, UnivariateVariationalDistribution):
                pass
            else:
                raise AquaError('Set univariate variational distribution to represent univariate data')

        self._optimizer = optimizer
        if self._optimizer is None:
            self._optimizer = ADAM(maxiter=1, tol=1e-6, lr=1e-5, beta_1=0.9, beta_2=0.99, noise_factor=1e-8,
                 eps=1e-10, amsgrad=True, snapshot_dir=None)
        self._shots = None

    def construct_circuit(self, quantum_instance, params=None):
        """
        Construct generator circuit.
        :param params: array, parameters which should be used to run the generator, if None use self._params
        :return: QuantumCircuit, constructed quantum circuit
        """

        q = QuantumRegister(int(np.sum(self._num_qubits)), name='q')
        qc = QuantumCircuit(q)
        if params is None:
            self.generator_circuit.build(qc=qc, q=q)
        else:
            generator_circuit_copy = deepcopy(self.generator_circuit)
            generator_circuit_copy.params = params
            generator_circuit_copy.build(qc=qc, q=q)

        c = ClassicalRegister(q.size, name='c')
        qc.add_register(c)
        if quantum_instance.is_statevector:
            return qc.copy(name='qc')
        else:
            qc.measure(q, c)
            return qc.copy(name='qc')

    def get_samples(self, quantum_instance, params=None, shots=None):
        """
        Get data samples from the generator.
        :param quantum_instance: QuantumInstance, used to run the generator
        :param params: array, parameters which should be used to run the generator, if None use self._params
        :param shots: int, if not None use a number of shots that is different from the number set in quantum_instance
        :return: array: generated samples, array: sample occurence in percentage
        """
        qc = self.construct_circuit(quantum_instance, params)
        if shots is not None:
            quantum_instance.set_config(shots=shots)
        result = quantum_instance.execute(qc)

        generated_samples=[]
        if quantum_instance.is_statevector:
            result = result.get_statevector(qc)
            values = np.multiply(result, np.conj(result))
            values = list(values.real)
            keys = []
            for j in range(len(values)):
                keys.append(np.binary_repr(j, int(sum(self._num_qubits))))
        else:
            result = result.get_counts(qc)
            keys = list(result)
            values = list(result.values())
            values = [float(v) / np.sum(values) for v in values]
        generated_samples_weights = values
        for i in range(len(keys)):
            index = 0
            temp = []
            for k, p in enumerate(self._num_qubits):
                bin_rep = 0
                j = 0
                while j < p:
                    bin_rep += int(keys[i][index]) * 2 ** (int(p) - j - 1)
                    j += 1
                    index += 1
                if len(self._num_qubits)>1:
                    temp.append(self._data_grid[k][int(bin_rep)])
                else:
                    temp.append(self._data_grid[int(bin_rep)])
            generated_samples.append(temp)

        return generated_samples, generated_samples_weights

    def _loss(self, x, weights):
        return (-1)*np.dot(weights, np.log(x))

    def _get_objective_function(self, quantum_instance, discriminator):
        def objective_function(params):
            generated_data, generated_prob = self.get_samples(quantum_instance, params=params, shots=self._shots)
            prediction_generated = discriminator.get_labels(generated_data).detach().numpy()
            return self._loss(prediction_generated, generated_prob)
        return objective_function

    def train(self, discriminator, quantum_instance, shots=None):
        """
        Perform one training step w.r.t to the generator's parameters
        :param discriminator: Discriminator, Discriminator used to compute the loss function
        :param quantum_instance:  QuantumInstance, used to run the generator
        :param shots: int, Number of shots for hardware or qasm execution
        :return: loss - updated generator loss function
        Note: nfev can be neglected b.c. the optimizer only performs one update step at a time
        """
        self._shots = shots
        # Force single optimization iteration
        self._optimizer._maxiter = 1
        self._optimizer._t = 0
        objective = self._get_objective_function(quantum_instance, discriminator)
        self.generator_circuit.params, loss, nfev = self._optimizer.optimize(num_vars=len(self.generator_circuit.params), objective_function=objective,
                                                        initial_point=self.generator_circuit.params)
        return loss
