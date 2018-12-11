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

import logging
import concurrent.futures
import psutil
import platform

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qiskit_aqua.algorithms.many_sample.qsvm._qsvm_kernel_abc import _QSVM_Kernel_ABC
from qiskit_aqua.utils import map_label_to_class_name, optimize_svm

logger = logging.getLogger(__name__)


class _QSVM_Kernel_Binary(_QSVM_Kernel_ABC):
    """The binary classifier."""

    BATCH_SIZE = 1000

    @staticmethod
    def _construct_circuit(x1, x2, num_qubits, feature_map, measurement, circuit_name=None):
        if x1.shape[0] != x2.shape[0]:
            raise ValueError("x1 and x2 must be the same dimension.")

        q = QuantumRegister(num_qubits, 'q')
        c = ClassicalRegister(num_qubits, 'c')
        qc = QuantumCircuit(q, c, name=circuit_name)
        # write input state from sample distribution
        qc += feature_map.construct_circuit(x1, q)
        qc += feature_map.construct_circuit(x2, q, inverse=True)
        if measurement:
            qc.barrier(q)
            qc.measure(q, c)
        return qc

    @staticmethod
    def _compute_overlap(results, circuit, is_statevector_sim, measurement_basis):
        if is_statevector_sim:
            temp = results.get_statevector(circuit, decimals=16)[0]
            #  |<0|Psi^daggar(y) x Psi(x)|0>|^2,
            kernel_value = np.dot(temp.T.conj(), temp).real
        else:
            result = results.get_counts(circuit)
            kernel_value = result.get(measurement_basis, 0) / sum(result.values())
        return kernel_value

    def construct_circuit(self, x1, x2, measurement=False):
        """
        Generate inner product of x1 and x2 with the given feature map.

        The dimension of x1 and x2 must be the same.

        Args:
            feature_map (FeatureMap): FeatureMap instance
            num_qubits (int): number of qubits
            x1 (numpy.ndarray): data points, 1-D array, dimension is D
            x2 (numpy.ndarray): data points, 1-D array, dimension is D
            measurement (bool): add measurement gates at the end
        """
        return _QSVM_Kernel_Binary._construct_circuit(x1, x2, self.num_qubits,
                                                      self.feature_map, measurement)

    def construct_kernel_matrix(self, x1_vec, x2_vec=None):
        """
        Construct kernel matrix, if x2_vec is None, self-innerproduct is conducted.

        Args:
            x1_vec (numpy.ndarray): data points, 2-D array, N1xD, where N1 is the number of data,
                                    D is the feature dimension
            x2_vec (numpy.ndarray): data points, 2-D array, N2xD, where N2 is the number of data,
                                    D is the feature dimension

        Returns:
            numpy.ndarray: 2-D matrix, N1xN2
        """
        from ._qsvm_kernel_binary import _QSVM_Kernel_Binary

        if x2_vec is None:
            is_symmetric = True
            x2_vec = x1_vec
        else:
            is_symmetric = False

        is_statevector_sim = self.qalgo.quantum_instance.is_statevector
        measurement = not is_statevector_sim
        measurement_basis = '0' * self.num_qubits
        mat = np.ones((x1_vec.shape[0], x2_vec.shape[0]))
        num_processes = psutil.cpu_count(logical=False) if platform.system() != "Windows" else 1

        # get all to-be-computed indices
        if is_symmetric:
            mus, nus = np.triu_indices(x1_vec.shape[0], k=1)  # remove diagonal term
        else:
            mus, nus = np.indices((x1_vec.shape[0], x2_vec.shape[0]))
            mus = np.asarray(mus.flat)
            nus = np.asarray(nus.flat)

        for idx in range(0, len(mus), self.BATCH_SIZE):
            circuits = {}
            to_be_simulated_circuits = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = {}
                for sub_idx in range(idx, min(idx + self.BATCH_SIZE, len(mus))):
                    i = mus[sub_idx]
                    j = nus[sub_idx]
                    x1 = x1_vec[i]
                    x2 = x2_vec[j]
                    if not np.all(x1 == x2):
                        futures["{}:{}".format(i, j)] = \
                            executor.submit(_QSVM_Kernel_Binary._construct_circuit,
                                            x1, x2, self.num_qubits, self.feature_map,
                                            measurement, "circuit{}:{}".format(i, j))

                for k, v in futures.items():
                    circuit = v.result()
                    circuits[k] = circuit
                    to_be_simulated_circuits.append(circuit)

            results = self.qalgo.quantum_instance.execute(to_be_simulated_circuits)
            kernel_values = {}

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                for idx, circuit in circuits.items():
                    kernel_values[idx] = executor.submit(_QSVM_Kernel_Binary._compute_overlap,
                                                         results, circuit, is_statevector_sim, measurement_basis)
                for k, v in kernel_values.items():
                    i, j = [int(x) for x in k.split(":")]
                    mat[i, j] = v.result()
                    if is_symmetric:
                        mat[j, i] = mat[i, j]
        return mat

    def get_predicted_confidence(self, data, return_kernel_matrix=False):
        """Get predicted confidence.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
        Returns:
            numpy.ndarray: Nx1 array, predicted confidence
            numpy.ndarray (optional): the kernel matrix, NxN1, where N1 is
                                      the number of support vectors.
        """
        alphas = self._ret['svm']['alphas']
        bias = self._ret['svm']['bias']
        svms = self._ret['svm']['support_vectors']
        yin = self._ret['svm']['yin']
        kernel_matrix = self.construct_kernel_matrix(data, svms)

        confidence = np.sum(yin * alphas * kernel_matrix, axis=1) + bias

        if return_kernel_matrix:
            return confidence, kernel_matrix
        else:
            return confidence

    def train(self, data, labels):
        """
        Train the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data
        """
        scaling = 1.0 if self.qalgo.quantum_instance.is_statevector else None
        kernel_matrix = self.construct_kernel_matrix(data)
        labels = labels * 2 - 1  # map label from 0 --> -1 and 1 --> 1
        labels = labels.astype(np.float)
        [alpha, b, support] = optimize_svm(kernel_matrix, labels, scaling=scaling)
        support_index = np.where(support)
        alphas = alpha[support_index]
        svms = data[support_index]
        yin = labels[support_index]

        self._ret['kernel_matrix_training'] = kernel_matrix
        self._ret['svm'] = {}
        self._ret['svm']['alphas'] = alphas
        self._ret['svm']['bias'] = b
        self._ret['svm']['support_vectors'] = svms
        self._ret['svm']['yin'] = yin

    def test(self, data, labels):
        """
        Test the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data

        Returns:
            float: accuracy
        """
        predicted_confidence, kernel_matrix = self.get_predicted_confidence(data, True)
        binarized_predictions = (np.sign(predicted_confidence) + 1) / 2  # remap -1 --> 0, 1 --> 1
        predicted_labels = binarized_predictions.astype(int)
        accuracy = np.sum(predicted_labels == labels.astype(int)) / labels.shape[0]
        logger.debug("Classification success for this set is {:.2f}% \n".format(accuracy * 100.0))
        self._ret['kernel_matrix_testing'] = kernel_matrix
        self._ret['testing_accuracy'] = accuracy
        # test_success_ratio is deprecated
        self._ret['test_success_ratio'] = accuracy
        return accuracy

    def predict(self, data):
        """
        Predict using the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        predicted_confidence = self.get_predicted_confidence(data)
        binarized_predictions = (np.sign(predicted_confidence) + 1) / 2  # remap -1 --> 0, 1 --> 1
        predicted_labels = binarized_predictions.astype(int)
        return predicted_labels

    def run(self):
        """Put the train, test, predict together."""
        self.train(self.training_dataset[0], self.training_dataset[1])
        if self.test_dataset is not None:
            self.test(self.test_dataset[0], self.test_dataset[1])
        if self.datapoints is not None:
            predicted_labels = self.predict(self.datapoints)
            predicted_classes = map_label_to_class_name(predicted_labels, self.label_to_class)
            self._ret['predicted_labels'] = predicted_labels
            self._ret['predicted_classes'] = predicted_classes

        return self._ret

    def load_model(self, file_path):
        model_npz = np.load(file_path)
        model = {'alphas': model_npz['alphas'],
                 'bias': model_npz['bias'],
                 'support_vectors': model_npz['support_vectors'],
                 'yin': model_npz['yin']}
        self._ret['svm'] = model

    def save_model(self, file_path):
        model = {'alphas': self._ret['svm']['alphas'],
                 'bias': self._ret['svm']['bias'],
                 'support_vectors': self._ret['svm']['support_vectors'],
                 'yin': self._ret['svm']['yin']}
        np.savez(file_path, **model)
