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

from qiskit_aqua.algorithms import QuantumAlgorithm
from qiskit_aqua import AquaError, PluggableType, get_pluggable_class
from qiskit_aqua.algorithms.many_sample.qsvm._qsvm_kernel_binary import _QSVM_Kernel_Binary
from qiskit_aqua.algorithms.many_sample.qsvm._qsvm_kernel_multiclass import _QSVM_Kernel_Multiclass
from qiskit_aqua.algorithms.many_sample.qsvm._qsvm_kernel_estimator import _QSVM_Kernel_Estimator
from qiskit_aqua.utils.dataset_helper import get_feature_dimension, get_num_classes
from qiskit_aqua.utils import split_dataset_to_data_and_labels

logger = logging.getLogger(__name__)


class QSVMKernel(QuantumAlgorithm):
    """
    Quantum SVM kernel method.

    Internally, it will run the binary classification or multiclass classification
    based on how many classes the data have.
    """

    CONFIGURATION = {
        'name': 'QSVM.Kernel',
        'description': 'QSVMKernel Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QSVM_Kernel_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'depends': ['multiclass_extension', 'feature_map'],
        'problems': ['svm_classification'],
        'defaults': {
            'feature_map': {
                'name': 'SecondOrderExpansion',
                'depth': 2
            }
        }
    }

    BATCH_SIZE = 1000

    def __init__(self, feature_map, training_dataset, test_dataset=None, datapoints=None,
                 multiclass_extension=None):
        """Constructor.

        Args:
            feature_map (FeatureMap): feature map module, used to transform data
            training_dataset (dict): training dataset.
            test_dataset (dict): testing dataset.
            datapoints (numpy.ndarray): prediction dataset.
            multiclass_extension (MultiExtension): if number of classes > 2, a multiclass scheme is
                                                    is needed.

        Raises:
            ValueError: if training_dataset is None
            AquaError: use binary classifer for classes > 3
        """
        super().__init__()
        if training_dataset is None:
            raise ValueError('Training dataset must be provided')

        is_multiclass = get_num_classes(training_dataset) > 2
        if is_multiclass:
            if multiclass_extension is None:
                raise AquaError('Dataset has more than two classes. '
                                'A multiclass extension must be provided.')
        else:
            if multiclass_extension is not None:
                logger.warning("Dataset has just two classes. "
                               "Supplied multiclass extension will be ignored")

        self.training_dataset, self.class_to_label = split_dataset_to_data_and_labels(
            training_dataset)
        if test_dataset is not None:
            self.test_dataset = split_dataset_to_data_and_labels(test_dataset,
                                                                 self.class_to_label)
        else:
            self.test_dataset = None

        self.label_to_class = {label: class_name for class_name, label
                               in self.class_to_label.items()}
        self.num_classes = len(list(self.class_to_label.keys()))
        self.datapoints = datapoints

        self.feature_map = feature_map
        self.num_qubits = self.feature_map.num_qubits

        if multiclass_extension is None:
            qsvm_instance = _QSVM_Kernel_Binary(self)
        else:
            qsvm_instance = _QSVM_Kernel_Multiclass(self, multiclass_extension)

        self.instance = qsvm_instance

    @classmethod
    def init_params(cls, params, algo_input):
        """Constructor from params."""
        num_qubits = get_feature_dimension(algo_input.training_dataset)
        fea_map_params = params.get(QuantumAlgorithm.SECTION_KEY_FEATURE_MAP)
        fea_map_params['num_qubits'] = num_qubits

        feature_map = get_pluggable_class(PluggableType.FEATURE_MAP,
                                          fea_map_params['name']).init_params(fea_map_params)

        multiclass_extension = None
        multiclass_extension_params = params.get(QuantumAlgorithm.SECTION_KEY_MULTICLASS_EXTENSION, None)
        if multiclass_extension_params is not None:
            multiclass_extension_params['params'] = [feature_map]
            multiclass_extension_params['estimator_cls'] = _QSVM_Kernel_Estimator

            multiclass_extension = get_pluggable_class(PluggableType.MULTICLASS_EXTENSION,
                                                       multiclass_extension_params['name']).init_params(multiclass_extension_params)
            logger.info("Multiclass classifier based on {}".format(multiclass_extension_params['name']))

        return cls(feature_map, algo_input.training_dataset, algo_input.test_dataset,
                   algo_input.datapoints, multiclass_extension)

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
            x1 (numpy.ndarray): data points, 1-D array, dimension is D
            x2 (numpy.ndarray): data points, 1-D array, dimension is D
            measurement (bool): add measurement gates at the end
        """
        return QSVMKernel._construct_circuit(x1, x2, self.num_qubits,
                                             self.feature_map, measurement)

    def construct_kernel_matrix(self, x1_vec, x2_vec=None, quantum_instance=None):
        """
        Construct kernel matrix, if x2_vec is None, self-innerproduct is conducted.

        Args:
            x1_vec (numpy.ndarray): data points, 2-D array, N1xD, where N1 is the number of data,
                                    D is the feature dimension
            x2_vec (numpy.ndarray): data points, 2-D array, N2xD, where N2 is the number of data,
                                    D is the feature dimension
            quantum_instance (QuantumInstance): quantum backend with all setting
        Returns:
            numpy.ndarray: 2-D matrix, N1xN2
        """
        self._quantum_instance = self._quantum_instance \
            if quantum_instance is None else quantum_instance
        from .qsvm_kernel import QSVMKernel

        if x2_vec is None:
            is_symmetric = True
            x2_vec = x1_vec
        else:
            is_symmetric = False

        is_statevector_sim = self.quantum_instance.is_statevector
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

        for idx in range(0, len(mus), QSVMKernel.BATCH_SIZE):
            circuits = {}
            to_be_simulated_circuits = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                futures = {}
                for sub_idx in range(idx, min(idx + QSVMKernel.BATCH_SIZE, len(mus))):
                    i = mus[sub_idx]
                    j = nus[sub_idx]
                    x1 = x1_vec[i]
                    x2 = x2_vec[j]
                    if not np.all(x1 == x2):
                        futures["{}:{}".format(i, j)] = \
                            executor.submit(QSVMKernel._construct_circuit,
                                            x1, x2, self.num_qubits, self.feature_map,
                                            measurement, "circuit{}:{}".format(i, j))

                for k, v in futures.items():
                    circuit = v.result()
                    circuits[k] = circuit
                    to_be_simulated_circuits.append(circuit)

            results = self.quantum_instance.execute(to_be_simulated_circuits)
            kernel_values = {}

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
                for idx, circuit in circuits.items():
                    kernel_values[idx] = executor.submit(QSVMKernel._compute_overlap,
                                                         results, circuit, is_statevector_sim,
                                                         measurement_basis)
                for k, v in kernel_values.items():
                    i, j = [int(x) for x in k.split(":")]
                    mat[i, j] = v.result()
                    if is_symmetric:
                        mat[j, i] = mat[i, j]
        return mat

    def train(self, data, labels, quantum_instance=None):
        """
        Train the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data
            quantum_instance (QuantumInstance): quantum backend with all setting
        """
        self._quantum_instance = self._quantum_instance \
            if quantum_instance is None else quantum_instance
        self.instance.train(data, labels)

    def test(self, data, labels, quantum_instance=None):
        """
        Test the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            labels (numpy.ndarray): Nx1 array, where N is the number of data
            quantum_instance (QuantumInstance): quantum backend with all setting
        Returns:
            float: accuracy
        """
        self._quantum_instance = self._quantum_instance \
            if quantum_instance is None else quantum_instance
        return self.instance.test(data, labels)

    def predict(self, data, quantum_instance=None):
        """
        Predict using the svm.

        Args:
            data (numpy.ndarray): NxD array, where N is the number of data,
                                  D is the feature dimension.
            quantum_instance (QuantumInstance): quantum backend with all setting
        Returns:
            numpy.ndarray: predicted labels, Nx1 array
        """
        self._quantum_instance = self._quantum_instance \
            if quantum_instance is None else quantum_instance
        return self.instance.predict(data)

    def _run(self):
        return self.instance.run()

    @property
    def ret(self):
        return self.instance.ret

    @ret.setter
    def ret(self, new_value):
        self.instance.ret = new_value

    def load_model(self, file_path):
        """Load a model from a file path.

        Args:
            file_path (str): tthe path of the saved model.
        """
        self.instance.load_model(file_path)

    def save_model(self, file_path):
        """Save the model to a file path.

        Args:
            file_path (str): a path to save the model.
        """
        self.instance.save_model(file_path)
