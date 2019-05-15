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

import logging
import sys

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.aqua import aqua_globals
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.algorithms.many_sample.qsvm._qsvm_binary import _QSVM_Binary
from qiskit.aqua.algorithms.many_sample.qsvm._qsvm_multiclass import _QSVM_Multiclass
from qiskit.aqua.algorithms.many_sample.qsvm._qsvm_estimator import _QSVM_Estimator
from qiskit.aqua.utils.dataset_helper import get_feature_dimension, get_num_classes
from qiskit.aqua.utils import split_dataset_to_data_and_labels

logger = logging.getLogger(__name__)


class QSVM(QuantumAlgorithm):
    """
    Quantum SVM method.

    Internally, it will run the binary classification or multiclass classification
    based on how many classes the data have.
    """

    CONFIGURATION = {
        'name': 'QSVM',
        'description': 'QSVM Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'QSVM_schema',
            'type': 'object',
            'properties': {
            },
            'additionalProperties': False
        },
        'problems': ['classification'],
        'depends': [
            {'pluggable_type': 'multiclass_extension'},
            {'pluggable_type': 'feature_map',
             'default': {
                 'name': 'SecondOrderExpansion',
                 'depth': 2
             }
             },
        ],
    }

    BATCH_SIZE = 1000

    def __init__(self, feature_map, training_dataset, test_dataset=None, datapoints=None,
                 multiclass_extension=None):
        """Constructor.

        Args:
            feature_map (FeatureMap): feature map module, used to transform data
            training_dataset (dict): training dataset.
            test_dataset (Optional[dict]): testing dataset.
            datapoints (Optional[numpy.ndarray]): prediction dataset.
            multiclass_extension (Optional[MultiExtension]): if number of classes > 2 then
                a multiclass scheme is needed.

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

        if datapoints is not None and not isinstance(datapoints, np.ndarray):
            datapoints = np.asarray(datapoints)
        self.datapoints = datapoints

        self.feature_map = feature_map
        self.num_qubits = self.feature_map.num_qubits

        if multiclass_extension is None:
            qsvm_instance = _QSVM_Binary(self)
        else:
            qsvm_instance = _QSVM_Multiclass(self, multiclass_extension)

        self.instance = qsvm_instance

    @classmethod
    def init_params(cls, params, algo_input):
        """Constructor from params."""
        feature_dimension = get_feature_dimension(algo_input.training_dataset)
        fea_map_params = params.get(Pluggable.SECTION_KEY_FEATURE_MAP)
        fea_map_params['feature_dimension'] = feature_dimension

        feature_map = get_pluggable_class(PluggableType.FEATURE_MAP,
                                          fea_map_params['name']).init_params(params)

        multiclass_extension = None
        multiclass_extension_params = params.get(Pluggable.SECTION_KEY_MULTICLASS_EXTENSION)
        if multiclass_extension_params is not None:
            multiclass_extension_params['params'] = [feature_map]
            multiclass_extension_params['estimator_cls'] = _QSVM_Estimator

            multiclass_extension = get_pluggable_class(PluggableType.MULTICLASS_EXTENSION,
                                                       multiclass_extension_params['name']).init_params(params)
            logger.info("Multiclass classifier based on {}".format(multiclass_extension_params['name']))

        return cls(feature_map, algo_input.training_dataset, algo_input.test_dataset,
                   algo_input.datapoints, multiclass_extension)

    @staticmethod
    def _construct_circuit(x, num_qubits, feature_map, measurement):
        x1, x2 = x
        if x1.shape[0] != x2.shape[0]:
            raise ValueError("x1 and x2 must be the same dimension.")

        q = QuantumRegister(num_qubits, 'q')
        c = ClassicalRegister(num_qubits, 'c')
        qc = QuantumCircuit(q, c)
        # write input state from sample distribution
        qc += feature_map.construct_circuit(x1, q)
        qc += feature_map.construct_circuit(x2, q, inverse=True)
        if measurement:
            qc.barrier(q)
            qc.measure(q, c)
        return qc

    @staticmethod
    def _compute_overlap(idx, results, is_statevector_sim, measurement_basis):
        if is_statevector_sim:
            temp = results.get_statevector(idx)[0]
            #  |<0|Psi^daggar(y) x Psi(x)|0>|^2,
            kernel_value = np.dot(temp.T.conj(), temp).real
        else:
            result = results.get_counts(idx)
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
        return QSVM._construct_circuit((x1, x2), self.num_qubits,
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
        from .qsvm import QSVM

        if x2_vec is None:
            is_symmetric = True
            x2_vec = x1_vec
        else:
            is_symmetric = False

        is_statevector_sim = self.quantum_instance.is_statevector
        measurement = not is_statevector_sim
        measurement_basis = '0' * self.num_qubits
        mat = np.ones((x1_vec.shape[0], x2_vec.shape[0]))

        # get all indices
        if is_symmetric:
            mus, nus = np.triu_indices(x1_vec.shape[0], k=1)  # remove diagonal term
        else:
            mus, nus = np.indices((x1_vec.shape[0], x2_vec.shape[0]))
            mus = np.asarray(mus.flat)
            nus = np.asarray(nus.flat)

        for idx in range(0, len(mus), QSVM.BATCH_SIZE):
            to_be_computed_list = []
            to_be_computed_index = []
            for sub_idx in range(idx, min(idx + QSVM.BATCH_SIZE, len(mus))):
                i = mus[sub_idx]
                j = nus[sub_idx]
                x1 = x1_vec[i]
                x2 = x2_vec[j]
                if not np.all(x1 == x2):
                    to_be_computed_list.append((x1, x2))
                    to_be_computed_index.append((i, j))

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Building circuits:")
                TextProgressBar(sys.stderr)
            circuits = parallel_map(QSVM._construct_circuit,
                                    to_be_computed_list,
                                    task_args=(self.num_qubits, self.feature_map,
                                               measurement),
                                    num_processes=aqua_globals.num_processes)

            results = self.quantum_instance.execute(circuits)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Calculating overlap:")
                TextProgressBar(sys.stderr)
            matrix_elements = parallel_map(QSVM._compute_overlap, range(len(circuits)),
                                           task_args=(results, is_statevector_sim, measurement_basis),
                                           num_processes=aqua_globals.num_processes)

            for idx in range(len(to_be_computed_index)):
                i, j = to_be_computed_index[idx]
                mat[i, j] = matrix_elements[idx]
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
