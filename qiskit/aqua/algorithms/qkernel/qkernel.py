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
"""
Quantum Kernel Class, for use in Quantum Kernel Algorithms.

"""

import logging
import numpy as np

from qiskit import ClassicalRegister
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from sklearn import preprocessing
from sklearn.preprocessing import KernelCenterer
from sklearn.metrics.pairwise import pairwise_kernels
from itertools import permutations

logger = logging.getLogger(__name__)


class QKernel():

    def __init__(self, construct_circuit_fn=None, num_qubits=None, quantum_instance=None, measurement_edit_distance=0):
        self.construct_circuit_fn = construct_circuit_fn
        self.num_qubits = num_qubits
        self.quantum_instance = quantum_instance
        self.measurement_edit_distance = measurement_edit_distance
        self.kernel_matrix = None
        self.counts = None

    BATCH_SIZE = 1000

    # @staticmethod
    def _construct_circuit(self, x, construct_circuit_fn):

        x1, x2 = x

        if x1.shape[0] != x2.shape[0]:
            raise ValueError("x1 and x2 must be the same dimension.")

        qc = construct_circuit_fn(x1) + construct_circuit_fn(x2).inverse()
        qc.barrier()
        qc.add_register(ClassicalRegister(qc.width(), 'c'))
        qc.measure(qc.qregs[0], qc.cregs[0])
        return qc

    def _compute_overlap(self, idx, results, measurement_basis):
        result = results.get_counts(idx)
        shots_in_basis = 0
        for basis_str in measurement_basis:
            shots_in_basis += result.get(basis_str, 0)
        kernel_value = shots_in_basis / sum(result.values())
        return kernel_value

    def construct_kernel_matrix(self, x1_vec,
                                x2_vec=None,
                                quantum_instance=None,
                                calculate_diags=False,
                                preserve_counts=False,
                                save_as_kernel=True):
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

        if x2_vec is None:
            is_symmetric = True
            x2_vec = x1_vec
        else:
            is_symmetric = False

        if not quantum_instance:
            if self.quantum_instance:
                quantum_instance = self.quantum_instance
            else:
                raise ValueError('quantum_instance must be set in either constructor or construct_kernel_matrix fn')

        measurement_basis = ['0' * self.num_qubits]
        for ones in range(1, self.measurement_edit_distance+1):
            zeros = self.num_qubits - ones
            measurement_basis += [''.join(i) for i in set(permutations(('0'*zeros) + ('1'*ones)))]
        # TODO pull bit flip probabilities from device and weight by probs of n flips.

        mat = np.ones((x1_vec.shape[0], x2_vec.shape[0]))
        if preserve_counts:
            self.counts = [[{} for x in range(x1_vec.shape[0])] for y in range(x2_vec.shape[0])]

        # get all indices
        if is_symmetric:
            k = 0 if calculate_diags else 1
            mus, nus = np.triu_indices(x1_vec.shape[0], k=k)
        else:
            mus, nus = np.indices((x1_vec.shape[0], x2_vec.shape[0]))
            mus = np.asarray(mus.flat)
            nus = np.asarray(nus.flat)

        for idx in range(0, len(mus), QKernel.BATCH_SIZE):
            to_be_computed_list = []
            to_be_computed_index = []
            for sub_idx in range(idx, min(idx + QKernel.BATCH_SIZE, len(mus))):
                i = mus[sub_idx]
                j = nus[sub_idx]
                x1 = x1_vec[i]
                x2 = x2_vec[j]
                if not i == j or calculate_diags:
                    to_be_computed_list.append((x1, x2))
                    to_be_computed_index.append((i, j))

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Building circuits:")
                TextProgressBar(sys.stderr)

            circuits = parallel_map(self._construct_circuit,
                                    to_be_computed_list,
                                    task_kwargs={'construct_circuit_fn': self.construct_circuit_fn})

            results = quantum_instance.execute(circuits)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Calculating overlap:")
                TextProgressBar(sys.stderr)
            matrix_elements = parallel_map(self._compute_overlap, range(len(circuits)),
                                           task_args=(results, measurement_basis))

            for idx in range(len(to_be_computed_index)):
                i, j = to_be_computed_index[idx]
                mat[i, j] = matrix_elements[idx]
                if is_symmetric:
                    mat[j, i] = mat[i, j]
                if preserve_counts:
                    self.counts[i][j] = results.get_counts(idx)
                    if is_symmetric:
                        self.counts[j][i] = self.counts[i][j]

        if save_as_kernel:
            self.kernel_matrix = mat
        return mat

    # Warning, very slow
    def callable_overlap(self, X, Y, quantum_instance, gamma=1.0):
        circ = self._construct_circuit((X, Y), self.construct_circuit_fn)
        results = quantum_instance.execute(circ)
        measurement_basis = ['0' * self.num_qubits]
        for ones in range(1, self.measurement_edit_distance+1):
            zeros = self.num_qubits - ones
            measurement_basis += [''.join(i) for i in set(permutations(('0'*zeros) + ('1'*ones)))]
        overlap = self._compute_overlap(circ, results=results, measurement_basis=measurement_basis)
        return overlap * gamma

    @property
    def distance_matrix(self):
        return np.sqrt(1 - self.kernel_matrix)

    def normalize_matrix(self, norm='l1'):
        self.kernel_matrix = preprocessing.normalize(self.kernel_matrix, norm=norm)
        return self.kernel_matrix

    def scale_matrix(self):
        self.kernel_matrix = preprocessing.scale(self.kernel_matrix)
        return self.kernel_matrix

    def center_matrix(self, metric='linear'):
        K = pairwise_kernels(self.kernel_matrix, metric=metric)
        transformer = KernelCenterer().fit(K)
        self.kernel_matrix = transformer.transform(K)
        return self.kernel_matrix