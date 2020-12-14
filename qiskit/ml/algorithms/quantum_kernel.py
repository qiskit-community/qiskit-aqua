# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Quantum Kernel Algorithm"""

from typing import Optional, Union

import logging

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit.providers import Backend, BaseBackend

from qiskit.aqua import QuantumInstance, AquaError

logger = logging.getLogger(__name__)


class QuantumKernel:
    """
    Quantum Kernel

    The general task of machine learning is to find and study patterns in data. For many
    algorithms, the datapoints are better understood in a higher dimensional feature space,
    through the use of a kernel function:

    K(x, y) = <f(x), f(y)>.

    Here K is the kernel function, x, y are n dimensional inputs. f is a map from n-dimension
    to m-dimension space. <x,y> denotes the dot product. Usually m is much larger than n.

    The quantum kernel algorithm calculates a kernel matrix, given datapoints x and y and feature
    map f, all of n dimension. This kernel matrix can then be used in classical machine learning
    algorithms such as support vector classification, spectral clustering or ridge regression.
    """

    BATCH_SIZE = 1000

    def __init__(self,
                 feature_map: QuantumCircuit,
                 enforce_psd: bool = False,
                 quantum_instance: Optional[
                     Union[QuantumInstance, BaseBackend, Backend]] = None) -> None:
        """
        Args
            feature_map:       # parameterized circuit to be used as the feature map
            enforce_psd:       # project to closest positive semidefinite matrix if x = y
            quantum_instance:  # Quantum Instance or Backend
        """

        self._feature_map = feature_map
        self._enforce_psd = enforce_psd
        self._quantum_instance = quantum_instance

    @property
    def feature_map(self) -> QuantumCircuit:
        """ Returns feature map """
        return self._feature_map

    @feature_map.setter
    def feature_map(self, feature_map: QuantumCircuit):
        """ Sets feature map """
        self._feature_map = feature_map

    @property
    def quantum_instance(self) -> QuantumInstance:
        """ Returns quantum instance """
        return self._quantum_instance

    @quantum_instance.setter
    def quantum_instance(self, quantum_instance: Union[Backend,
                                                       BaseBackend, QuantumInstance]) -> None:
        """ Sets quantum instance """
        if isinstance(quantum_instance, (BaseBackend, Backend)):
            self._quantum_instance = QuantumInstance(quantum_instance)
        else:
            self._quantum_instance = quantum_instance

    def _construct_circuit(self, x, y, is_statevector_sim=False):
        """
        Helper function to generate inner product circuit for the given feature map.
        """
        q = QuantumRegister(self._feature_map.num_qubits, 'q')
        c = ClassicalRegister(self._feature_map.num_qubits, 'c')
        qc = QuantumCircuit(q, c)

        x_dict = dict(zip(self._feature_map.parameters, x))
        psi_x = self._feature_map.assign_parameters(x_dict)
        qc.append(psi_x.to_instruction(), qc.qubits)

        if not is_statevector_sim:
            y_dict = dict(zip(self._feature_map.parameters, y))
            psi_y_dag = self._feature_map.assign_parameters(y_dict)
            qc.append(psi_y_dag.to_instruction().inverse(), qc.qubits)

            qc.barrier(q)
            qc.measure(q, c)
        return qc

    def _compute_overlap(self, idx, results, is_statevector_sim):
        """
        Helper function to compute overlap for given input.
        """
        if is_statevector_sim:
            i, j = idx
            v_a = results.get_statevector(int(i))
            v_b = results.get_statevector(int(j))
            # |<0|Psi^daggar(y) x Psi(x)|0>|^2, take the amplitude
            tmp = np.vdot(v_a, v_b)
            kernel_value = np.vdot(tmp, tmp).real  # pylint: disable=no-member
        else:
            result = results.get_counts(idx)

            measurement_basis = '0' * self._feature_map.num_qubits
            kernel_value = result.get(measurement_basis, 0) / sum(result.values())
        return kernel_value

    def evaluate(self, x_vec, y_vec=None):
        """
        Construct kernel matrix for given data and feature map

        If y is None, self inner product is calculated.
        If using `statevector_simulator`, only build circuits for  Phi(x)|0>, then perform inner
        product classically.

        Args:
            x_vec (numpy.ndarray): 2D array of datapoints, NxD, where N is the number of datapoints,
                               D is the feature dimension
            y_vec (numpy.ndarray): 2D array of datapoints, MxD, where M is the number of datapoints,
                               D is the feature dimension

        Returns:
            numpy.ndarray: 2-D matrix, NxM

        Raises:
            AquaError:
                - A quantum instance or backend has not been provided to the class
            ValueError:
                - x_vec and/or y_vec are not two dimensional arrays
                - x_vec and/or y_vec have incompatible dimension with feature map
        """
        if self._quantum_instance is None:
            raise AquaError("A QuantumInstance or Backend "
                            "must be supplied to run the quantum kernel.")
        if isinstance(self._quantum_instance, (BaseBackend, Backend)):
            self._quantum_instance = self.quantum_instance(self._quantum_instance)

        if x_vec.ndim != 2:
            raise ValueError("x_vec must be a 2D array")

        if y_vec is not None and y_vec.ndim != 2:
            raise ValueError("y_vec must be a 2D array")

        if x_vec.shape[1] != self._feature_map.num_parameters:
            raise ValueError("x_vec and class feature map incompatible dimensions.\n" +
                             "x_vec has %s dimensions, but feature map has %s." %
                             (x_vec.shape[1], self._feature_map.num_parameters))

        if y_vec is not None and y_vec.shape[1] != self._feature_map.num_parameters:
            raise ValueError("y_vec and class feature map incompatible dimensions.\n" +
                             "y_vec has %s dimensions, but feature map has %s." %
                             (y_vec.shape[1], self._feature_map.num_parameters))

        # determine if calculating self inner product
        is_symmetric = True
        if y_vec is None:
            y_vec = x_vec
        elif not np.array_equal(x_vec, y_vec):
            is_symmetric = False

        # initialize kernel matrix
        kernel = np.zeros((x_vec.shape[0], y_vec.shape[0]))

        # set diagonal to 1 if symmetric
        if is_symmetric:
            np.fill_diagonal(kernel, 1)

        # get indices to calculate
        if is_symmetric:
            mus, nus = np.triu_indices(x_vec.shape[0], k=1)  # remove diagonal
        else:
            mus, nus = np.indices((x_vec.shape[0], y_vec.shape[0]))
            mus = np.asarray(mus.flat)
            nus = np.asarray(nus.flat)

        is_statevector_sim = self._quantum_instance.is_statevector

        # calculate kernel
        if is_statevector_sim:  # using state vector simulator
            if is_symmetric:
                to_be_computed_data = x_vec
            else:  # not symmetric
                to_be_computed_data = np.concatenate((x_vec, y_vec))

            feature_map_params = ParameterVector('par_x', self._feature_map.num_parameters)
            parameterized_circuit = self._construct_circuit(
                feature_map_params, feature_map_params,
                is_statevector_sim=is_statevector_sim)
            parameterized_circuit = self._quantum_instance.transpile(parameterized_circuit)[0]
            circuits = [parameterized_circuit.assign_parameters({feature_map_params: x})
                        for x in to_be_computed_data]

            results = self._quantum_instance.execute(circuits)

            offset = 0 if is_symmetric else len(x_vec)
            matrix_elements = [self._compute_overlap(idx, results, is_statevector_sim)
                               for idx in list(zip(mus, nus + offset))]

            # using parallel_map causes an error
            # matrix_elements = parallel_map(self._compute_overlap,
            #                                list(zip(mus, nus + offset)),
            #                                task_args=(results,is_statevector_sim),
            #                                num_processes=aqua_globals.num_processes)

            for i, j, value in zip(mus, nus, matrix_elements):
                kernel[i, j] = value
                if is_symmetric:
                    kernel[j, i] = kernel[i, j]

        else:  # not using state vector simulator
            feature_map_params_x = ParameterVector('par_x', self._feature_map.num_parameters)
            feature_map_params_y = ParameterVector('par_y', self._feature_map.num_parameters)
            parameterized_circuit = self._construct_circuit(
                feature_map_params_x, feature_map_params_y,
                is_statevector_sim=is_statevector_sim)
            parameterized_circuit = self._quantum_instance.transpile(parameterized_circuit)[0]

            for idx in range(0, len(mus), QuantumKernel.BATCH_SIZE):
                to_be_computed_data_pair = []
                to_be_computed_index = []
                for sub_idx in range(idx, min(idx + QuantumKernel.BATCH_SIZE, len(mus))):
                    i = mus[sub_idx]
                    j = nus[sub_idx]
                    xi = x_vec[i]  # pylint: disable=invalid-name
                    yj = y_vec[j]  # pylint: disable=invalid-name
                    if not np.all(xi == yj):
                        to_be_computed_data_pair.append((xi, yj))
                        to_be_computed_index.append((i, j))

                circuits = [parameterized_circuit.assign_parameters({feature_map_params_x: x,
                                                                     feature_map_params_y: y})
                            for x, y in to_be_computed_data_pair]

                results = self._quantum_instance.execute(circuits)

                matrix_elements = [self._compute_overlap(circuit, results, is_statevector_sim)
                                   for circuit in range(len(circuits))]

                # using parallel_map causes an error
                # matrix_elements = parallel_map(QuantumKernel._compute_overlap,
                #                                range(len(circuits)),
                #                                task_args=(results,is_statevector_sim),
                #                                num_processes=aqua_globals.num_processes)

                for (i, j), value in zip(to_be_computed_index, matrix_elements):
                    kernel[i, j] = value
                    if is_symmetric:
                        kernel[j, i] = kernel[i, j]

            if self._enforce_psd and is_symmetric:
                # Find the closest positive semi-definite approximation to symmetric kernel matrix.
                # The (symmetric) matrix should always be positive semi-definite by construction,
                # but this can be violated in case of noise, such as sampling noise, thus the
                # adjustment is only done if NOT using the statevector simulation.
                D, U = np.linalg.eig(kernel)  # pylint: disable=invalid-name
                kernel = U @ np.diag(np.maximum(0, D)) @ U.transpose()

        return kernel
