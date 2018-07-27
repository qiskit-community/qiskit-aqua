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

import copy
import itertools
from functools import reduce
import logging
import sys

import numpy as np
from scipy import sparse as scisparse
from scipy import linalg as scila
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.wrapper import execute as q_execute
from qiskit.tools.qi.pauli import Pauli, label_to_pauli, sgn_prod
from qiskit.qasm import pi

from qiskit_aqua import AlgorithmError
from qiskit_aqua.utils import PauliGraph, summarize_circuits

logger = logging.getLogger(__name__)


class Operator(object):

    MAX_CIRCUITS_PER_JOB = 300

    """
    Operators relevant for quantum applications

    Note:
        For grouped paulis represnetation, all operations will always convert it to paulis and then convert it back.
        (It might be a performance issue.)
    """

    def __init__(self, paulis=None, grouped_paulis=None, matrix=None, coloring="largest-degree"):
        """
        Args:
            paulis ([[float, Pauli]]): each list contains a coefficient (real number) and a corresponding Pauli class object.
            grouped_paulis ([[[float, Pauli]]]): each list of list contains a grouped paulis.
            matrix (numpy.ndarray or scipy.sparse.csr_matrix) : a 2-D sparse matrix represents operator (using CSR format internally)
            coloring (bool): method to group paulis.
        """
        self._paulis = paulis
        self._coloring = coloring
        self._grouped_paulis = grouped_paulis
        if matrix is not None:
            matrix = matrix if scisparse.issparse(matrix) else scisparse.csr_matrix(matrix)
            matrix = matrix if scisparse.isspmatrix_csr(matrix) else matrix.to_csr(copy=True)

        self._matrix = matrix
        self._to_dia_matrix(mode="matrix")

        # use for fast lookup whether or not the paulis is existed.
        self._simplify_paulis()

        self._summarize_circuits = False

    def _add_extend_or_combine(self, rhs, mode):
        """
        Add two operators either extend (in-place) or combine (copy) them.
        The addition performs optimized combiniation of two operators.
        If `rhs` has identical basis, the coefficient are combined rather than
        appended.

        Args:
            rhs (Operator): to-be-combined operator
            mode (str): in-place or not.

        Returns:
            Operator: the operator.
        """
        result_paulis = None
        result_grouped_paulis = None
        result_matrix = None

        if mode == 'inplace':
            lhs = self
        elif mode == 'non-inplace':
            lhs = copy.deepcopy(self)

        if lhs._paulis is not None and rhs._paulis is not None:
            for pauli in rhs._paulis:
                pauli_label = pauli[1].to_label()
                idx = lhs._paulis_table.get(pauli_label, None)
                if idx is not None:
                    lhs._paulis[idx][0] += pauli[0]
                else:
                    lhs._paulis_table[pauli_label] = len(lhs._paulis)
                    lhs._paulis.append(pauli)
            result_paulis = lhs._paulis
        elif lhs._grouped_paulis is not None and rhs._grouped_paulis is not None:
            lhs._grouped_paulis_to_paulis()
            rhs._grouped_paulis_to_paulis()
            lhs = lhs + rhs
            lhs._paulis_to_grouped_paulis()
            result_grouped_paulis = lhs._grouped_paulis
        elif lhs._matrix is not None and rhs._matrix is not None:
            lhs._matrix = lhs._matrix + rhs._matrix
            result_matrix = lhs._matrix
        else:
            raise TypeError("the representations of two Operators should be the same. ({}, {})".format(
                lhs.representations, rhs.representations))

        return lhs

    def __add__(self, rhs):
        """Overload + operation"""
        return self._add_extend_or_combine(rhs, 'non-inplace')

    def __iadd__(self, rhs):
        """Overload += operation"""
        return self._add_extend_or_combine(rhs, 'inplace')

    def __eq__(self, rhs):
        """Overload == operation"""
        if self._matrix is not None and rhs._matrix is not None:
            return np.all(self._matrix == rhs._matrix)
        if self._paulis is not None and rhs._paulis is not None:
            if len(self._paulis) != len(rhs._paulis):
                return False
            for coeff, pauli in self._paulis:
                found_pauli = False
                rhs_coeff = 0.0
                for coeff2, pauli2 in rhs._paulis:
                    if pauli == pauli2:
                        found_pauli = True
                        rhs_coeff = coeff2
                        break
                if found_pauli == False and rhs_coeff != 0.0:  # since we might have 0 weights of paulis.
                    return False
                if coeff != rhs_coeff:
                    return False
            return True
        if self._grouped_paulis is not None and rhs._grouped_paulis is not None:
            self._grouped_paulis_to_paulis()
            rhs._grouped_paulis_to_paulis()
            return self.__eq__(rhs)

    def __ne__(self, rhs):
        return not self.__eq__(rhs)

    def chop(self, threshold=1e-15):
        """
        Eliminate the real and imagine part of coeff in each pauli by `threshold`.
        If pauli's coeff is less then `threshold` in both real and imagine parts, the pauli is removed.
        To align the internal representations, all available representations are chopped.
        The chopped result is stored back to original property.
        Note: if coeff is real-only, the imag part is skipped.

        Args:
            threshold (float): threshold chops the paulis
        """
        def chop_real_imag(coeff, threshold):
            temp_real = coeff.real if np.absolute(coeff.real) >= threshold else 0.0
            temp_imag = coeff.imag if np.absolute(coeff.imag) >= threshold else 0.0
            if temp_real == 0.0 and temp_imag == 0.0:
                return 0.0
            else:
                new_coeff = temp_real + 1j * temp_imag
                return new_coeff

        if self._paulis is not None:
            for i in range(len(self._paulis)):
                self._paulis[i][0] = chop_real_imag(self._paulis[i][0], threshold)
            paulis = [x for x in self._paulis if x[0] != 0.0]
            self._paulis = paulis
            self._paulis_table = {pauli[1].to_label(): i for i, pauli in enumerate(self._paulis)}

        if self._grouped_paulis is not None:
            grouped_paulis = []
            for group_idx in range(1, len(self._grouped_paulis)):
                for pauli_idx in range(len(self._grouped_paulis[group_idx])):
                    self._grouped_paulis[group_idx][pauli_idx][0] = chop_real_imag(
                        self._grouped_paulis[group_idx][pauli_idx][0], threshold)
                paulis = [x for x in self._grouped_paulis[group_idx] if x[0] != 0.0]
                grouped_paulis.append(paulis)
            self._grouped_paulis = grouped_paulis

        if self._matrix is not None:
            rows, cols = self._matrix.nonzero()
            for row, col in zip(rows, cols):
                self._matrix[row, col] = chop_real_imag(self._matrix[row, col], threshold)
            self._matrix.eliminate_zeros()

        if self._dia_matrix is not None:
            temp_real = self._dia_matrix.real
            temp_imag = self._dia_matrix.imag
            real_chopped_idx = np.absolute(temp_real) < threshold
            imag_chopped_idx = np.absolute(temp_imag) < threshold
            temp_real[real_chopped_idx] = 0.0
            temp_imag[imag_chopped_idx] = 0.0
            self._dia_matrix = temp_real + 1j * temp_imag

    def _simplify_paulis(self):
        """
        Merge the paulis (grouped_paulis) whose bases are identical but the pauli with zero coefficient
        would not be removed.

        Usually used in construction.
        """
        if self._paulis is not None:
            new_paulis = []
            new_paulis_table = {}
            for curr_paulis in self._paulis:
                pauli_label = curr_paulis[1].to_label()
                new_idx = new_paulis_table.get(pauli_label, None)
                if new_idx is not None:
                    new_paulis[new_idx][0] += curr_paulis[0]
                else:
                    new_paulis_table[pauli_label] = len(new_paulis)
                    new_paulis.append(curr_paulis)

            self._paulis = new_paulis
            self._paulis_table = new_paulis_table

        elif self._grouped_paulis is not None:
            self._grouped_paulis_to_paulis()
            self._simplify_paulis()
            self._paulis_to_grouped_paulis()

    def __mul__(self, rhs):
        """
        Overload * operation. Only support two Operators have the same representation mode.

        Returns:
            Operator: the multipled Operator.

        Raises:
            TypeError, if two Operators do not have the same representations.
        """
        if self._paulis is not None and rhs._paulis is not None:
            ret_pauli = Operator(paulis=[])
            for existed_pauli in self._paulis:
                for pauli in rhs._paulis:
                    basis, sign = sgn_prod(existed_pauli[1], pauli[1])
                    coeff = existed_pauli[0] * pauli[0] * sign
                    pauli_term = [coeff, basis]
                    ret_pauli += Operator(paulis=[pauli_term])
            return ret_pauli

        elif self._grouped_paulis is not None and rhs._grouped_paulis is not None:
            self._grouped_paulis_to_paulis()
            rhs._grouped_paulis_to_paulis()
            mul_pauli = self * rhs
            mul_pauli._paulis_to_grouped_paulis()
            ret_grouped_pauli = Operator(paulis=mul_pauli._paulis, grouped_paulis=mul_pauli._grouped_paulis)
            return ret_grouped_pauli

        elif self._matrix is not None and rhs._matrix is not None:
            ret_matrix = self._matrix.dot(rhs._matrix)
            return Operator(matrix=ret_matrix)
        else:
            raise TypeError("the representations of two Operators should be the same. ({}, {})".format(
                self.representations, rhs.representations))

    @property
    def coloring(self):
        """Getter of method of grouping paulis"""
        return self._coloring

    @coloring.setter
    def coloring(self, new_coloring):
        """Setter of method of grouping paulis"""
        self._coloring = new_coloring

    def _to_dia_matrix(self, mode):
        """
        Convert the reprenetations into diagonal matrix if possible and then store it back.
        For paulis, if all paulis are Z or I (identity), convert to dia_matrix.

        Args:
            mode (str): "matrix", "paulis" or "grouped_paulis".
        """
        if mode not in ['matrix', 'paulis', 'grouped_paulis']:
            raise ValueError(
                'Mode should be one of "matrix", "paulis", "grouped_paulis"')

        if mode == 'matrix' and self._matrix is not None:
            dia_matrix = self._matrix.diagonal()
            if not scisparse.csr_matrix(dia_matrix).nnz == self._matrix.nnz:
                dia_matrix = None
            self._dia_matrix = dia_matrix

        elif mode == 'paulis' and self._paulis is not None:
            if self._paulis == []:
                self._dia_matrix = None
            else:
                valid_dia_matrix_flag = True
                dia_matrix = 0.0
                for idx in range(len(self._paulis)):
                    coeff, pauli = self._paulis[idx][0], self._paulis[idx][1]
                    if not (np.all(pauli.w == 0)):
                        valid_dia_matrix_flag = False
                        break
                    dia_matrix += coeff * pauli.to_spmatrix().diagonal()
                self._dia_matrix = dia_matrix.copy() if valid_dia_matrix_flag else None

        elif mode == 'grouped_paulis' and self._grouped_paulis is not None:
            self._grouped_paulis_to_paulis()
            self._to_dia_matrix(mode='paulis')

        else:
            self._dia_matrix = None

    @property
    def paulis(self):
        """Getter of Pauli list."""
        return self._paulis

    @property
    def grouped_paulis(self):
        """Getter of grouped Pauli list."""
        return self._grouped_paulis

    @property
    def matrix(self):
        """Getter of matrix; if matrix is diagonal, diagonal matrix is returned instead."""
        return self._dia_matrix if self._dia_matrix is not None else self._matrix

    def enable_summarize_circuits(self):
        self._summarize_circuits = True

    def disable_summarize_circuits(self):
        self._summarize_circuits = False

    @property
    def representations(self):
        """
        Return the available represnetations in the Operator.

        Returns:
            list: available representations ([str])
        """
        ret = []
        if self._paulis is not None:
            ret.append("paulis")
        if self._grouped_paulis is not None:
            ret.append("grouped_paulis")
        if self._matrix is not None:
            ret.append("matrix")
        return ret

    @property
    def num_qubits(self):
        """
        number of qubits required for the operator.

        Returns:
            int: number of qubits

        """
        if self._paulis is not None:
            if self._paulis != []:
                return len(self._paulis[0][1].v)
            else:
                return 0
        elif self._grouped_paulis is not None and self._grouped_paulis != []:
            return len(self._grouped_paulis[0][0][1].v)
        else:
            return int(np.log2(self._matrix.shape[0]))

    @staticmethod
    def load_from_file(file_name):
        """
        Load paulis in a file to construct an Operator, only support paulis as an input and its coefficient is real.
        E.g.:
            IIII
            0.34511
            ZZZZ
            0.31256
            XXYY
            5.84215
            ...

        Args:
            file_name (str): path to the file, which contains a list of Paulis and coefficients.

        Returns:
            Operator class: the loaded operator.

        Note:
            Do we need to support complex coefficient? If so, what is the format?
        """
        with open(file_name, 'r+') as file:
            ham_array = file.readlines()
        ham_array = [x.strip() for x in ham_array]
        paulis = [[float(ham_array[2 * i + 1]), label_to_pauli(ham_array[2 * i])]
                  for i in range(len(ham_array) // 2)]

        return Operator(paulis=paulis)

    def save_to_file(self, file_name):
        """
        Save operator to a file in pauli representation.

        Args:
            file_name (str): path to the file

        """
        with open(file_name, 'w') as f:
            self._check_representation("paulis")
            for pauli in self._paulis:
                print("{}".format(pauli[1].to_label()), file=f)
                print("{}".format(pauli[0]), file=f)

    @staticmethod
    def load_from_dict(dictionary):
        """
        Load paulis in a dict to construct an Operator,
        the dict must be represented as follows: label and coeff (real and imag).
        E.g.:
           {'paulis':
               [
                   {'label': 'IIII',
                    'coeff': {'real': -0.33562957575267038, 'imag': 0.0}},
                   {'label': 'ZIII',
                    'coeff': {'real': 0.28220597164664896, 'imag': 0.0}},
                    ...
                ]
            }

        Args:
            dictionary (dict): dictionary, which contains a list of Paulis and coefficients.

        Returns:
            Operator: the loaded operator.
        """
        if 'paulis' not in dictionary:
            raise AlgorithmError('Dictionary missing "paulis" key')

        paulis = []
        for op in dictionary['paulis']:
            if 'label' not in op:
                raise AlgorithmError('Dictionary missing "label" key')

            pauli_label = op['label']
            if 'coeff' not in op:
                raise AlgorithmError('Dictionary missing "coeff" key')

            pauli_coeff = op['coeff']
            if 'real' not in pauli_coeff:
                raise AlgorithmError('Dictionary missing "real" key')

            coeff = pauli_coeff['real']
            if 'imag' in pauli_coeff:
                coeff = complex(pauli_coeff['real'], pauli_coeff['imag'])

            paulis.append([coeff, label_to_pauli(pauli_label)])

        return Operator(paulis=paulis)

    def save_to_dict(self):
        """
        Save operator to a dict in pauli represnetation.

        Returns:
            dict: a dictionary contains an operator with pauli representation.
        """
        self._check_representation("paulis")
        ret_dict = {'paulis': []}
        for pauli in self._paulis:
            op = {'label': pauli[1].to_label()}
            if isinstance(pauli[0], complex):
                op['coeff'] = {'real': np.real(pauli[0]),
                               'imag': np.imag(pauli[0])
                               }
            else:
                op['coeff'] = {'real': pauli[0]}

            ret_dict['paulis'].append(op)

        return ret_dict

    def print_operators(self, print_format='paulis'):
        """
        Print out the paulis in the selected representation.

        Args:
            print_format (str): "paulis", "grouped_paulis", "matrix"

        Returns:
            str: a formated operator.

        Raises:
            ValueError: if `print_format` is not supported.
        """
        ret = ""
        if print_format == 'paulis':
            self._check_representation("paulis")
            for pauli in self._paulis:
                ret += "{}\t{}\n".format(pauli[1].to_label(), pauli[0])
            if ret == "":
                ret += "Pauli list is empty."
        elif print_format == 'grouped_paulis':
            self._check_representation("grouped_paulis")
            for i in range(len(self._grouped_paulis)):
                ret += 'Post Rotations of TPB set {} '.format(i)
                ret += ': {} '.format(self._grouped_paulis[i][0][1].to_label())
                ret += '\n'
                for j in range(1, len(self._grouped_paulis[i])):
                    ret += '{} '.format(self._grouped_paulis[i][j][1].to_label())
                    ret += '{}\n'.format(self._grouped_paulis[i][j][0])
                ret += '\n'
            if ret == "":
                ret += "Grouped pauli list is empty."
        elif print_format == 'matrix':
            self._check_representation("matrix")
            ret = str(self._matrix.toarray())
        else:
            raise ValueError('Mode should be one of "matrix", "paulis", "grouped_paulis"')
        return ret

    def _eval_with_statevector(self, operator_mode, input_circuit, backend, execute_config):
        """
        Evaluate an Operator with the `input_circuit`.
        This mode interacts with the quantum state rather than the sampled results from the measurement.
        - Psi is wave function
        - Psi is dense matrix

        Args:
            operator_mode (str): representation of operator, including paulis, grouped_paulis and matrix
            input_circuit (QuantumCircuit): the quantum circuit.
            backend (str): backend selection for quantum machine.
            execute_config (dict): execution setting to quautum backend, refer to qiskit.wrapper.execute for details.

        Returns:
            float: average of evaluations

        Raises:
            AlgorithmError: if it tries to use non-statevector simulator.
        """
        if "statevector" not in backend:
            raise AlgorithmError(
                "statevector can be only used in statevector simulator but {} is used".format(backend))

        avg = 0.0
        if operator_mode == "matrix":
            self._check_representation("matrix")
            if self._dia_matrix is None:
                self._to_dia_matrix(mode='matrix')

            job = q_execute(input_circuit, backend=backend, **execute_config)

            if self._summarize_circuits and logger.isEnabledFor(logging.DEBUG):
                logger.debug(summarize_circuits(input_circuit))

            result = job.result()
            quantum_state = np.asarray(result.get_statevector(input_circuit))

            if self._dia_matrix is not None:
                avg = np.sum(self._dia_matrix * np.absolute(quantum_state) ** 2)
            else:
                avg = np.vdot(quantum_state, self._matrix.dot(quantum_state))

        else:
            self._check_representation("paulis")
            n_qubits = self.num_qubits
            circuits = []
            base_circuit = QuantumCircuit() + input_circuit
            circuits.append(base_circuit)
            # Trial circuit w/o the final rotations
            # Execute trial circuit with final rotations for each Pauli in
            # hamiltonian and store from circuits[1] on

            for idx, pauli in enumerate(self._paulis):
                circuit = QuantumCircuit() + base_circuit
                q = circuit.get_qregs()['q']
                for qubit_idx in range(n_qubits):
                    if pauli[1].v[qubit_idx] == 0 and pauli[1].w[qubit_idx] == 1:
                        circuit.u3(np.pi, 0.0, np.pi, q[qubit_idx]) #x
                    elif pauli[1].v[qubit_idx] == 1 and pauli[1].w[qubit_idx] == 0:
                        circuit.u1(np.pi, q[qubit_idx]) #z
                    elif pauli[1].v[qubit_idx] == 1 and pauli[1].w[qubit_idx] == 1:
                        circuit.u3(np.pi, np.pi/2, np.pi/2, q[qubit_idx]) #y
                circuits.append(circuit)

            jobs = []
            chunks = int(np.ceil(len(circuits) / self.MAX_CIRCUITS_PER_JOB))
            for i in range(chunks):
                sub_circuits = circuits[i*self.MAX_CIRCUITS_PER_JOB:(i+1)*self.MAX_CIRCUITS_PER_JOB]
                jobs.append(q_execute(sub_circuits, backend=backend, **execute_config))

            if self._summarize_circuits and logger.isEnabledFor(logging.DEBUG):
                logger.debug(summarize_circuits(circuits))

            results = []
            for job in jobs:
                results.append(job.result())
            result = reduce(lambda x, y: x + y, results)

            quantum_state_0 = np.asarray(result.get_statevector(circuits[0]))

            for idx, pauli in enumerate(self._paulis):
                quantum_state_i = np.asarray(result.get_statevector(circuits[idx+1]))
                # inner product with final rotations of (i)-th Pauli
                avg += pauli[0] * (np.vdot(quantum_state_0, quantum_state_i))

        return avg

    def _eval_multiple_shots(self, operator_mode, input_circuit, backend, execute_config, qjob_config):
        """
        Evaluate an Operator with the `input_circuit`. This mode interacts with the quantum machine and uses
        the statistic results.

        Args:
            operator_mode (str): representation of operator, including paulis, grouped_paulis and matrix
            input_circuit (QuantumCircuit): the quantum circuit.
            backend (str): backend selection for quantum machine.
            execute_config (dict): execution setting to quautum backend, refer to qiskit.wrapper.execute for details.
            qjob_config (dict): the setting to retrieve results from quantum backend, including timeout and wait.

        Returns:
            float, float: mean and standard deviation of evaluation results
        """
        num_shots = execute_config.get("shots", 1)
        avg, std_dev, variance = 0.0, 0.0, 0.0
        n_qubits = self.num_qubits
        circuits = []

        base_circuit = QuantumCircuit() + input_circuit
        c = base_circuit.get_cregs().get('c', ClassicalRegister(n_qubits, name='c'))
        base_circuit.add(c)

        if operator_mode == "paulis":
            self._check_representation("paulis")

            for idx, pauli in enumerate(self._paulis):
                circuit = QuantumCircuit() + base_circuit
                q = circuit.get_qregs()['q']
                c = circuit.get_cregs()['c']

                for qubit_idx in range(n_qubits):
                    # Measure X
                    if pauli[1].v[qubit_idx] == 0 and pauli[1].w[qubit_idx] == 1:
                        circuit.u2(0.0, np.pi, q[qubit_idx]) #h
                    # Measure Y
                    elif pauli[1].v[qubit_idx] == 1 and pauli[1].w[qubit_idx] == 1:
                        circuit.u1(np.pi/2, q[qubit_idx]).inverse() #s
                        circuit.u2(0.0, np.pi, q[qubit_idx]) #h
                    circuit.measure(q[qubit_idx], c[qubit_idx])

                circuits.append(circuit)

            jobs = []
            chunks = int(np.ceil(len(circuits) / self.MAX_CIRCUITS_PER_JOB))
            for i in range(chunks):
                sub_circuits = circuits[i*self.MAX_CIRCUITS_PER_JOB:(i+1)*self.MAX_CIRCUITS_PER_JOB]
                jobs.append(q_execute(sub_circuits, backend=backend, **execute_config))

            if self._summarize_circuits and logger.isEnabledFor(logging.DEBUG):
                logger.debug(summarize_circuits(circuits))

            results = []
            for job in jobs:
                results.append(job.result(**qjob_config))
            result = reduce(lambda x, y: x + y, results)

            avg_paulis = []
            for idx, pauli in enumerate(self._paulis):
                measured_results = result.get_counts(circuits[idx])
                avg_paulis.append(Operator._measure_pauli_z(measured_results, pauli[1]))
                avg += pauli[0] * avg_paulis[idx]
                variance += (pauli[0] ** 2) * Operator._covariance(measured_results, pauli[1], pauli[1],
                                                                   avg_paulis[idx], avg_paulis[idx])

        elif operator_mode == 'grouped_paulis':
            self._check_representation("grouped_paulis")

            for idx, tpb_set in enumerate(self._grouped_paulis):
                circuit = QuantumCircuit() + base_circuit
                q = circuit.get_qregs()['q']
                c = circuit.get_cregs()['c']
                for qubit_idx in range(n_qubits):
                    # Measure X
                    if tpb_set[0][1].v[qubit_idx] == 0 and tpb_set[0][1].w[qubit_idx] == 1:
                        circuit.u2(0.0, np.pi, q[qubit_idx]) #h
                    # Measure Y
                    elif tpb_set[0][1].v[qubit_idx] == 1 and tpb_set[0][1].w[qubit_idx] == 1:
                        circuit.u1(np.pi/2, q[qubit_idx]).inverse() #s
                        circuit.u2(0.0, np.pi, q[qubit_idx]) #h
                    circuit.measure(q[qubit_idx], c[qubit_idx])
                circuits.append(circuit)

            # Execute all the stacked quantum circuits - one for each TPB set
            jobs = []
            chunks = int(np.ceil(len(circuits) / self.MAX_CIRCUITS_PER_JOB))
            for i in range(chunks):
                sub_circuits = circuits[i*self.MAX_CIRCUITS_PER_JOB:(i+1)*self.MAX_CIRCUITS_PER_JOB]
                jobs.append(q_execute(sub_circuits, backend=backend, **execute_config))

            if self._summarize_circuits and logger.isEnabledFor(logging.DEBUG):
                logger.debug(summarize_circuits(circuits))

            results = []
            for job in jobs:
                results.append(job.result(**qjob_config))
            result = reduce(lambda x, y: x + y, results)

            for tpb_idx, tpb_set in enumerate(self._grouped_paulis):
                avg_paulis = []
                measured_results = result.get_counts(circuits[tpb_idx])
                # Compute the averages of each pauli in tpb_set
                for pauli_idx, pauli in enumerate(tpb_set):
                    avg_paulis.append(Operator._measure_pauli_z(measured_results, pauli[1]))
                    avg += pauli[0] * avg_paulis[pauli_idx]

                # Compute the covariance matrix elements of tpb_set
                # and add up to the total standard deviation
                # tpb_set = grouped_paulis, tensor product basis set
                for pauli_1_idx, pauli_1 in enumerate(tpb_set):
                    for pauli_2_idx, pauli_2 in enumerate(tpb_set):
                        variance += pauli_1[0] * pauli_2[0] * \
                            Operator._covariance(measured_results, pauli_1[1], pauli_2[1],
                                                 avg_paulis[pauli_1_idx], avg_paulis[pauli_2_idx])

        std_dev = np.sqrt(variance / num_shots)

        return avg, std_dev

    def _eval_directly(self, quantum_state):
        self._check_representation("matrix")
        if self._dia_matrix is None:
            self._to_dia_matrix(mode='matrix')
        if self._dia_matrix is not None:
            avg = np.sum(self._dia_matrix * np.absolute(quantum_state) ** 2)
        else:
            avg = np.vdot(quantum_state, self._matrix.dot(quantum_state))
        return avg

    def eval(self, operator_mode, input_circuit, backend, execute_config={}, qjob_config={}):
        """
        Supporting three ways to evaluate the given circuits with the operator.
        1. If `input_circuit` is a numpy.ndarray, it will directly perform inner product with the operator.
        2. If `backend` is a statevector simulator, use quantum backend to get statevector
           and then evaluate with the operator.
        3. Other cases: it use with quanutm backend (simulator or real quantum machine),
           to obtain the mean and standard deviation of measured results.

        Args:
            operator_mode (str): representation of operator, including paulis, grouped_paulis and matrix
            input_circuit (QuantumCircuit or numpy.ndarray): the quantum circuit.
            backend (str): backend selection for quantum machine.
            execute_config (dict): execution setting to quautum backend, refer to qiskit.wrapper.execute for details.
            qjob_config (dict): the setting to retrieve results from quantum backend, including timeout and wait.

        Returns:
            float, float: mean and standard deviation of avg
        """

        # If the statevector is already a vector, skip the evaluation from quantum simulator.

        if isinstance(input_circuit, np.ndarray):
            avg = self._eval_directly(input_circuit)
            std_dev = 0.0
        else:
            if backend.startswith('local'):
                self.MAX_CIRCUITS_PER_JOB = sys.maxsize
            if "statevector" in backend:
                execute_config['shots'] = 1
                avg = self._eval_with_statevector(operator_mode, input_circuit, backend, execute_config)
                std_dev = 0.0
            else:
                avg, std_dev = self._eval_multiple_shots(operator_mode, input_circuit, backend, execute_config, qjob_config)
        return avg, std_dev

    def convert(self, input_format, output_format, force=False):
        """
        A wrapper for conversion among all representations.
        Note that, if the output target is already there, it will skip the conversion.
        The result is stored back into its property directly.

        Args:
            input_format (str): case-insensitive input format,
                                should be one of "paulis", "grouped_paulis", "matrix"
            output_format (str): case-insensitive output format,
                                 should be one of "paulis", "grouped_paulis", "matrix"
            force (bool): convert to targeted format regardless its present.

        Raises:
            ValueError: if the unsupported output_format is specified.
        """
        input_format = input_format.lower()
        output_format = output_format.lower()

        if input_format not in ["paulis", "grouped_paulis", "matrix"]:
            raise ValueError(
                "Input format {} is not supported".format(input_format))

        if output_format not in ["paulis", "grouped_paulis", "matrix"]:
            raise ValueError(
                "Output format {} is not supported".format(output_format))

        if output_format == "paulis" and (self._paulis is None or force):
            if input_format == "matrix":
                self._matrix_to_paulis()
            elif input_format == "grouped_paulis":
                self._grouped_paulis_to_paulis()

        elif output_format == "grouped_paulis" and (self._grouped_paulis is None or force):
            if self._grouped_paulis == []:
                return
            if input_format == "paulis":
                self._paulis_to_grouped_paulis()
            elif input_format == "matrix":
                self._matrix_to_grouped_paulis()

        elif output_format == "matrix" and (self._matrix is None or force):
            if input_format == "paulis":
                self._paulis_to_matrix()
            elif input_format == "grouped_paulis":
                self._grouped_paulis_to_matrix()

    def _grouped_paulis_to_paulis(self):
        """
        Convert grouped paulis to paulis, and save it in internal property directly.

        Note:
            Ideally, all paulis in grouped_paulis should be unique.
            No need to check whether it is existed.
        """
        if self._grouped_paulis == []:
            return
        paulis = []
        for group in self._grouped_paulis:
            for idx in range(1, len(group)):  # the first one is the header.
                paulis.append(group[idx])
        self._paulis = paulis

    def _matrix_to_paulis(self):
        """
        Convert matrix to paulis, and save it in internal property directly.

        Note:
            Conversion from Paulis to matrix: H = sum_i alpha_i * Pauli_i
            Conversion from matrix to Paulis: alpha_i = coeff * Trace(H.Pauli_i) (dot product of trace)
                where coeff = 2^(- # of qubits), # of qubit = log2(dim of matrix)
        """
        if self._matrix.nnz == 0:
            return

        num_qubits = self.num_qubits
        coeff = 2 ** (-num_qubits)

        paulis = []
        # generate all possible paulis basis
        for basis in itertools.product('IXYZ', repeat=num_qubits):
            pauli_i = label_to_pauli(''.join(basis))
            trace_value = np.sum(self._matrix.dot(pauli_i.to_spmatrix()).diagonal())
            alpha_i = trace_value * coeff
            if alpha_i != 0.0:
                paulis.append([alpha_i, pauli_i])
        self._paulis = paulis

    def _paulis_to_grouped_paulis(self):
        """
        Convert paulis to grouped_paulis, and save it in internal property directly.
        Groups a list of [coeff,Pauli] into tensor product basis (tpb) sets
        """
        if self._paulis == []:
            return
        if self._coloring is not None:
            self._grouped_paulis = PauliGraph(self._paulis, mode=self._coloring).grouped_paulis
        else:
            temp_paulis = copy.deepcopy(self._paulis)
            n = self.num_qubits
            grouped_paulis = []
            sorted_paulis = []

            def check_pauli_in_list(target, pauli_list):
                ret = False
                for pauli in pauli_list:
                    if target[1] == pauli[1]:
                        ret = True
                        break
                return ret
            for i in range(len(temp_paulis)):
                p_1 = temp_paulis[i]
                if not check_pauli_in_list(p_1, sorted_paulis):
                    paulis_temp = []
                    # pauli_list_temp.extend(p_1) # this is going to signal the total
                    # post-rotations of the set (set master)
                    paulis_temp.append(p_1)
                    paulis_temp.append(copy.deepcopy(p_1))
                    paulis_temp[0][0] = 0.0  # zero coeff for HEADER
                    for j in range(i+1, len(temp_paulis)):
                        p_2 = temp_paulis[j]
                        if not check_pauli_in_list(p_2, sorted_paulis) and p_1[1] != p_2[1]:
                            j = 0
                            for i in range(n):
                                # p_2 is identity, p_1 is identity, p_1 and p_2 has same basis
                                if not ((p_2[1].v[i] == 0 and p_2[1].w[i] == 0) or
                                        (p_1[1].v[i] == 0 and p_1[1].w[i] == 0) or
                                        (p_2[1].v[i] == p_1[1].v[i] and
                                         p_2[1].w[i] == p_1[1].w[i])):
                                    break
                                else:
                                    # update master, if p_2 is not identity
                                    if p_2[1].v[i] == 1 or p_2[1].w[i] == 1:
                                        paulis_temp[0][1].v[i] = p_2[1].v[i]
                                        paulis_temp[0][1].w[i] = p_2[1].w[i]
                                j += 1
                            if j == n:
                                paulis_temp.append(p_2)
                                sorted_paulis.append(p_2)
                    grouped_paulis.append(paulis_temp)
            self._grouped_paulis = grouped_paulis

    def _matrix_to_grouped_paulis(self):
        """
        Convert matrix to grouped_paulis, and save it in internal property directly.
        """
        if self._matrix.nnz == 0:
            return
        self._matrix_to_paulis()
        self._paulis_to_grouped_paulis()

    def _paulis_to_matrix(self):
        """
        Convert paulis to matrix, and save it in internal property directly.
        If all paulis are Z or I (identity), convert to dia_matrix.
        """
        if self._paulis == []:
            return
        self._to_dia_matrix(mode='paulis')
        if self._dia_matrix is None:

            p = self._paulis[0]
            hamiltonian = p[0] * p[1].to_spmatrix()
            for idx in range(1, len(self._paulis)):
                p = self._paulis[idx]
                hamiltonian += p[0] * p[1].to_spmatrix()
            self._matrix = hamiltonian

    def _grouped_paulis_to_matrix(self):
        """
        Convert grouped_paulis to matrix, and save it in internal property directly.
        If all paulis are Z or I (identity), convert to dia_matrix.
        """
        if self._grouped_paulis == []:
            return
        self._to_dia_matrix(mode='grouped_paulis')
        if self._dia_matrix is None:
            p = self._grouped_paulis[0][1]
            hamiltonian = p[0] * p[1].to_spmatrix()
            for idx in range(2, len(self._grouped_paulis[0])):
                p = self._grouped_paulis[0][idx]
                hamiltonian += p[0] * p[1].to_spmatrix()
            for group_idx in range(1, len(self._grouped_paulis)):
                group = self._grouped_paulis[group_idx]
                for idx in range(1, len(group)):
                    p = group[idx]
                    hamiltonian += p[0] * p[1].to_spmatrix()
            self._matrix = hamiltonian

    @staticmethod
    def _measure_pauli_z(data, pauli):
        """
        Appropriate post-rotations on the state are assumed.

        Args:
            data (dict): a dictionary of the form data = {'00000': 10} ({str: int})
            pauli (Pauli): a Pauli object

        Returns:
            float: Expected value of paulis given data
        """
        observable = 0
        tot = sum(data.values())
        for key in data:
            value = 1
            for j in range(pauli.numberofqubits):
                if ((pauli.v[j] == 1 or pauli.w[j] == 1) and
                        key[pauli.numberofqubits - j - 1] == '1'):
                    value = -value

            observable = observable + value * data[key] / tot
        return observable

    @staticmethod
    def _covariance(data, pauli_1, pauli_2, avg_1, avg_2):
        """
        Compute the covariance matrix element between two
        Paulis, given the measurement outcome.
        Appropriate post-rotations on the state are assumed.

        Args:
            data (dict): a dictionary of the form data = {'00000': 10} ({str:int})
            pauli_1 (Pauli): a Pauli class member
            pauli_2 (Pauli): a Pauli class member
            avg_1 (float): expectation value of pauli_1 on `data`
            avg_2 (float): expectation value of pauli_2 on `data`

        Returns:
            float: the element of the covariance matrix between two Paulis
        """
        cov = 0.0
        shots = sum(data.values())
        n_qub = pauli_1.numberofqubits

        if shots == 1:
            return cov

        for key in data:
            sign_1 = 1
            sign_2 = 1
            for j in range(n_qub):
                if ((pauli_1.v[j] == 1 or pauli_1.w[j] == 1) and
                        key[n_qub - j - 1] == '1'):
                    sign_1 = -sign_1
            for j in range(n_qub):
                if ((pauli_2.v[j] == 1 or pauli_2.w[j] == 1) and
                        key[n_qub - j - 1] == '1'):
                    sign_2 = -sign_2
            cov += (sign_1 - avg_1) * (sign_2 - avg_2) * data[key] / (shots - 1)
        return cov

    def two_qubit_reduced_operator(self, m, threshold=10**-13):
        """
        Eliminates the central and last qubit in a list of Pauli that has
        diagonal operators (Z,I) at those positions

        Chemistry specific method:
        It can be used to taper two qubits in parity and binary-tree mapped
        fermionic Hamiltonians when the spin orbitals are ordered in two spin
        sectors, (block spin order) according to the number of particles in the system.

        Args:
            m (int): number of fermionic particles
            threshold (float): threshold for Pauli simplification

        Returns:
            Operator: a new operator whose qubit number is reduced by 2.

        """
        if self._paulis is None or self._paulis == []:
            return self

        operator_out = Operator(paulis=[])
        par_1 = 1 if m % 2 == 0 else -1
        par_2 = 1 if m % 4 == 0 else -1

        n = len(self._paulis[0][1].v)
        last_idx = n - 1
        mid_idx = n // 2 - 1
        for pauli_term in self._paulis:  # loop over Pauli terms
            coeff_out = pauli_term[0]
            # Z operator encountered at qubit n/2-1
            if pauli_term[1].v[mid_idx] == 1 and pauli_term[1].w[mid_idx] == 0:
                coeff_out = par_2 * coeff_out
            # Z operator encountered at qubit n-1
            if pauli_term[1].v[last_idx] == 1 and pauli_term[1].w[last_idx] == 0:
                coeff_out = par_1 * coeff_out
            v_temp = []
            w_temp = []
            for j in range(n-1):
                if j != mid_idx:
                    # for j in range(n):
                    #     if j != n // 2 - 1 and j != n - 1:
                    v_temp.append(pauli_term[1].v[j])
                    w_temp.append(pauli_term[1].w[j])
            pauli_term_out = [coeff_out, Pauli(np.array(v_temp), np.array(w_temp))]
            if np.absolute(coeff_out) > threshold:
                operator_out += Operator(paulis=[pauli_term_out])
        operator_out.chop(threshold=threshold)

        return operator_out

    def reorder_paulis(self, grouping=None):
        """
        Reorder the pauli terms according to the specified grouping.

        Args:
            grouping (str): The name of the grouping, currently supports 'random' and 'default'.
                'random' corresponds to the order of Operator.paulis;
                'default' corresponds to the grouping as specified by Operator.grouped_paulis

        Returns:
            list: The list of pauli terms as ordered per the specified grouping
        """
        self._check_representation("paulis")
        if grouping == 'random':
            return self._paulis
        elif grouping == 'default':
            if self.grouped_paulis is None:
                self._paulis_to_grouped_paulis()
            return [pauli for group in self._grouped_paulis for pauli in group[1:]]
        else:
            raise ValueError('Unrecognized grouping {}.'.format(grouping))

    def construct_evolution_circuit(self, slice_pauli_list, evo_time, num_time_slices, state_registers,
                                    ancillary_registers=None, ctl_idx=0, unitary_power=None, use_basis_gates=True):
        """
        Construct the evolution circuit according to the supplied specification.

        Args:
            slice_pauli_list (list): The list of pauli terms corresponding to a single time slice to be evolved
            evo_time (int): The evolution time
            num_time_slices (int): The number of time slices for the expansion
            state_registers (QuantumRegister): The Qiskit QuantumRegister corresponding to the qubits of the system
            ancillary_registers (QuantumRegister): The optional Qiskit QuantumRegister corresponding to the control
                qubits for the state_registers of the system
            ctl_idx (int): The index of the qubit of the control ancillary_registers to use
            unitary_power (int): The power to which the unitary operator is to be raised
            use_basis_gates (bool): boolean flag for indicating only using basis gates when building circuit.

        Returns:
            QuantumCircuit: The Qiskit QuantumCircuit corresponding to specified evolution.
        """
        if state_registers is None:
            raise ValueError('Quantum state registers are required.')

        n_qubits = self.num_qubits
        qc = QuantumCircuit(state_registers)
        if ancillary_registers is not None:
            qc.add(ancillary_registers)

        # for each pauli [IXYZ]+, record the list of qubit pairs needing CX's
        cnot_qubit_pairs = [None] * len(slice_pauli_list)
        # for each pauli [IXYZ]+, record the highest index of the nontrivial pauli gate (X,Y, or Z)
        top_XYZ_pauli_indices = [-1] * len(slice_pauli_list)

        for pauli_idx, pauli in enumerate(reversed(slice_pauli_list)):
            # changes bases if necessary
            nontrivial_pauli_indices = []
            for qubit_idx in range(n_qubits):
                # pauli I
                if pauli[1].v[qubit_idx] == 0 and pauli[1].w[qubit_idx] == 0:
                    continue

                if cnot_qubit_pairs[pauli_idx] is None:
                    nontrivial_pauli_indices.append(qubit_idx)

                if pauli[1].w[qubit_idx] == 1:
                    # pauli X
                    if pauli[1].v[qubit_idx] == 0:
                        if use_basis_gates:
                            qc.u2(0.0, pi, state_registers[qubit_idx])
                        else:
                            qc.h(state_registers[qubit_idx])
                    # pauli Y
                    elif pauli[1].v[qubit_idx] == 1:
                        if use_basis_gates:
                            qc.u3(pi / 2, -pi / 2, pi / 2, state_registers[qubit_idx])
                        else:
                            qc.rx(pi / 2, state_registers[qubit_idx])
                # pauli Z
                elif pauli[1].v[qubit_idx] == 1 and pauli[1].w[qubit_idx] == 0:
                    pass
                else:
                    raise ValueError('Unrecognized pauli: {}'.format(pauli[1]))

            if len(nontrivial_pauli_indices) > 0:
                top_XYZ_pauli_indices[pauli_idx] = nontrivial_pauli_indices[-1]

            # insert lhs cnot gates
            if cnot_qubit_pairs[pauli_idx] is None:
                cnot_qubit_pairs[pauli_idx] = list(zip(
                    sorted(nontrivial_pauli_indices)[:-1],
                    sorted(nontrivial_pauli_indices)[1:]
                ))

            for pair in cnot_qubit_pairs[pauli_idx]:
                qc.cx(state_registers[pair[0]], state_registers[pair[1]])

            # insert Rz gate
            if top_XYZ_pauli_indices[pauli_idx] >= 0:
                if ancillary_registers is None:
                    lam = (2.0 * pauli[0] * evo_time / num_time_slices).real
                    if use_basis_gates:
                        qc.u1(lam, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                    else:
                        qc.rz(lam, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                else:
                    unitary_power = (2 ** ctl_idx) if unitary_power is None else unitary_power
                    lam = (2.0 * pauli[0] * evo_time / num_time_slices * unitary_power).real

                    if use_basis_gates:
                        qc.u1(lam / 2, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                        qc.cx(ancillary_registers[ctl_idx], state_registers[top_XYZ_pauli_indices[pauli_idx]])
                        qc.u1(-lam / 2, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                        qc.cx(ancillary_registers[ctl_idx], state_registers[top_XYZ_pauli_indices[pauli_idx]])
                    else:
                        qc.crz(lam, ancillary_registers[ctl_idx], state_registers[top_XYZ_pauli_indices[pauli_idx]])

            # insert rhs cnot gates
            for pair in reversed(cnot_qubit_pairs[pauli_idx]):
                qc.cx(state_registers[pair[0]], state_registers[pair[1]])

            # revert bases if necessary
            for qubit_idx in range(n_qubits):
                if pauli[1].w[qubit_idx] == 1:
                    # pauli X
                    if pauli[1].v[qubit_idx] == 0:
                        if use_basis_gates:
                            qc.u2(0.0, pi, state_registers[qubit_idx])
                        else:
                            qc.h(state_registers[qubit_idx])
                    # pauli Y
                    elif pauli[1].v[qubit_idx] == 1:
                        if use_basis_gates:
                            qc.u3(-pi / 2, -pi / 2, pi / 2, state_registers[qubit_idx])
                        else:
                            qc.rx(-pi / 2, state_registers[qubit_idx])
        # repeat the slice
        qc.data *= num_time_slices
        return qc

    @staticmethod
    def _suzuki_expansion_slice_matrix(pauli_list, lam, expansion_order):
        """
        Compute the matrix for a single slice of the suzuki expansion following the paper
        https://arxiv.org/pdf/quant-ph/0508139.pdf

        Args:
            pauli_list (list): The operator's complete list of pauli terms for the suzuki expansion
            lam (float): The parameter lambda as defined in said paper
            expansion_order (int): The order for the suzuki expansion

        Returns:
            numpy array: The matrix representation corresponding to the specified suzuki expansion
        """
        if expansion_order == 1:
            left = reduce(
                lambda x, y: x @ y,
                [scila.expm(lam / 2 * c * p.to_spmatrix().tocsc()) for c, p in pauli_list]
            )
            right = reduce(
                lambda x, y: x @ y,
                [scila.expm(lam / 2 * c * p.to_spmatrix().tocsc()) for c, p in reversed(pauli_list)]
            )
            return left @ right
        else:
            pk = (4 - 4 ** (1 / (2 * expansion_order - 1))) ** -1
            side_base = Operator._suzuki_expansion_slice_matrix(
                pauli_list,
                lam * pk,
                expansion_order - 1
            )
            side = side_base @ side_base
            middle = Operator._suzuki_expansion_slice_matrix(
                pauli_list,
                lam * (1 - 4 * pk),
                expansion_order - 1
            )
            return side @ middle @ side

    @staticmethod
    def _suzuki_expansion_slice_pauli_list(pauli_list, lam_coef, expansion_order):
        """
        Similar to _suzuki_expansion_slice_matrix, with the difference that this method
        computes the list of pauli terms for a single slice of the suzuki expansion,
        which can then be fed to construct_evolution_circuit to build the QuantumCircuit.
        """
        if expansion_order == 1:
            half = [[lam_coef / 2 * c, p] for c, p in pauli_list]
            return half + list(reversed(half))
        else:
            pk = (4 - 4 ** (1 / (2 * expansion_order - 1))) ** -1
            side_base = Operator._suzuki_expansion_slice_pauli_list(
                pauli_list,
                lam_coef * pk,
                expansion_order - 1
            )
            side = side_base * 2
            middle = Operator._suzuki_expansion_slice_pauli_list(
                pauli_list,
                lam_coef * (1 - 4 * pk),
                expansion_order - 1
            )
            return side + middle + side

    def evolve(self, state_in, evo_time, evo_mode, num_time_slices, quantum_registers=None,
               paulis_grouping='random', expansion_mode='trotter', expansion_order=1):
        """
        Carry out the dynamics evolution for the operator under supplied specifications.

        Args:
            state_in: The initial state for the evolution
            evo_time (int): The evolution time
            evo_mode (str): The mode under which the evolution is carried out.
                Currently only support 'matrix' or 'circuit'
            num_time_slices (int): The number of time slices for the expansion
            quantum_registers (QuantumRegister): The QuantumRegister to build the QuantumCircuit off of
            paulis_grouping (str): The grouping to dictate the ordering of the pauli terms.
                See reorder_paulis method for more details.
            expansion_mode (str): The mode under which the expansion is to be done.
                Currently support 'trotter', which follows the expansion as discussed in
                http://science.sciencemag.org/content/273/5278/1073,
                and 'suzuki', which corresponds to the discussion in
                https://arxiv.org/pdf/quant-ph/0508139.pdf
            expansion_order (int): The order for suzuki expansion

        Returns:
            Depending on the evo_mode specified, either return the matrix vector multiplication result
            or the constructed QuantumCircuit.

        """
        if num_time_slices < 0 or not isinstance(num_time_slices, int):
            raise ValueError('Number of time slices should be a non-negative integer.')
        if not (expansion_mode == 'trotter' or expansion_mode == 'suzuki'):
            raise NotImplementedError('Expansion mode {} not supported.'.format(expansion_mode))

        pauli_list = self.reorder_paulis(grouping=paulis_grouping)

        if evo_mode == 'matrix':
            self._check_representation("matrix")

            if num_time_slices == 0:
                return scila.expm(-1.j * evo_time * self._matrix.tocsc()) @ state_in
            else:
                if len(pauli_list) == 1:
                    approx_matrix_slice = scila.expm(
                        -1.j * evo_time / num_time_slices * pauli_list[0][0] * pauli_list[0][1].to_spmatrix().tocsc()
                    )
                else:
                    if expansion_mode == 'trotter':
                        approx_matrix_slice = reduce(
                            lambda x, y: x @ y,
                            [
                                scila.expm(-1.j * evo_time / num_time_slices * c * p.to_spmatrix().tocsc())
                                for c, p in pauli_list
                            ]
                        )
                    # suzuki expansion
                    elif expansion_mode == 'suzuki':
                        approx_matrix_slice = Operator._suzuki_expansion_slice_matrix(
                            pauli_list,
                            -1.j * evo_time / num_time_slices,
                            expansion_order
                        )
                    else:
                        raise ValueError('Unrecognized expansion mode {}.'.format(expansion_mode))
                return reduce(lambda x, y: x @ y, [approx_matrix_slice] * num_time_slices) @ state_in

        elif evo_mode == 'circuit':
            if num_time_slices == 0:
                raise ValueError('Number of time slices should be a positive integer for {} mode.'.format(evo_mode))
            else:
                if quantum_registers is None:
                    raise ValueError('Quantum registers are needed for circuit construction.')
                if len(pauli_list) == 1:
                    slice_pauli_list = pauli_list
                else:
                    if expansion_mode == 'trotter':
                        slice_pauli_list = pauli_list
                    # suzuki expansion
                    else:
                        slice_pauli_list = Operator._suzuki_expansion_slice_pauli_list(
                            pauli_list,
                            1,
                            expansion_order
                        )
                return self.construct_evolution_circuit(
                    slice_pauli_list, evo_time, num_time_slices, quantum_registers
                )
        else:
            raise ValueError('Evolution mode should be either "matrix" or "circuit".')

    def is_empty(self):
        """
        Check Operator is empty or not.

        Returns:
            bool: is empty?
        """
        if self._matrix is None and self._dia_matrix is None \
            and (self._paulis == [] or self._paulis is None) \
            and (self._grouped_paulis == [] or self._grouped_paulis is None):

            return True
        else:
            return False

    def _check_representation(self, targeted_represnetation):
        """
        Check the targeted representation is existed or not, if not, find available represnetations
        and then convert to the targeted one.

        Args:
            targeted_representation (str): should be one of paulis, grouped_paulis and matrix

        Raises:
            ValueError: if the `targeted_representation` is not recognized.
        """
        if targeted_represnetation == 'paulis':
            if self._paulis is None:
                if self._matrix is not None:
                    self._matrix_to_paulis()
                elif self._grouped_paulis is not None:
                    self._grouped_paulis_to_paulis()
                else:
                    raise AlgorithmError(
                        "at least having one of the three operator representations.")

        elif targeted_represnetation == 'grouped_paulis':
            if self._grouped_paulis is None:
                if self._paulis is not None:
                    self._paulis_to_grouped_paulis()
                elif self._matrix is not None:
                    self._matrix_to_grouped_paulis()
                else:
                    raise AlgorithmError(
                        "at least having one of the three operator representations.")

        elif targeted_represnetation == 'matrix':
            if self._matrix is None:
                if self._paulis is not None:
                    self._paulis_to_matrix()
                elif self._grouped_paulis is not None:
                    self._grouped_paulis_to_matrix()
                else:
                    raise AlgorithmError(
                        "at least having one of the three operator representations.")
        else:
            raise ValueError(
                '"targeted_represnetation" should be one of "paulis", "grouped_paulis" and "matrix".'
            )

    @staticmethod
    def row_echelon_F2(matrix_in):
        """
        Computes the row Echelon form of a binary matrix on the binary
        finite field

        Args:
            matrix_in (np.ndarray): binary matrix

        Returns:
            np.ndarray : matrix_in in Echelon row form
        """

        size = matrix_in.shape

        for i in range(size[0]):
            pivot_index = 0
            for j in range(size[1]):
                if matrix_in[i, j] == 1:
                    pivot_index = j
                    break
            for k in range(size[0]):
                if k != i and matrix_in[k, pivot_index] == 1:
                    matrix_in[k, :] = np.mod(matrix_in[k, :] + matrix_in[i, :], 2)

        matrix_out_temp = copy.deepcopy(matrix_in)
        indices = []
        matrix_out = np.zeros(size)

        for i in range(size[0] - 1):
            if np.array_equal(matrix_out_temp[i, :], np.zeros(size[1])):
                indices.append(i)
        for row in np.sort(indices)[::-1]:
            matrix_out_temp = np.delete(matrix_out_temp, (row), axis=0)

        matrix_out[0:size[0] - len(indices), :] = matrix_out_temp
        matrix_out = matrix_out.astype(int)

        return matrix_out

    @staticmethod
    def kernel_F2(matrix_in):
        """
        Computes the kernel of a binary matrix on the binary finite field

        Args:
            matrix_in (np.ndarray): binary matrix

        Returns:
            [np.ndarray]: the list of kernel vectors
        """

        size = matrix_in.shape
        kernel = []
        matrix_in_id = np.vstack((matrix_in, np.identity(size[1])))
        matrix_in_id_ech = (Operator.row_echelon_F2(matrix_in_id.transpose())).transpose()

        for col in range(size[1]):
            if (np.array_equal(matrix_in_id_ech[0:size[0], col], np.zeros(size[0])) and not
            np.array_equal(matrix_in_id_ech[size[0]:, col], np.zeros(size[1]))) :
                kernel.append(matrix_in_id_ech[size[0]:, col])

        return kernel

    def find_Z2_symmetries(self):
        """
        Finds Z2 Pauli-type symmetries of an Operator

        Returns:
            [Pauli]: the list of Pauli objects representing the Z2 symmetries
            [Pauli]: the list of single - qubit Pauli objects to construct the Cliffors operators
            [Operators]: the list of Clifford unitaries to block diagonalize Operator
            [int]: the list of support of the single-qubit Pauli objects used to build the clifford operators
        """

        Pauli_symmetries = []
        sq_paulis = []
        cliffords = []
        sq_list = []

        stacked_paulis = []
        for pauli in self._paulis:
            stacked_paulis.append(np.concatenate((pauli[1].w, pauli[1].v), axis=0))

        stacked_matrix = np.array(np.stack(stacked_paulis))
        symmetries = Operator.kernel_F2(stacked_matrix)
        stacked_symmetries = np.stack(symmetries)
        symm_shape = stacked_symmetries.shape

        for row in range(symm_shape[0]):

            Pauli_symmetries.append(Pauli(stacked_symmetries[row, : symm_shape[1] // 2],
                                          stacked_symmetries[row, symm_shape[1] // 2 : ]))

            stacked_symm_del = np.delete(stacked_symmetries, (row), axis=0)
            for col in range(symm_shape[1] // 2):

                # case symmetries other than one at (row) have Z or I on col qubit
                Z_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not (stacked_symm_del[symm_idx, col] == 0
                            and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] in (0, 1)):
                        Z_or_I = False
                if Z_or_I == True:
                    if ((stacked_symmetries[row, col] == 1 and
                         stacked_symmetries[row, col + symm_shape[1] // 2] == 0) or
                         (stacked_symmetries[row, col] == 1 and
                         stacked_symmetries[row, col + symm_shape[1] // 2] == 1)):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2),
                                               np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].v[col] = 0
                        sq_paulis[row].w[col] = 1
                        sq_list.append(col)
                        break

                # case symmetries other than one at (row) have X or I on col qubit
                X_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not (stacked_symm_del[symm_idx, col] in (0, 1) and
                            stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0):
                        X_or_I = False
                if X_or_I == True:
                    if ( (stacked_symmetries[row, col] == 0 and
                          stacked_symmetries[row, col + symm_shape[1] // 2] == 1) or
                         (stacked_symmetries[row, col] == 1 and
                          stacked_symmetries[row, col + symm_shape[1] // 2] == 1) ):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2), np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].v[col] = 1
                        sq_paulis[row].w[col] = 0
                        sq_list.append(col)
                        break

                # case symmetries other than one at (row)  have Y or I on col qubit
                Y_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not ( (stacked_symm_del[symm_idx, col] == 1 and
                              stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 1)
                        or   (stacked_symm_del[symm_idx, col] == 0 and
                              stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0) ):
                        Y_or_I = False
                if Y_or_I == True:
                    if ( (stacked_symmetries[row, col] == 0 and
                          stacked_symmetries[row, col + symm_shape[1] // 2] == 1) or
                         (stacked_symmetries[row, col] == 1 and
                          stacked_symmetries[row, col + symm_shape[1] // 2] == 0) ):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2), np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].v[col] = 1
                        sq_paulis[row].w[col] = 1
                        sq_list.append(col)
                        break

        for symm_idx, Pauli_symm in enumerate(Pauli_symmetries):
            cliffords.append(Operator([[1/np.sqrt(2), Pauli_symm], [1/np.sqrt(2), sq_paulis[symm_idx]]]))

        return Pauli_symmetries, sq_paulis, cliffords, sq_list

    @staticmethod
    def qubit_tapering(operator, cliffords, sq_list, tapering_values):
        """
        Builds an Operator which has a number of qubits tapered off,
        based on a block-diagonal Operator built using a list of cliffords.
        The block-diagonal subspace is an input parameter, set through the list
        tapering_values, which takes values +/- 1.

        Args:
            operator (Operator): the target operator to be tapered
            cliffords ([Operator]): list of unitary Clifford transformation
            sq_list ([int]): position of the single-qubit operators that anticommute
            with the cliffords
            tapering_values ([int]): array of +/- 1 used to select the subspace. Length
            has to be equal to the length of cliffords and sq_list

        Returns:
            Operator : the tapered operator
        """

        if len(cliffords) != len(sq_list):
            raise ValueError('number of Clifford unitaries has to be the same as lenght of single\
            qubit list and tapering values')
        if len(sq_list) != len(tapering_values):
            raise ValueError('number of Clifford unitaries has to be the same as lenght of single\
            qubit list and tapering values')

        for clifford in cliffords:
            operator = clifford * operator * clifford

        operator_out = Operator(paulis=[])
        n = len(operator.paulis[0][1].v)
        for pauli_term in operator.paulis:
            coeff_out = pauli_term[0]
            for qubit_idx, qubit in enumerate(sq_list):
                if not (pauli_term[1].v[qubit] == 0 and pauli_term[1].w[qubit] == 0):
                    coeff_out = tapering_values[qubit_idx] * coeff_out
            v_temp = []
            w_temp = []
            for j in range(n):
                if j not in sq_list:
                    v_temp.append(pauli_term[1].v[j])
                    w_temp.append(pauli_term[1].w[j])
            pauli_term_out = [coeff_out, Pauli(np.array(v_temp), np.array(w_temp))]
            operator_out += Operator(paulis=[pauli_term_out])

        return operator_out

    def zeros_coeff_elimination(self):
        """
        Elinminate paulis or grouped paulis whose coefficients are zeros.

        The difference from `_simplify_paulis` method is that, this method will not remove duplicated
        paulis.
        """
        if self._paulis is not None:
            new_paulis = [pauli for pauli in self._paulis if pauli[0] != 0]
            self._paulis = new_paulis
            self._paulis_table = {pauli[1].to_label(): i for i, pauli in enumerate(self._paulis)}

        elif self._grouped_paulis is not None:
            self._grouped_paulis_to_paulis()
            self.zeros_coeff_elimination()
            self._paulis_to_grouped_paulis()
            self._paulis = None
