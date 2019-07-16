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

import copy
import itertools
from functools import reduce
import logging
import json
from operator import iadd as op_iadd, isub as op_isub
import sys

import numpy as np
from scipy import sparse as scisparse
from scipy import linalg as scila
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.qasm import pi
from qiskit.assembler.run_config import RunConfig
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.aqua import AquaError, aqua_globals
from qiskit.aqua.utils import PauliGraph, compile_and_run_circuits, find_regs_by_name
from qiskit.aqua.utils.backend_utils import is_statevector_backend

logger = logging.getLogger(__name__)


class Operator(object):

    """
    Operators relevant for quantum applications

    Note:
        For grouped paulis representation, all operations will always convert it to paulis and then convert it back.
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

    def _extend_or_combine(self, rhs, mode, operation=op_iadd):
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

        if mode == 'inplace':
            lhs = self
        elif mode == 'non-inplace':
            lhs = copy.deepcopy(self)

        if lhs._paulis is not None and rhs._paulis is not None:
            for pauli in rhs._paulis:
                pauli_label = pauli[1].to_label()
                idx = lhs._paulis_table.get(pauli_label, None)
                if idx is not None:
                    lhs._paulis[idx][0] = operation(lhs._paulis[idx][0], pauli[0])
                else:
                    lhs._paulis_table[pauli_label] = len(lhs._paulis)
                    pauli[0] = operation(0.0, pauli[0])
                    lhs._paulis.append(pauli)
        elif lhs._grouped_paulis is not None and rhs._grouped_paulis is not None:
            lhs._grouped_paulis_to_paulis()
            rhs._grouped_paulis_to_paulis()
            lhs = operation(lhs, rhs)
            lhs._paulis_to_grouped_paulis()
        elif lhs._matrix is not None and rhs._matrix is not None:
            lhs._matrix = operation(lhs._matrix, rhs._matrix)
        else:
            raise TypeError("the representations of two Operators should be the same. ({}, {})".format(
                lhs.representations, rhs.representations))

        return lhs

    def __add__(self, rhs):
        """Overload + operation"""
        return self._extend_or_combine(rhs, 'non-inplace', op_iadd)

    def __iadd__(self, rhs):
        """Overload += operation"""
        return self._extend_or_combine(rhs, 'inplace', op_iadd)

    def __sub__(self, rhs):
        """Overload - operation"""
        return self._extend_or_combine(rhs, 'non-inplace', op_isub)

    def __isub__(self, rhs):
        """Overload -= operation"""
        return self._extend_or_combine(rhs, 'inplace', op_isub)

    def __neg__(self):
        """Overload unary - """
        ret = copy.deepcopy(self)
        ret.scaling_coeff(-1.0)
        return ret

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
                if not found_pauli and rhs_coeff != 0.0:  # since we might have 0 weights of paulis.
                    return False
                if coeff != rhs_coeff:
                    return False
            return True

        if self._grouped_paulis is not None and rhs._grouped_paulis is not None:
            self._grouped_paulis_to_paulis()
            rhs._grouped_paulis_to_paulis()
            return self.__eq__(rhs)

    def __ne__(self, rhs):
        """ != """
        return not self.__eq__(rhs)

    def __str__(self):
        """Overload str()"""
        curr_repr = ""
        length = ""
        group = None
        if self._paulis is not None:
            curr_repr = 'paulis'
            length = len(self._paulis)
        elif self._grouped_paulis is not None:
            curr_repr = 'grouped_paulis'
            group = len(self._grouped_paulis)
            length = sum([len(gp) - 1 for gp in self._grouped_paulis])
        elif self._matrix is not None:
            curr_repr = 'matrix'
            length = "{}x{}".format(2 ** self.num_qubits, 2 ** self.num_qubits)

        ret = "Representation: {}, qubits: {}, size: {}{}".format(
            curr_repr, self.num_qubits, length, "" if group is None else " (number of groups: {})".format(group))

        return ret

    def copy(self):
        """Get a copy of self."""
        return copy.deepcopy(self)

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
            if self._dia_matrix is not None:
                self._to_dia_matrix('paulis')

        elif self._grouped_paulis is not None:
            grouped_paulis = []
            for group_idx in range(1, len(self._grouped_paulis)):
                for pauli_idx in range(len(self._grouped_paulis[group_idx])):
                    self._grouped_paulis[group_idx][pauli_idx][0] = chop_real_imag(
                        self._grouped_paulis[group_idx][pauli_idx][0], threshold)
                paulis = [x for x in self._grouped_paulis[group_idx] if x[0] != 0.0]
                grouped_paulis.append(paulis)
            self._grouped_paulis = grouped_paulis
            if self._dia_matrix is not None:
                self._to_dia_matrix('grouped_paulis')

        elif self._matrix is not None:
            rows, cols = self._matrix.nonzero()
            for row, col in zip(rows, cols):
                self._matrix[row, col] = chop_real_imag(self._matrix[row, col], threshold)
            self._matrix.eliminate_zeros()
            if self._dia_matrix is not None:
                self._to_dia_matrix('matrix')

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
                    basis, sign = Pauli.sgn_prod(existed_pauli[1], pauli[1])
                    coeff = existed_pauli[0] * pauli[0] * sign
                    if abs(coeff) > 1e-15:
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

    @property
    def aer_paulis(self):
        if getattr(self, '_aer_paulis', None) is None:
            self.to_paulis()
            aer_paulis = []
            for coeff, p in self._paulis:
                new_coeff = [coeff.real, coeff.imag]
                new_p = p.to_label()
                aer_paulis.append([new_coeff, new_p])
            self._aer_paulis = aer_paulis
        return self._aer_paulis

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
                    if not (np.all(np.logical_not(pauli.x))):
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
        Return the available representations in the Operator.

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
                return len(self._paulis[0][1])
            else:
                return 0
        elif self._grouped_paulis is not None and self._grouped_paulis != []:
            return len(self._grouped_paulis[0][0][1])
        else:
            return int(np.log2(self._matrix.shape[0]))

    @staticmethod
    def load_from_file(file_name, before_04=False):
        """
        Load paulis in a file to construct an Operator.

        Args:
            file_name (str): path to the file, which contains a list of Paulis and coefficients.
            before_04 (bool): support the format < 0.4.

        Returns:
            Operator class: the loaded operator.
        """
        with open(file_name, 'r') as file:
            return Operator.load_from_dict(json.load(file), before_04=before_04)

    def save_to_file(self, file_name):
        """
        Save operator to a file in pauli representation.

        Args:
            file_name (str): path to the file

        """
        with open(file_name, 'w') as f:
            json.dump(self.save_to_dict(), f)

    @staticmethod
    def load_from_dict(dictionary, before_04=False):
        """
        Load paulis in a dict to construct an Operator, \
        the dict must be represented as follows: label and coeff (real and imag). \
        E.g.: \
           {'paulis': \
               [ \
                   {'label': 'IIII', \
                    'coeff': {'real': -0.33562957575267038, 'imag': 0.0}}, \
                   {'label': 'ZIII', \
                    'coeff': {'real': 0.28220597164664896, 'imag': 0.0}}, \
                    ... \
                ] \
            } \

        Args:
            dictionary (dict): dictionary, which contains a list of Paulis and coefficients.
            before_04 (bool): support the format < 0.4.

        Returns:
            Operator: the loaded operator.
        """
        if 'paulis' not in dictionary:
            raise AquaError('Dictionary missing "paulis" key')

        paulis = []
        for op in dictionary['paulis']:
            if 'label' not in op:
                raise AquaError('Dictionary missing "label" key')

            pauli_label = op['label']
            if 'coeff' not in op:
                raise AquaError('Dictionary missing "coeff" key')

            pauli_coeff = op['coeff']
            if 'real' not in pauli_coeff:
                raise AquaError('Dictionary missing "real" key')

            coeff = pauli_coeff['real']
            if 'imag' in pauli_coeff:
                coeff = complex(pauli_coeff['real'], pauli_coeff['imag'])

            pauli_label = pauli_label[::-1] if before_04 else pauli_label
            paulis.append([coeff, Pauli.from_label(pauli_label)])

        return Operator(paulis=paulis)

    def save_to_dict(self):
        """
        Save operator to a dict in pauli representation.

        Returns:
            dict: a dictionary contains an operator with pauli representation.
        """
        self._check_representation("paulis")
        ret_dict = {"paulis": []}
        for pauli in self._paulis:
            op = {"label": pauli[1].to_label()}
            if isinstance(pauli[0], complex):
                op["coeff"] = {"real": np.real(pauli[0]),
                               "imag": np.imag(pauli[0])
                               }
            else:
                op["coeff"] = {"real": pauli[0]}

            ret_dict["paulis"].append(op)

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
                ret = ''.join([ret, "{}\t{}\n".format(pauli[1].to_label(), pauli[0])])
            if ret == "":
                ret = ''.join([ret, "Pauli list is empty."])
        elif print_format == 'grouped_paulis':
            self._check_representation("grouped_paulis")
            for i in range(len(self._grouped_paulis)):
                ret = ''.join([ret, 'Post Rotations of TPB set {} '.format(i)])
                ret = ''.join([ret, ': {} '.format(self._grouped_paulis[i][0][1].to_label())])
                ret = ''.join([ret, '\n'])
                for j in range(1, len(self._grouped_paulis[i])):
                    ret = ''.join([ret, '{} '.format(self._grouped_paulis[i][j][1].to_label())])
                    ret = ''.join([ret, '{}\n'.format(self._grouped_paulis[i][j][0])])
                ret = ''.join([ret, '\n'])
            if ret == "":
                ret = ''.join([ret, "Grouped pauli list is empty."])
        elif print_format == 'matrix':
            self._check_representation("matrix")
            ret = str(self._matrix.toarray())
        else:
            raise ValueError('Mode should be one of "matrix", "paulis", "grouped_paulis"')
        return ret

    def construct_evaluation_circuit(self, operator_mode, input_circuit, backend, use_simulator_operator_mode=False):
        """
        Construct the circuits for evaluation.

        Args:
            operator_mode (str): representation of operator, including paulis, grouped_paulis and matrix
            input_circuit (QuantumCircuit): the quantum circuit.
            backend (BaseBackend): backend selection for quantum machine.
            use_simulator_operator_mode (bool): if aer_provider is used, we can do faster
                           evaluation for pauli mode on statevector simualtion

        Returns:
            [QuantumCircuit]: the circuits for evaluation.
        """
        if is_statevector_backend(backend):
            if operator_mode == 'matrix':
                circuits = [input_circuit]
            else:
                self._check_representation("paulis")
                if use_simulator_operator_mode:
                    circuits = [input_circuit]
                else:
                    n_qubits = self.num_qubits
                    q = find_regs_by_name(input_circuit, 'q')
                    circuits = [input_circuit]
                    for idx, pauli in enumerate(self._paulis):
                        circuit = QuantumCircuit() + input_circuit
                        if np.all(np.logical_not(pauli[1].z)) and np.all(np.logical_not(pauli[1].x)):  # all I
                            continue
                        for qubit_idx in range(n_qubits):
                            if not pauli[1].z[qubit_idx] and pauli[1].x[qubit_idx]:
                                circuit.u3(np.pi, 0.0, np.pi, q[qubit_idx])  # x
                            elif pauli[1].z[qubit_idx] and not pauli[1].x[qubit_idx]:
                                circuit.u1(np.pi, q[qubit_idx])  # z
                            elif pauli[1].z[qubit_idx] and pauli[1].x[qubit_idx]:
                                circuit.u3(np.pi, np.pi/2, np.pi/2, q[qubit_idx])  # y
                        circuits.append(circuit)
        else:
            if operator_mode == 'matrix':
                raise AquaError("matrix mode can not be used with non-statevector simulator.")

            n_qubits = self.num_qubits
            circuits = []

            base_circuit = QuantumCircuit() + input_circuit
            c = find_regs_by_name(base_circuit, 'c', qreg=False)
            if c is None:
                c = ClassicalRegister(n_qubits, name='c')
            base_circuit.add_register(c)

            if operator_mode == "paulis":
                self._check_representation("paulis")

                for idx, pauli in enumerate(self._paulis):
                    circuit = QuantumCircuit() + base_circuit
                    q = find_regs_by_name(circuit, 'q')
                    c = find_regs_by_name(circuit, 'c', qreg=False)
                    for qubit_idx in range(n_qubits):
                        if pauli[1].x[qubit_idx]:
                            if pauli[1].z[qubit_idx]:
                                # Measure Y
                                circuit.u1(-np.pi/2, q[qubit_idx])  # sdg
                                circuit.u2(0.0, np.pi, q[qubit_idx])  # h
                            else:
                                # Measure X
                                circuit.u2(0.0, np.pi, q[qubit_idx])  # h
                    circuit.barrier(q)
                    circuit.measure(q, c)
                    circuits.append(circuit)
            else:
                self._check_representation("grouped_paulis")

                for idx, tpb_set in enumerate(self._grouped_paulis):
                    circuit = QuantumCircuit() + base_circuit
                    q = find_regs_by_name(circuit, 'q')
                    c = find_regs_by_name(circuit, 'c', qreg=False)
                    for qubit_idx in range(n_qubits):
                        if tpb_set[0][1].x[qubit_idx]:
                            if tpb_set[0][1].z[qubit_idx]:
                                # Measure Y
                                circuit.u1(-np.pi/2, q[qubit_idx])  # sdg
                                circuit.u2(0.0, np.pi, q[qubit_idx])  # h
                            else:
                                # Measure X
                                circuit.u2(0.0, np.pi, q[qubit_idx])  # h
                    circuit.barrier(q)
                    circuit.measure(q, c)
                    circuits.append(circuit)
        return circuits

    def evaluate_with_result(self, operator_mode, circuits, backend, result, use_simulator_operator_mode=False):
        """
        Use the executed result with operator to get the evaluated value.

        Args:
            operator_mode (str): representation of operator, including paulis, grouped_paulis and matrix
            circuits (list of qiskit.QuantumCircuit): the quantum circuits.
            backend (str): backend selection for quantum machine.
            result (qiskit.Result): the result from the backend.
            use_simulator_operator_mode (bool): if aer_provider is used, we can do faster
                           evaluation for pauli mode on statevector simualtion
        Returns:
            float: the mean value
            float: the standard deviation
        """
        avg, std_dev, variance = 0.0, 0.0, 0.0
        if is_statevector_backend(backend):
            if operator_mode == "matrix":
                self._check_representation("matrix")
                if self._dia_matrix is None:
                    self._to_dia_matrix(mode='matrix')
                quantum_state = np.asarray(result.get_statevector(circuits[0]))
                if self._dia_matrix is not None:
                    avg = np.sum(self._dia_matrix * np.absolute(quantum_state) ** 2)
                else:
                    avg = np.vdot(quantum_state, self._matrix.dot(quantum_state))
            else:
                self._check_representation("paulis")
                if use_simulator_operator_mode:
                    temp = result.data(circuits[0])['snapshots']['expectation_value']['test'][0]['value']
                    avg = temp[0] + 1j * temp[1]
                else:
                    quantum_state = np.asarray(result.get_statevector(circuits[0]))
                    circuit_idx = 1
                    for idx, pauli in enumerate(self._paulis):
                        if np.all(np.logical_not(pauli[1].z)) and np.all(np.logical_not(pauli[1].x)):
                            avg += pauli[0]
                        else:
                            quantum_state_i = np.asarray(result.get_statevector(circuits[circuit_idx]))
                            avg += pauli[0] * (np.vdot(quantum_state, quantum_state_i))
                            circuit_idx += 1
        else:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug("Computing the expectation from measurement results:")
                TextProgressBar(sys.stderr)
            num_shots = sum(list(result.get_counts(circuits[0]).values()))
            if operator_mode == "paulis":
                self._check_representation("paulis")
                results = parallel_map(Operator._routine_paulis_with_shots,
                                       [(pauli, result.get_counts(circuits[idx]))
                                        for idx, pauli in enumerate(self._paulis)],
                                       num_processes=aqua_globals.num_processes)
                for result in results:
                    avg += result[0]
                    variance += result[1]
            else:
                self._check_representation("grouped_paulis")
                results = parallel_map(Operator._routine_grouped_paulis_with_shots,
                                       [(tpb_set, result.get_counts(circuits[tpb_idx]))
                                        for tpb_idx, tpb_set in enumerate(self._grouped_paulis)],
                                       num_processes=aqua_globals.num_processes)
                for result in results:
                    avg += result[0]
                    variance += result[1]

            std_dev = np.sqrt(variance / num_shots)

        return avg, std_dev

    @staticmethod
    def _routine_grouped_paulis_with_shots(args):
        tpb_set, measured_results = args
        avg_paulis = []
        avg = 0.0
        variance = 0.0
        for pauli_idx, pauli in enumerate(tpb_set):
            if pauli_idx == 0:
                continue
            observable = Operator._measure_pauli_z(measured_results, pauli[1])
            avg_paulis.append(observable)
            avg += pauli[0] * observable

        # Compute the covariance matrix elements of tpb_set
        # and add up to the total standard deviation
        # tpb_set = grouped_paulis, tensor product basis set
        for pauli_1_idx, pauli_1 in enumerate(tpb_set):
            for pauli_2_idx, pauli_2 in enumerate(tpb_set):
                if pauli_1_idx == 0 or pauli_2_idx == 0:
                    continue
                variance += pauli_1[0] * pauli_2[0] * \
                    Operator._covariance(measured_results, pauli_1[1], pauli_2[1],
                                         avg_paulis[pauli_1_idx-1], avg_paulis[pauli_2_idx-1])
        return avg, variance

    @staticmethod
    def _routine_paulis_with_shots(args):
        pauli, measured_results = args
        curr_result = Operator._measure_pauli_z(measured_results, pauli[1])
        avg = pauli[0] * curr_result
        variance = (pauli[0] ** 2) * Operator._covariance(measured_results, pauli[1], pauli[1],
                                                          curr_result, curr_result)
        return avg, variance

    def _eval_directly(self, quantum_state):
        self._check_representation("matrix")
        if self._dia_matrix is None:
            self._to_dia_matrix(mode='matrix')
        if self._dia_matrix is not None:
            avg = np.sum(self._dia_matrix * np.absolute(quantum_state) ** 2)
        else:
            avg = np.vdot(quantum_state, self._matrix.dot(quantum_state))
        return avg

    def eval(self, operator_mode, input_circuit, backend, backend_config=None, compile_config=None,
             run_config=None, qjob_config=None, noise_config=None):
        """
        Supporting three ways to evaluate the given circuits with the operator.
        1. If `input_circuit` is a numpy.ndarray, it will directly perform inner product with the operator.
        2. If `backend` is a statevector simulator, use quantum backend to get statevector \
           and then evaluate with the operator.
        3. Other cases: it use with quanutm backend (simulator or real quantum machine), \
           to obtain the mean and standard deviation of measured results.

        Args:
            operator_mode (str): representation of operator, including paulis, grouped_paulis and matrix
            input_circuit (QuantumCircuit or numpy.ndarray): the quantum circuit.
            backend (BaseBackend): backend selection for quantum machine.
            backend_config (dict): configuration for backend
            compile_config (dict): configuration for compilation
            run_config (RunConfig): configuration for running a circuit
            qjob_config (dict): the setting to retrieve results from quantum backend, including timeout and wait.
            noise_config (dict) the setting of noise model for the qasm simulator in the Aer provider.

        Returns:
            float, float: mean and standard deviation of avg
        """
        backend_config = backend_config or {}
        compile_config = compile_config or {}
        if run_config is not None:
            if isinstance(run_config, dict):
                run_config = RunConfig(**run_config)
        else:
            run_config = RunConfig()
        qjob_config = qjob_config or {}
        noise_config = noise_config or {}

        if isinstance(input_circuit, np.ndarray):
            avg = self._eval_directly(input_circuit)
            std_dev = 0.0
        else:
            if is_statevector_backend(backend):
                run_config.shots = 1

            circuits = self.construct_evaluation_circuit(operator_mode, input_circuit, backend)
            result = compile_and_run_circuits(circuits, backend=backend, backend_config=backend_config,
                                              compile_config=compile_config, run_config=run_config,
                                              qjob_config=qjob_config, noise_config=noise_config,
                                              show_circuit_summary=self._summarize_circuits)
            avg, std_dev = self.evaluate_with_result(operator_mode, circuits, backend, result)

        return avg, std_dev

    def to_paulis(self):
        self._check_representation('paulis')

    def to_grouped_paulis(self):
        self._check_representation('grouped_paulis')

    def to_matrix(self):
        self._check_representation('matrix')

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
        self._matrix = None
        self._grouped_paulis = None

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
            pauli_i = Pauli.from_label(''.join(basis))
            trace_value = np.sum(self._matrix.dot(pauli_i.to_spmatrix()).diagonal())
            alpha_i = trace_value * coeff
            if alpha_i != 0.0:
                paulis.append([alpha_i, pauli_i])
        self._paulis = paulis
        self._matrix = None
        self._grouped_paulis = None

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
                                if not ((not p_2[1].z[i] and not p_2[1].x[i]) or
                                        (not p_1[1].z[i] and not p_1[1].x[i]) or
                                        (p_2[1].z[i] == p_1[1].z[i] and
                                         p_2[1].x[i] == p_1[1].x[i])):
                                    break
                                else:
                                    # update master, if p_2 is not identity
                                    if p_2[1].z[i] or p_2[1].x[i]:
                                        paulis_temp[0][1].update_z(p_2[1].z[i], i)
                                        paulis_temp[0][1].update_x(p_2[1].x[i], i)
                                j += 1
                            if j == n:
                                paulis_temp.append(p_2)
                                sorted_paulis.append(p_2)
                    grouped_paulis.append(paulis_temp)
            self._grouped_paulis = grouped_paulis
        self._matrix = None
        self._paulis = None

    def _matrix_to_grouped_paulis(self):
        """
        Convert matrix to grouped_paulis, and save it in internal property directly.
        """
        if self._matrix.nnz == 0:
            return
        self._matrix_to_paulis()
        self._paulis_to_grouped_paulis()
        self._matrix = None
        self._paulis = None

    def _paulis_to_matrix(self):
        """
        Convert paulis to matrix, and save it in internal property directly.
        If all paulis are Z or I (identity), convert to dia_matrix.
        """
        if self._paulis == []:
            return
        p = self._paulis[0]
        hamiltonian = p[0] * p[1].to_spmatrix()
        for idx in range(1, len(self._paulis)):
            p = self._paulis[idx]
            hamiltonian += p[0] * p[1].to_spmatrix()
        self._matrix = hamiltonian
        self._to_dia_matrix(mode='matrix')
        self._paulis = None
        self._grouped_paulis = None

    def _grouped_paulis_to_matrix(self):
        """
        Convert grouped_paulis to matrix, and save it in internal property directly.
        If all paulis are Z or I (identity), convert to dia_matrix.
        """
        if self._grouped_paulis == []:
            return
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
        self._to_dia_matrix(mode='matrix')
        self._paulis = None
        self._grouped_paulis = None

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
        observable = 0.0
        num_shots = sum(data.values())
        p_z_or_x = np.logical_or(pauli.z, pauli.x)
        for key, value in data.items():
            bitstr = np.asarray(list(key))[::-1].astype(np.bool)
            sign = -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p_z_or_x)) else 1.0
            observable += sign * value
        observable /= num_shots
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
        num_shots = sum(data.values())

        if num_shots == 1:
            return cov

        p1_z_or_x = np.logical_or(pauli_1.z, pauli_1.x)
        p2_z_or_x = np.logical_or(pauli_2.z, pauli_2.x)
        for key, value in data.items():
            bitstr = np.asarray(list(key))[::-1].astype(np.bool)
            sign_1 = -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p1_z_or_x)) else 1.0
            sign_2 = -1.0 if np.logical_xor.reduce(np.logical_and(bitstr, p2_z_or_x)) else 1.0
            cov += (sign_1 - avg_1) * (sign_2 - avg_2) * value
        cov /= (num_shots - 1)
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

        n = self.num_qubits
        last_idx = n - 1
        mid_idx = n // 2 - 1
        for pauli_term in self._paulis:  # loop over Pauli terms
            coeff_out = pauli_term[0]
            # Z operator encountered at qubit n/2-1
            if pauli_term[1].z[mid_idx] and not pauli_term[1].x[mid_idx]:
                coeff_out = par_2 * coeff_out
            # Z operator encountered at qubit n-1
            if pauli_term[1].z[last_idx] and not pauli_term[1].x[last_idx]:
                coeff_out = par_1 * coeff_out

            # TODO: can change to delete
            z_temp = []
            x_temp = []
            for j in range(n - 1):
                if j != mid_idx:
                    z_temp.append(pauli_term[1].z[j])
                    x_temp.append(pauli_term[1].x[j])
            pauli_term_out = [coeff_out, Pauli(np.asarray(z_temp), np.asarray(x_temp))]
            if np.absolute(coeff_out) > threshold:
                operator_out += Operator(paulis=[pauli_term_out])
        operator_out.chop(threshold=threshold)

        return operator_out

    def get_flat_pauli_list(self):
        """
        Get the flat list of paulis

        Returns:
            list: The list of pauli terms
        """
        if self._paulis is not None:
            return [] + self._paulis
        else:
            if self._grouped_paulis is not None:
                return [pauli for group in self._grouped_paulis for pauli in group[1:]]
            elif self._matrix is not None:
                self._check_representation('paulis')
                return [] + self._paulis

    @staticmethod
    def construct_evolution_circuit(slice_pauli_list, evo_time, num_time_slices, state_registers,
                                    ancillary_registers=None, ctl_idx=0, unitary_power=None, use_basis_gates=True,
                                    shallow_slicing=False):
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
            shallow_slicing (bool): boolean flag for indicating using shallow qc.data reference repetition for slicing

        Returns:
            QuantumCircuit: The Qiskit QuantumCircuit corresponding to specified evolution.
        """
        if state_registers is None:
            raise ValueError('Quantum state registers are required.')

        qc_slice = QuantumCircuit(state_registers)
        if ancillary_registers is not None:
            qc_slice.add_register(ancillary_registers)

        # for each pauli [IXYZ]+, record the list of qubit pairs needing CX's
        cnot_qubit_pairs = [None] * len(slice_pauli_list)
        # for each pauli [IXYZ]+, record the highest index of the nontrivial pauli gate (X,Y, or Z)
        top_XYZ_pauli_indices = [-1] * len(slice_pauli_list)

        for pauli_idx, pauli in enumerate(reversed(slice_pauli_list)):
            n_qubits = pauli[1].numberofqubits
            # changes bases if necessary
            nontrivial_pauli_indices = []
            for qubit_idx in range(n_qubits):
                # pauli I
                if not pauli[1].z[qubit_idx] and not pauli[1].x[qubit_idx]:
                    continue

                if cnot_qubit_pairs[pauli_idx] is None:
                    nontrivial_pauli_indices.append(qubit_idx)

                if pauli[1].x[qubit_idx]:
                    # pauli X
                    if not pauli[1].z[qubit_idx]:
                        if use_basis_gates:
                            qc_slice.u2(0.0, pi, state_registers[qubit_idx])
                        else:
                            qc_slice.h(state_registers[qubit_idx])
                    # pauli Y
                    elif pauli[1].z[qubit_idx]:
                        if use_basis_gates:
                            qc_slice.u3(pi / 2, -pi / 2, pi / 2, state_registers[qubit_idx])
                        else:
                            qc_slice.rx(pi / 2, state_registers[qubit_idx])
                # pauli Z
                elif pauli[1].z[qubit_idx] and not pauli[1].x[qubit_idx]:
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
                qc_slice.cx(state_registers[pair[0]], state_registers[pair[1]])

            # insert Rz gate
            if top_XYZ_pauli_indices[pauli_idx] >= 0:
                if ancillary_registers is None:
                    lam = (2.0 * pauli[0] * evo_time / num_time_slices).real
                    if use_basis_gates:
                        qc_slice.u1(lam, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                    else:
                        qc_slice.rz(lam, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                else:
                    unitary_power = (2 ** ctl_idx) if unitary_power is None else unitary_power
                    lam = (2.0 * pauli[0] * evo_time / num_time_slices * unitary_power).real

                    if use_basis_gates:
                        qc_slice.u1(lam / 2, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                        qc_slice.cx(ancillary_registers[ctl_idx], state_registers[top_XYZ_pauli_indices[pauli_idx]])
                        qc_slice.u1(-lam / 2, state_registers[top_XYZ_pauli_indices[pauli_idx]])
                        qc_slice.cx(ancillary_registers[ctl_idx], state_registers[top_XYZ_pauli_indices[pauli_idx]])
                    else:
                        qc_slice.crz(lam, ancillary_registers[ctl_idx],
                                     state_registers[top_XYZ_pauli_indices[pauli_idx]])

            # insert rhs cnot gates
            for pair in reversed(cnot_qubit_pairs[pauli_idx]):
                qc_slice.cx(state_registers[pair[0]], state_registers[pair[1]])

            # revert bases if necessary
            for qubit_idx in range(n_qubits):
                if pauli[1].x[qubit_idx]:
                    # pauli X
                    if not pauli[1].z[qubit_idx]:
                        if use_basis_gates:
                            qc_slice.u2(0.0, pi, state_registers[qubit_idx])
                        else:
                            qc_slice.h(state_registers[qubit_idx])
                    # pauli Y
                    elif pauli[1].z[qubit_idx]:
                        if use_basis_gates:
                            qc_slice.u3(-pi / 2, -pi / 2, pi / 2, state_registers[qubit_idx])
                        else:
                            qc_slice.rx(-pi / 2, state_registers[qubit_idx])

        # repeat the slice
        if shallow_slicing:
            logger.info('Under shallow slicing mode, the qc.data reference is repeated shallowly. '
                        'Thus, changing gates of one slice of the output circuit might affect other slices.')
            qc_slice.data *= num_time_slices
            qc = qc_slice
        else:
            qc = QuantumCircuit()
            for _ in range(num_time_slices):
                qc += qc_slice
        return qc

    @staticmethod
    def _suzuki_expansion_slice_matrix(pauli_list, lam, expansion_order):
        """
        Compute the matrix for a single slice of the suzuki expansion following the paper
        https://arxiv.org/pdf/quant-ph/0508139.pdf

        Args:
            pauli_list (list): The operator's complete list of pauli terms for the suzuki expansion
            lam (complex): The parameter lambda as defined in said paper
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

    def evolve(
            self,
            state_in=None,
            evo_time=0,
            evo_mode=None,
            num_time_slices=0,
            quantum_registers=None,
            expansion_mode='trotter',
            expansion_order=1
    ):
        """
        Carry out the eoh evolution for the operator under supplied specifications.

        Args:
            state_in: The initial state for the evolution
            evo_time (int): The evolution time
            evo_mode (str): The mode under which the evolution is carried out.
                Currently only support 'matrix' or 'circuit'
            num_time_slices (int): The number of time slices for the expansion
            quantum_registers (QuantumRegister): The QuantumRegister to build the QuantumCircuit off of
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

        pauli_list = self.get_flat_pauli_list()

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

    def _check_representation(self, targeted_representation):
        """
        Check the targeted representation is existed or not, if not, find available representations
        and then convert to the targeted one.

        Args:
            targeted_representation (str): should be one of paulis, grouped_paulis and matrix

        Raises:
            ValueError: if the `targeted_representation` is not recognized.
        """
        if targeted_representation == 'paulis':
            if self._paulis is None:
                if self._matrix is not None:
                    self._matrix_to_paulis()
                elif self._grouped_paulis is not None:
                    self._grouped_paulis_to_paulis()
                else:
                    raise AquaError(
                        "at least having one of the three operator representations.")

        elif targeted_representation == 'grouped_paulis':
            if self._grouped_paulis is None:
                if self._paulis is not None:
                    self._paulis_to_grouped_paulis()
                elif self._matrix is not None:
                    self._matrix_to_grouped_paulis()
                else:
                    raise AquaError(
                        "at least having one of the three operator representations.")

        elif targeted_representation == 'matrix':
            if self._matrix is None:
                if self._paulis is not None:
                    self._paulis_to_matrix()
                elif self._grouped_paulis is not None:
                    self._grouped_paulis_to_matrix()
                else:
                    raise AquaError(
                        "at least having one of the three operator representations.")
        else:
            raise ValueError(
                '"targeted_representation" should be one of "paulis", "grouped_paulis" and "matrix".'
            )

    @staticmethod
    def row_echelon_F2(matrix_in):
        """
        Computes the row Echelon form of a binary matrix on the binary
        finite field

        Args:
            matrix_in (numpy.ndarray): binary matrix

        Returns:
            numpy.ndarray : matrix_in in Echelon row form
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
            matrix_in (numpy.ndarray): binary matrix

        Returns:
            [numpy.ndarray]: the list of kernel vectors
        """

        size = matrix_in.shape
        kernel = []
        matrix_in_id = np.vstack((matrix_in, np.identity(size[1])))
        matrix_in_id_ech = (Operator.row_echelon_F2(matrix_in_id.transpose())).transpose()

        for col in range(size[1]):
            if (np.array_equal(matrix_in_id_ech[0:size[0], col], np.zeros(size[0])) and not
                    np.array_equal(matrix_in_id_ech[size[0]:, col], np.zeros(size[1]))):
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

        if self.is_empty():
            logger.info("Operator is empty.")
            return [], [], [], []

        self._check_representation("paulis")

        for pauli in self._paulis:
            stacked_paulis.append(np.concatenate((pauli[1].x, pauli[1].z), axis=0).astype(np.int))

        stacked_matrix = np.array(np.stack(stacked_paulis))
        symmetries = Operator.kernel_F2(stacked_matrix)

        if len(symmetries) == 0:
            logger.info("No symmetry is found.")
            return [], [], [], []

        stacked_symmetries = np.stack(symmetries)
        symm_shape = stacked_symmetries.shape

        for row in range(symm_shape[0]):

            Pauli_symmetries.append(Pauli(stacked_symmetries[row, : symm_shape[1] // 2],
                                          stacked_symmetries[row, symm_shape[1] // 2:]))

            stacked_symm_del = np.delete(stacked_symmetries, (row), axis=0)
            for col in range(symm_shape[1] // 2):
                # case symmetries other than one at (row) have Z or I on col qubit
                Z_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not (stacked_symm_del[symm_idx, col] == 0
                            and stacked_symm_del[symm_idx, col + symm_shape[1] // 2] in (0, 1)):
                        Z_or_I = False
                if Z_or_I:
                    if ((stacked_symmetries[row, col] == 1 and
                         stacked_symmetries[row, col + symm_shape[1] // 2] == 0) or
                        (stacked_symmetries[row, col] == 1 and
                         stacked_symmetries[row, col + symm_shape[1] // 2] == 1)):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2),
                                               np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].z[col] = False
                        sq_paulis[row].x[col] = True
                        sq_list.append(col)
                        break

                # case symmetries other than one at (row) have X or I on col qubit
                X_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not (stacked_symm_del[symm_idx, col] in (0, 1) and
                            stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0):
                        X_or_I = False
                if X_or_I:
                    if ((stacked_symmetries[row, col] == 0 and
                         stacked_symmetries[row, col + symm_shape[1] // 2] == 1) or
                        (stacked_symmetries[row, col] == 1 and
                         stacked_symmetries[row, col + symm_shape[1] // 2] == 1)):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2), np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].z[col] = True
                        sq_paulis[row].x[col] = False
                        sq_list.append(col)
                        break

                # case symmetries other than one at (row)  have Y or I on col qubit
                Y_or_I = True
                for symm_idx in range(symm_shape[0] - 1):
                    if not ((stacked_symm_del[symm_idx, col] == 1 and
                             stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 1)
                            or (stacked_symm_del[symm_idx, col] == 0 and
                                stacked_symm_del[symm_idx, col + symm_shape[1] // 2] == 0)):
                        Y_or_I = False
                if Y_or_I:
                    if ((stacked_symmetries[row, col] == 0 and
                         stacked_symmetries[row, col + symm_shape[1] // 2] == 1) or
                        (stacked_symmetries[row, col] == 1 and
                         stacked_symmetries[row, col + symm_shape[1] // 2] == 0)):
                        sq_paulis.append(Pauli(np.zeros(symm_shape[1] // 2), np.zeros(symm_shape[1] // 2)))
                        sq_paulis[row].z[col] = True
                        sq_paulis[row].x[col] = True
                        sq_list.append(col)
                        break

        for symm_idx, Pauli_symm in enumerate(Pauli_symmetries):
            cliffords.append(Operator([[1 / np.sqrt(2), Pauli_symm], [1 / np.sqrt(2), sq_paulis[symm_idx]]]))

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
            Operator : the tapered operator, or empty operator if the `operator` is empty.
        """

        if len(cliffords) == 0 or len(sq_list) == 0 or len(tapering_values) == 0:
            logger.warning("Cliffords, single qubit list and tapering values cannot be empty.\n"
                           "Return the original operator instead.")
            return operator

        if len(cliffords) != len(sq_list):
            logger.warning("Number of Clifford unitaries has to be the same as length of single"
                           "qubit list and tapering values.\n"
                           "Return the original operator instead.")
            return operator
        if len(sq_list) != len(tapering_values):
            logger.warning("Number of Clifford unitaries has to be the same as length of single"
                           "qubit list and tapering values.\n"
                           "Return the original operator instead.")
            return operator

        if operator.is_empty():
            logger.warning("The operator is empty, return the empty operator directly.")
            return operator

        operator.to_paulis()

        for clifford in cliffords:
            operator = clifford * operator * clifford

        operator_out = Operator(paulis=[])
        for pauli_term in operator.paulis:
            coeff_out = pauli_term[0]
            for idx, qubit_idx in enumerate(sq_list):
                if not (not pauli_term[1].z[qubit_idx] and not pauli_term[1].x[qubit_idx]):
                    coeff_out = tapering_values[idx] * coeff_out
            z_temp = np.delete(pauli_term[1].z.copy(), np.asarray(sq_list))
            x_temp = np.delete(pauli_term[1].x.copy(), np.asarray(sq_list))
            pauli_term_out = [coeff_out, Pauli(z_temp, x_temp)]
            operator_out += Operator(paulis=[pauli_term_out])

        operator_out.zeros_coeff_elimination()
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

    def scaling_coeff(self, scaling_factor):
        """
        Constant scale the coefficient in an operator.

        Note that: the behavior of scaling in paulis (grouped_paulis) might be different from matrix

        Args:
            scaling_factor (float): the sacling factor
        """
        if self._paulis is not None:
            for idx in range(len(self._paulis)):
                self._paulis[idx] = [self._paulis[idx][0] * scaling_factor, self._paulis[idx][1]]
        elif self._grouped_paulis is not None:
            self._grouped_paulis_to_paulis()
            self._scale_paulis(scaling_factor)
            self._paulis_to_grouped_paulis()
        elif self._matrix is not None:
            self._matrix *= scaling_factor
            if self._dia_matrix is not None:
                self._dia_matrix *= scaling_factor
