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
import concurrent.futures
import itertools
from functools import reduce
import logging
import json
from operator import iadd as op_iadd, isub as op_isub
import psutil

import numpy as np
from scipy import sparse as scisparse
from scipy import linalg as scila
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.quantum_info import Pauli
from qiskit.qasm import pi
from qiskit.qobj import RunConfig

from qiskit.aqua.operators import Operator
from qiskit.aqua import AquaError
from qiskit.aqua.utils import PauliGraph, compile_and_run_circuits, find_regs_by_name
from qiskit.aqua.utils.backend_utils import is_statevector_backend

logger = logging.getLogger(__name__)


class MatrixOperator(Operator):

    """
    Operator with underlying matrix representation

    """

    def __init__(self, matrix=None):
        """
        Args:
            matrix (numpy.ndarray or scipy.sparse.csr_matrix) : a 2-D sparse matrix represents operator (using CSR format internally)
        """
        if matrix is not None:
            matrix = matrix if scisparse.issparse(matrix) else scisparse.csr_matrix(matrix)
            matrix = matrix if scisparse.isspmatrix_csr(matrix) else matrix.to_csr(copy=True)

        self._matrix = matrix
        self._to_dia_matrix(mode="matrix")

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

        if isinstance(rhs, MatrixOperator):
            lhs._matrix = operation(lhs._matrix, rhs._matrix)
        else:
            raise TypeError("the representations of two Operators should be the same. ({}, {})".format(
                type(self), type(rhs)))

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
        if isinstance(rhs, MatrixOperator):
            return np.all(self._matrix == rhs._matrix)
        else:
            raise TypeError("the representations of two Operators should be the same. ({}, {})".format(
                type(self), type(rhs)))


    def __ne__(self, rhs):
        """ != """
        return not self.__eq__(rhs)

    #TODO beef up
    def __str__(self):
        """Overload str()"""
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

        rows, cols = self._matrix.nonzero()
        for row, col in zip(rows, cols):
            self._matrix[row, col] = chop_real_imag(self._matrix[row, col], threshold)
        self._matrix.eliminate_zeros()
        if self._dia_matrix is not None:
            self._to_dia_matrix('matrix')

    def __mul__(self, rhs):
        """
        Overload * operation. Only support two Operators have the same representation mode.

        Returns:
            Operator: the multipled Operator.

        Raises:
            TypeError, if two Operators do not have the same representations.
        """
        if isinstance(rhs, MatrixOperator):
            ret_matrix = self._matrix.dot(rhs._matrix)
            return Operator(matrix=ret_matrix)
        else:
            raise TypeError("the representations of two Operators should be the same. ({}, {})".format(
                type(self), type(rhs)))

    #TODO change to use PauliOperator
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
        """

        if self._matrix is not None:
            dia_matrix = self._matrix.diagonal()
            if not scisparse.csr_matrix(dia_matrix).nnz == self._matrix.nnz:
                dia_matrix = None
            self._dia_matrix = dia_matrix
        else:
            self._dia_matrix = None

    @property
    def matrix(self):
        """Getter of matrix; if matrix is diagonal, diagonal matrix is returned instead."""
        return self._dia_matrix if self._dia_matrix is not None else self._matrix

    def enable_summarize_circuits(self):
        self._summarize_circuits = True

    def disable_summarize_circuits(self):
        self._summarize_circuits = False

    @property
    def num_qubits(self):
        """
        number of qubits required for the operator.

        Returns:
            int: number of qubits

        """
        return int(np.log2(self._matrix.shape[0]))

    #TODO figure out if we need this
    @staticmethod
    def load_from_file(file_name):
        """
        Load paulis in a file to construct an Operator.

        Args:
            file_name (str): path to the file, which contains a list of Paulis and coefficients.
            before_04 (bool): support the format < 0.4.

        Returns:
            Operator class: the loaded operator.
        """
        with open(file_name, 'r') as file:
            return something

    # TODO figure out if we need this
    def save_to_file(self, file_name):
        """
        Save operator to a file in pauli representation.

        Args:
            file_name (str): path to the file

        """
        # with open(file_name, 'w') as f:
        #     json.dump(self._matrix, f)

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

    def construct_evaluation_circuit(self, input_circuit, backend):
        """
        Construct the circuits for evaluation.

        Args:
            input_circuit (QuantumCircuit): the quantum circuit.
            backend (BaseBackend): backend selection for quantum machine.
        Returns:
            [QuantumCircuit]: the circuits for evaluation.
        """
        if is_statevector_backend(backend):
            circuits = [input_circuit]
        else:
            raise AquaError("matrix mode can not be used with non-statevector simulator.")
        return circuits

    def evaluate_with_result(self, circuits, backend, result, use_simulator_operator_mode=False):
        """
        Use the executed result with operator to get the evaluated value.

        Args:
            circuits (list of qiskit.QuantumCircuit): the quantum circuits.
            backend (str): backend selection for quantum machine.
            result (qiskit.Result): the result from the backend.
        Returns:
            float: the mean value
            float: the standard deviation
        """
        avg, std_dev, variance = 0.0, 0.0, 0.0
        if is_statevector_backend(backend):
            self._check_representation("matrix")
            if self._dia_matrix is None:
                self._to_dia_matrix(mode='matrix')
            quantum_state = np.asarray(result.get_statevector(circuits[0]))
            if self._dia_matrix is not None:
                avg = np.sum(self._dia_matrix * np.absolute(quantum_state) ** 2)
            else:
                avg = np.vdot(quantum_state, self._matrix.dot(quantum_state))
        else:
            #TODO convert to another representation
            self._check_representation("grouped_paulis")
            with concurrent.futures.ProcessPoolExecutor(max_workers=cpu_count) as executor:
                futures = [executor.submit(Operator._routine_grouped_paulis_with_shots, tpb_set,
                                           result.get_counts(circuits[tpb_idx]))
                           for tpb_idx, tpb_set in enumerate(self._grouped_paulis)]

                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    avg += result[0]
                    variance += result[1]

            std_dev = np.sqrt(variance / num_shots)

        return avg, std_dev

    def _eval_directly(self, quantum_state):
        if self._dia_matrix is None:
            self._to_dia_matrix(mode='matrix')
        if self._dia_matrix is not None:
            avg = np.sum(self._dia_matrix * np.absolute(quantum_state) ** 2)
        else:
            avg = np.vdot(quantum_state, self._matrix.dot(quantum_state))
        return avg

    # TODO replace input circuit with different states
    # TODO replace with QuantumInstance?
    def eval(self, input_circuit, backend, backend_config=None, compile_config=None,
             run_config=None, qjob_config=None, noise_config=None):
        """
        Supporting three ways to evaluate the given circuits with the operator.
        1. If `input_circuit` is a numpy.ndarray, it will directly perform inner product with the operator.
        2. If `backend` is a statevector simulator, use quantum backend to get statevector \
           and then evaluate with the operator.
        3. Other cases: it use with quanutm backend (simulator or real quantum machine), \
           to obtain the mean and standard deviation of measured results.

        Args:
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
                has_shared_circuits = True

                if operator_mode == 'matrix':
                    has_shared_circuits = False
            else:
                has_shared_circuits = False

            circuits = self.construct_evaluation_circuit(operator_mode, input_circuit, backend)
            result = compile_and_run_circuits(circuits, backend=backend, backend_config=backend_config,
                                              compile_config=compile_config, run_config=run_config,
                                              qjob_config=qjob_config, noise_config=noise_config, show_circuit_summary=self._summarize_circuits,
                                              has_shared_circuits=has_shared_circuits)
            avg, std_dev = self.evaluate_with_result(operator_mode, circuits, backend, result)

        return avg, std_dev

    # TODO change to generate representation and then call operator constructor
    def to_paulis(self):
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
        return Operator(paulis=paulis)

    # TODO change to generate representation and then call operator constructor
    def to_grouped_paulis(self):
        self._check_representation('grouped_paulis')

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

    #TODO convert
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

    #TODO convert
    def evolve(self, state_in, evo_time, evo_mode, num_time_slices, quantum_registers=None,
               expansion_mode='trotter', expansion_order=1):
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
        if self._matrix is None and self._dia_matrix is None
            return True
        else:
            return False

    #TODO probably remove?
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
                    raise AquaError(
                        "at least having one of the three operator representations.")

        elif targeted_represnetation == 'grouped_paulis':
            if self._grouped_paulis is None:
                if self._paulis is not None:
                    self._paulis_to_grouped_paulis()
                elif self._matrix is not None:
                    self._matrix_to_grouped_paulis()
                else:
                    raise AquaError(
                        "at least having one of the three operator representations.")

        elif targeted_represnetation == 'matrix':
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
                '"targeted_represnetation" should be one of "paulis", "grouped_paulis" and "matrix".'
            )

    #TODO make not static
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

    # TODO make not static
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

    def scaling_coeff(self, scaling_factor):
        """
        Constant scale the coefficient in an operator.

        Note that: the behavior of scaling in paulis (grouped_paulis) might be different from matrix

        Args:
            scaling_factor (float): the sacling factor
        """
        self._matrix *= scaling_factor
        if self._dia_matrix is not None:
            self._dia_matrix *= scaling_factor
