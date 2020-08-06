# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""The HHL algorithm."""

from typing import Optional, Union, Dict, Any
import logging
from copy import deepcopy
import numpy as np

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers import BaseBackend
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.ignis.verification.tomography import state_tomography_circuits, \
    StateTomographyFitter
from qiskit.converters import circuit_to_dag
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.components.reciprocals import Reciprocal
from qiskit.aqua.components.eigs import Eigenvalues

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class HHL(QuantumAlgorithm):
    r"""The HHL algorithm.

    The HHL algorithm (after the author's surnames Harrow-Hassidim-Lloyd) is a quantum algorithm
    to solve systems of linear equations :math:`A\overrightarrow{x}=\overrightarrow{b}`.
    Using Quantum Phase Estimation, the linear system is transformed into diagonal form in which
    the matrix :math:`A` is easily invertible. The inversion is achieved by rotating an ancillary
    qubit by an angle :math:`\arcsin{ \frac{C}{\lambda_\mathrm{i}}}` around the y-axis where
    :math:`\lambda_\mathrm{i}` are the eigenvalues of :math:`A`. After uncomputing the register
    storing the eigenvalues using the inverse QPE, one measures the ancillary qubit. A measurement
    of 1 indicates that the matrix inversion succeeded. This leaves the system in a state
    proportional to the solution vector :math:`|x\rangle`. In many cases one is not interested in
    the single vector elements of :math:`|x\rangle` but only on certain properties. These are
    accessible by using problem-specific operators. Another use-case is the implementation in a
    larger quantum program.

    When using non-hermitian matrices and matrices with dimensions other than :math:`2^{n}`
    the must be converted to an hermitian matrix and next higher dimension :math:`2^{n}`,
    respectively. The *truncate_hermitian*, *truncate_powerdim* flags and *orig_size* are used
    to indicate conversion and the returned result of the HHL algorithm for expanded matrices will
    be truncated. The :meth:`matrix_resize` method is provided for convenience to do this but
    any method of your choice can be used.

    To further explain *truncate_hermitian* indicates whether or not to truncate matrix and
    result vector to half the dimension by simply cutting off entries with other indices after
    the input matrix was expanded to be hermitian following

    .. math::

       \begin{pmatrix}
       0 & A^\mathsf{H}\\
       A & 0
       \end{pmatrix}

    where the conjugate transpose of matrix :math:`A` is denoted by :math:`A^\mathsf{H}`.
    The truncation of the result vector is done by simply cutting off entries of the upper half.

    *truncate_powerdim* indicates whether to truncate matrix and result vector from
    dimension :math:`2^{n}` to dimension given by *orig_size* by simply cutting off entries with
    larger indices.

    Running the algorithm will execute the circuit and return the result
    vector, measured (real hardware backend) or derived (qasm_simulator) via
    state tomography or calculated from the statevector (statevector_simulator).

    See also https://arxiv.org/abs/0811.3171
    """

    def __init__(
            self,
            matrix: np.ndarray,
            vector: np.ndarray,
            truncate_powerdim: bool = False,
            truncate_hermitian: bool = False,
            eigs: Optional[Eigenvalues] = None,
            init_state: Optional[InitialState] = None,
            reciprocal: Optional[Reciprocal] = None,
            num_q: int = 0,
            num_a: int = 0,
            orig_size: Optional[int] = None,
            quantum_instance: Optional[Union[QuantumInstance, BaseBackend]] = None) -> None:
        """
        Args:
            matrix: The input matrix of linear system of equations
            vector: The input vector of linear system of equations
            truncate_powerdim: Flag indicating expansion to 2**n matrix to be truncated
            truncate_hermitian: Flag indicating expansion to hermitian matrix to be truncated
            eigs: The eigenvalue estimation instance
            init_state: The initial quantum state preparation
            reciprocal: The eigenvalue reciprocal and controlled rotation instance
            num_q: Number of qubits required for the matrix Operator instance
            num_a: Number of ancillary qubits for Eigenvalues instance
            orig_size: The original dimension of the problem (if truncate_powerdim)
            quantum_instance: Quantum Instance or Backend
        Raises:
            ValueError: Invalid input
        """
        super().__init__(quantum_instance)
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square!")
        if matrix.shape[0] != len(vector):
            raise ValueError("Input vector dimension does not match input "
                             "matrix dimension!")
        if not np.allclose(matrix, matrix.conj().T):
            raise ValueError("Input matrix must be hermitian!")
        if np.log2(matrix.shape[0]) % 1 != 0:
            raise ValueError("Input matrix dimension must be 2**n!")
        if truncate_powerdim and orig_size is None:
            raise ValueError("Truncation to {} dimensions is not "
                             "possible!".format(orig_size))

        self._matrix = matrix
        self._vector = vector
        self._truncate_powerdim = truncate_powerdim
        self._truncate_hermitian = truncate_hermitian
        self._eigs = eigs
        self._init_state = init_state
        self._reciprocal = reciprocal
        self._num_q = num_q
        self._num_a = num_a
        self._circuit = None
        self._io_register = None
        self._eigenvalue_register = None
        self._ancilla_register = None
        self._success_bit = None
        self._original_dimension = orig_size
        self._ret = {}  # type: Dict[str, Any]

    @staticmethod
    def matrix_resize(matrix, vector):
        """Resizes matrix if necessary

        Args:
            matrix (np.array): the input matrix of linear system of equations
            vector (np.array): the input vector of linear system of equations
        Returns:
            tuple: new matrix, vector, truncate_powerdim, truncate_hermitian
        Raises:
            ValueError: invalid input
        """
        if not isinstance(matrix, np.ndarray):
            matrix = np.asarray(matrix)
        if not isinstance(vector, np.ndarray):
            vector = np.asarray(vector)

        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input matrix must be square!")
        if matrix.shape[0] != len(vector):
            raise ValueError("Input vector dimension does not match input "
                             "matrix dimension!")

        truncate_powerdim = False
        truncate_hermitian = False
        orig_size = None
        if orig_size is None:
            orig_size = len(vector)

        is_powerdim = np.log2(matrix.shape[0]) % 1 == 0
        if not is_powerdim:
            logger.warning("Input matrix does not have dimension 2**n. It "
                           "will be expanded automatically.")
            matrix, vector = HHL.expand_to_powerdim(matrix, vector)
            truncate_powerdim = True

        is_hermitian = np.allclose(matrix, matrix.conj().T)
        if not is_hermitian:
            logger.warning("Input matrix is not hermitian. It will be "
                           "expanded to a hermitian matrix automatically.")
            matrix, vector = HHL.expand_to_hermitian(matrix, vector)
            truncate_hermitian = True

        return (matrix, vector, truncate_powerdim, truncate_hermitian)

    def construct_circuit(self, measurement=False):
        """Construct the HHL circuit.

        Args:
            measurement (bool): indicate whether measurement on ancillary qubit
                should be performed

        Returns:
            QuantumCircuit: the QuantumCircuit object for the constructed circuit
        """

        q = QuantumRegister(self._num_q, name="io")
        qc = QuantumCircuit(q)

        # InitialState
        qc += self._init_state.construct_circuit("circuit", q)

        # EigenvalueEstimation (QPE)
        qc += self._eigs.construct_circuit("circuit", q)
        a = self._eigs._output_register

        # Reciprocal calculation with rotation
        qc += self._reciprocal.construct_circuit("circuit", a)
        s = self._reciprocal._anc

        # Inverse EigenvalueEstimation
        qc += self._eigs.construct_inverse("circuit", self._eigs._circuit)

        # Measurement of the ancilla qubit
        if measurement:
            c = ClassicalRegister(1)
            qc.add_register(c)
            qc.measure(s, c)
            self._success_bit = c

        self._io_register = q
        self._eigenvalue_register = a
        self._ancilla_register = s
        self._circuit = qc
        return qc

    @staticmethod
    def expand_to_powerdim(matrix, vector):
        """ Expand a matrix to the next-larger 2**n dimensional matrix with
        ones on the diagonal and zeros on the off-diagonal and expand the
        vector with zeros accordingly.

        Args:
            matrix (np.array): the input matrix
            vector (np.array): the input vector

        Returns:
           tuple(np.array, np.array): the expanded matrix, the expanded vector
        """
        mat_dim = matrix.shape[0]
        next_higher = int(np.ceil(np.log2(mat_dim)))
        new_matrix = np.identity(2 ** next_higher)
        new_matrix = np.array(new_matrix, dtype=complex)
        new_matrix[:mat_dim, :mat_dim] = matrix[:, :]
        matrix = new_matrix
        new_vector = np.zeros((1, 2 ** next_higher))
        new_vector[0, :vector.shape[0]] = vector
        vector = new_vector.reshape(np.shape(new_vector)[1])
        return matrix, vector

    @staticmethod
    def expand_to_hermitian(matrix, vector):
        """ Expand a non-hermitian matrix A to a hermitian matrix by
        [[0, A.H], [A, 0]] and expand vector b to [b.conj, b].

        Args:
            matrix (np.array): the input matrix
            vector (np.array): the input vector

        Returns:
            tuple(np.array, np.array): the expanded matrix, the expanded vector
        """
        #
        half_dim = matrix.shape[0]
        full_dim = 2 * half_dim
        new_matrix = np.zeros([full_dim, full_dim])
        new_matrix = np.array(new_matrix, dtype=complex)
        new_matrix[0:half_dim, half_dim:full_dim] = matrix[:, :]
        new_matrix[half_dim:full_dim, 0:half_dim] = matrix.conj().T[:, :]
        matrix = new_matrix
        new_vector = np.zeros((1, full_dim))
        new_vector = np.array(new_vector, dtype=complex)
        new_vector[0, :vector.shape[0]] = vector.conj()
        new_vector[0, vector.shape[0]:] = vector
        vector = new_vector.reshape(np.shape(new_vector)[1])
        return matrix, vector

    def _resize_vector(self, vec):
        if self._truncate_hermitian:
            half_dim = int(vec.shape[0] / 2)
            vec = vec[:half_dim]
        if self._truncate_powerdim:
            vec = vec[:self._original_dimension]
        return vec

    def _resize_matrix(self, matrix):
        if self._truncate_hermitian:
            full_dim = matrix.shape[0]
            half_dim = int(full_dim / 2)
            new_matrix = np.ndarray(shape=(half_dim, half_dim), dtype=complex)
            new_matrix[:, :] = matrix[0:half_dim, half_dim:full_dim]
            matrix = new_matrix
        if self._truncate_powerdim:
            new_matrix = \
                np.ndarray(shape=(self._original_dimension, self._original_dimension),
                           dtype=complex)
            new_matrix[:, :] = matrix[:self._original_dimension, :self._original_dimension]
            matrix = new_matrix
        return matrix

    def _statevector_simulation(self):
        """The statevector simulation.

        The HHL result gets extracted from the statevector. Only for
        statevector simulator available.
        """
        res = self._quantum_instance.execute(self._circuit)
        sv = np.asarray(res.get_statevector(self._circuit))
        # Extract solution vector from statevector
        vec = self._reciprocal.sv_to_resvec(sv, self._num_q)
        # remove added dimensions
        self._ret['probability_result'] = \
            np.real(self._resize_vector(vec).dot(self._resize_vector(vec).conj()))
        vec = vec / np.linalg.norm(vec)
        self._hhl_results(vec)

    def _state_tomography(self):
        """The state tomography.

        The HHL result gets extracted via state tomography. Available for
        qasm simulator and real hardware backends.
        """

        # Preparing the state tomography circuits
        tomo_circuits = state_tomography_circuits(self._circuit,
                                                  self._io_register)
        tomo_circuits_noanc = deepcopy(tomo_circuits)
        ca = ClassicalRegister(1)
        for circ in tomo_circuits:
            circ.add_register(ca)
            circ.measure(self._reciprocal._anc, ca[0])

        # Extracting the probability of successful run
        results = self._quantum_instance.execute(tomo_circuits)
        probs = []
        for circ in tomo_circuits:
            counts = results.get_counts(circ)
            s, f = 0, 0
            for k, v in counts.items():
                if k[0] == "1":
                    s += v
                else:
                    f += v
            probs.append(s / (f + s))
        probs = self._resize_vector(probs)
        self._ret["probability_result"] = np.real(probs)

        # Filtering the tomo data for valid results with ancillary measured
        # to 1, i.e. c1==1
        results_noanc = self._tomo_postselect(results)
        tomo_data = StateTomographyFitter(results_noanc, tomo_circuits_noanc)
        rho_fit = tomo_data.fit('lstsq')
        vec = np.sqrt(np.diag(rho_fit))
        self._hhl_results(vec)

    def _tomo_postselect(self, results):
        new_results = deepcopy(results)

        for resultidx, _ in enumerate(results.results):
            old_counts = results.get_counts(resultidx)
            new_counts = {}

            # change the size of the classical register
            new_results.results[resultidx].header.creg_sizes = [
                new_results.results[resultidx].header.creg_sizes[0]]
            new_results.results[resultidx].header.clbit_labels = \
                new_results.results[resultidx].header.clbit_labels[0:-1]
            new_results.results[resultidx].header.memory_slots = \
                new_results.results[resultidx].header.memory_slots - 1

            for reg_key in old_counts:
                reg_bits = reg_key.split(' ')
                if reg_bits[0] == '1':
                    new_counts[reg_bits[1]] = old_counts[reg_key]

            data_counts = new_results.results[resultidx].data.counts
            new_results.results[resultidx].data.counts = \
                new_counts if isinstance(data_counts, dict) else data_counts.from_dict(new_counts)

        return new_results

    def _hhl_results(self, vec):
        res_vec = self._resize_vector(vec)
        in_vec = self._resize_vector(self._vector)
        matrix = self._resize_matrix(self._matrix)
        self._ret["output"] = res_vec
        # Rescaling the output vector to the real solution vector
        tmp_vec = matrix.dot(res_vec)
        f1 = np.linalg.norm(in_vec) / np.linalg.norm(tmp_vec)
        # "-1+1" to fix angle error for -0.-0.j
        f2 = sum(np.angle(in_vec * tmp_vec.conj() - 1 + 1)) / (np.log2(matrix.shape[0]))
        self._ret["solution"] = f1 * res_vec * np.exp(-1j * f2)

    def _run(self):
        if self._quantum_instance.is_statevector:
            self.construct_circuit(measurement=False)
            self._statevector_simulation()
        else:
            self.construct_circuit(measurement=False)
            self._state_tomography()
        # Adding a bit of general result information
        self._ret["matrix"] = self._resize_matrix(self._matrix)
        self._ret["vector"] = self._resize_vector(self._vector)
        self._ret["circuit_info"] = circuit_to_dag(self._circuit).properties()
        return self._ret
