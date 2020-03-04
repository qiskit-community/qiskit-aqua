# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" QEquationOfMotion algorithm """

from typing import Optional, List, Union
import logging
import copy
import itertools
import sys

import numpy as np
from scipy import linalg
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar
from qiskit.aqua import AquaError, aqua_globals
from qiskit.aqua.operators import (LegacyBaseOperator,
                                   WeightedPauliOperator,
                                   Z2Symmetries,
                                   TPBGroupedWeightedPauliOperator,
                                   commutator)
from qiskit.aqua.operators.legacy import op_converter

from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry import FermionicOperator

logger = logging.getLogger(__name__)


class QEquationOfMotion:
    """ QEquationOfMotion algorithm """
    def __init__(self, operator: LegacyBaseOperator,
                 num_orbitals: int,
                 num_particles: Union[List[int], int],
                 qubit_mapping: Optional[str] = None,
                 two_qubit_reduction: bool = False,
                 active_occupied: Optional[List[int]] = None,
                 active_unoccupied: Optional[List[int]] = None,
                 is_eom_matrix_symmetric: bool = True,
                 se_list: Optional[List[List[int]]] = None,
                 de_list: Optional[List[List[int]]] = None,
                 z2_symmetries: Optional[Z2Symmetries] = None,
                 untapered_op: Optional[LegacyBaseOperator] = None) -> None:
        """Constructor.

        Args:
            operator: qubit operator
            num_orbitals: total number of spin orbitals
            num_particles: number of particles, if it is a list,
                                        the first number
                                        is alpha and the second number if beta.
            qubit_mapping: qubit mapping type
            two_qubit_reduction: two qubit reduction is applied or not
            active_occupied: list of occupied orbitals to include, indices are
                                    0 to n where n is num particles // 2
            active_unoccupied: list of unoccupied orbitals to include, indices are
                                    0 to m where m is (num_orbitals - num particles) // 2
            is_eom_matrix_symmetric: is EoM matrix symmetric
            se_list: single excitation list, overwrite the setting in active space
            de_list: double excitation list, overwrite the setting in active space
            z2_symmetries: represent the Z2 symmetries
            untapered_op: if the operator is tapered, we need untapered operator
                            to build element of EoM matrix
        """
        self._operator = operator
        self._num_orbitals = num_orbitals
        self._num_particles = num_particles
        self._qubit_mapping = qubit_mapping
        self._two_qubit_reduction = two_qubit_reduction
        self._active_occupied = active_occupied
        self._active_unoccupied = active_unoccupied

        se_list_default, de_list_default = UCCSD.compute_excitation_lists(
            self._num_particles, self._num_orbitals, self._active_occupied, self._active_unoccupied)

        if se_list is None:
            self._se_list = se_list_default
        else:
            self._se_list = se_list
            logger.info("Use user-specified single excitation list: %s", self._se_list)

        if de_list is None:
            self._de_list = de_list_default
        else:
            self._de_list = de_list
            logger.info("Use user-specified double excitation list: %s", self._de_list)

        self._z2_symmetries = z2_symmetries if z2_symmetries is not None \
            else Z2Symmetries([], [], [])
        self._untapered_op = untapered_op if untapered_op is not None else operator

        self._is_eom_matrix_symmetric = is_eom_matrix_symmetric

    def calculate_excited_states(self, wave_fn, excitations_list=None, quantum_instance=None):
        """Calculate energy gap of excited states from the reference state.

        Args:
            wave_fn (Union(QuantumCircuit, numpy.ndarray)): wavefunction of reference state
            excitations_list (list): excitation list for calculating the excited states
            quantum_instance (QuantumInstance): a quantum instance with configured settings

        Returns:
            list: energy gaps to the reference state
            dict: information of qeom matrices

        Raises:
            ValueError: wrong setting for wave_fn and quantum_instance
        """
        if isinstance(wave_fn, QuantumCircuit):
            if quantum_instance is None:
                raise ValueError("quantum_instance is required when wavn_fn is a QuantumCircuit.")
            temp_quantum_instance = copy.deepcopy(quantum_instance)
            if temp_quantum_instance.is_statevector and temp_quantum_instance.noise_config == {}:
                initial_statevector = quantum_instance.execute(wave_fn).get_statevector(wave_fn)
                q = QuantumRegister(self._operator.num_qubits, name='q')
                tmp_wave_fn = QuantumCircuit(q)
                tmp_wave_fn.append(wave_fn.to_instruction(), q)
                logger.info("Under noise-free and statevector simulation, "
                            "the wave_fn is reused and set in initial_statevector "
                            "for faster simulation.")
                temp_quantum_instance.set_config(initial_statevector=initial_statevector)
                wave_fn = QuantumCircuit(q)
        else:
            temp_quantum_instance = None

        # this is required to assure paulis mode is there regardless how you compute VQE
        # it might be slow if you calculate vqe through matrix mode and then convert
        # it back to paulis
        self._operator = op_converter.to_weighted_pauli_operator(self._operator)
        self._untapered_op = op_converter.to_weighted_pauli_operator(self._untapered_op)

        excitations_list = self._de_list + self._se_list if excitations_list is None \
            else excitations_list

        # build all hopping operators
        hopping_operators, type_of_commutativities = self.build_hopping_operators(excitations_list)
        # build all commutators
        q_commutators, w_commutators, m_commutators, v_commutators, available_entry = \
            self.build_all_commutators(excitations_list, hopping_operators, type_of_commutativities)
        # build qeom matrices (the step involves quantum)
        m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std = \
            self.build_eom_matrices(excitations_list, q_commutators, w_commutators,
                                    m_commutators, v_commutators, available_entry,
                                    wave_fn, temp_quantum_instance)
        excitation_energies_gap = self.compute_excitation_energies(m_mat, v_mat, q_mat, w_mat)

        logger.info('Net excited state values (gap to reference state): %s',
                    excitation_energies_gap)

        eom_matrices = {'m_mat': m_mat, 'v_mat': v_mat, 'q_mat': q_mat, 'w_mat': w_mat,
                        'm_mat_std': m_mat_std, 'v_mat_std': v_mat_std,
                        'q_mat_std': q_mat_std, 'w_mat_std': w_mat_std}

        return excitation_energies_gap, eom_matrices

    def build_hopping_operators(self, excitations_list):
        """Building all hopping operators defined in excitation list.

        Args:
            excitations_list (list): single excitations list + double excitation list

        Returns:
            dict: all hopping operators based on excitations_list,
                    key is the string of single/double excitation;
                    value is corresponding operator.
        """
        size = len(excitations_list)

        # get all to-be-processed index
        if self._is_eom_matrix_symmetric:
            mus, nus = np.triu_indices(size)
        else:
            mus, nus = np.indices((size, size))
            mus = np.asarray(mus.flat)
            nus = np.asarray(nus.flat)

        # build all hopping operators
        hopping_operators = {}
        type_of_commutativities = {}
        to_be_executed_list = []
        for idx, _ in enumerate(mus):
            m_u = mus[idx]
            n_u = nus[idx]
            for excitations in [excitations_list[m_u], excitations_list[n_u],
                                list(reversed(excitations_list[n_u]))]:
                key = '_'.join([str(x) for x in excitations])
                if key not in hopping_operators:
                    to_be_executed_list.append(excitations)
                    hopping_operators[key] = None
                    type_of_commutativities[key] = None

        result = parallel_map(QEquationOfMotion._build_single_hopping_operator,
                              to_be_executed_list,
                              task_args=(self._num_particles, self._num_orbitals,
                                         self._qubit_mapping,
                                         self._two_qubit_reduction, self._z2_symmetries),
                              num_processes=aqua_globals.num_processes)

        for excitations, res in zip(to_be_executed_list, result):
            key = '_'.join([str(x) for x in excitations])
            hopping_operators[key] = res[0]
            type_of_commutativities[key] = res[1]

        return hopping_operators, type_of_commutativities

    def build_all_commutators(self, excitations_list, hopping_operators, type_of_commutativities):
        """Building all commutators for Q, W, M, V matrices.

        Args:
            excitations_list (list): single excitations list + double excitation list
            hopping_operators (dict): all hopping operators based on excitations_list,
                                      key is the string of single/double excitation;
                                      value is corresponding operator.
            type_of_commutativities (dict): if tapering is used, it records the commutativities of
                                     hopping operators with the
                                     Z2 symmetries found in the original operator.
        Returns:
            dict: key: a string of matrix indices; value: the commutators for Q matrix
            dict: key: a string of matrix indices; value: the commutators for W matrix
            dict: key: a string of matrix indices; value: the commutators for M matrix
            dict: key: a string of matrix indices; value: the commutators for V matrix
            int: number of entries in the matrix
        """
        size = len(excitations_list)
        m_commutators = np.empty((size, size), dtype=object)
        v_commutators = np.empty((size, size), dtype=object)
        q_commutators = np.empty((size, size), dtype=object)
        w_commutators = np.empty((size, size), dtype=object)
        # get all to-be-processed index
        if self._is_eom_matrix_symmetric:
            mus, nus = np.triu_indices(size)
        else:
            mus, nus = np.indices((size, size))
            mus = np.asarray(mus.flat)
            nus = np.asarray(nus.flat)

        def _build_one_sector(available_hopping_ops):

            to_be_computed_list = []
            for idx, _ in enumerate(mus):
                m_u = mus[idx]
                n_u = nus[idx]
                left_op = available_hopping_ops.get(
                    '_'.join([str(x) for x in excitations_list[m_u]]), None)
                right_op_1 = available_hopping_ops.get(
                    '_'.join([str(x) for x in excitations_list[n_u]]), None)
                right_op_2 = available_hopping_ops.get(
                    '_'.join([str(x) for x in reversed(excitations_list[n_u])]), None)
                to_be_computed_list.append((m_u, n_u, left_op, right_op_1, right_op_2))

            if logger.isEnabledFor(logging.INFO):
                logger.info("Building all commutators:")
                TextProgressBar(sys.stderr)
            results = parallel_map(QEquationOfMotion._build_commutator_rountine,
                                   to_be_computed_list,
                                   task_args=(self._untapered_op, self._z2_symmetries),
                                   num_processes=aqua_globals.num_processes)
            for result in results:
                m_u, n_u, q_mat_op, w_mat_op, m_mat_op, v_mat_op = result
                q_commutators[m_u][n_u] = op_converter.to_tpb_grouped_weighted_pauli_operator(
                    q_mat_op, TPBGroupedWeightedPauliOperator.sorted_grouping) \
                    if q_mat_op is not None else q_commutators[m_u][n_u]
                w_commutators[m_u][n_u] = op_converter.to_tpb_grouped_weighted_pauli_operator(
                    w_mat_op, TPBGroupedWeightedPauliOperator.sorted_grouping) \
                    if w_mat_op is not None else w_commutators[m_u][n_u]
                m_commutators[m_u][n_u] = op_converter.to_tpb_grouped_weighted_pauli_operator(
                    m_mat_op, TPBGroupedWeightedPauliOperator.sorted_grouping) \
                    if m_mat_op is not None else m_commutators[m_u][n_u]
                v_commutators[m_u][n_u] = op_converter.to_tpb_grouped_weighted_pauli_operator(
                    v_mat_op, TPBGroupedWeightedPauliOperator.sorted_grouping) \
                    if v_mat_op is not None else v_commutators[m_u][n_u]

        available_entry = 0
        if not self._z2_symmetries.is_empty():
            for targeted_tapering_values in \
                    itertools.product([1, -1], repeat=len(self._z2_symmetries.symmetries)):
                logger.info("In sector: (%s)", ','.join([str(x) for x in targeted_tapering_values]))
                # remove the excited operators which are not suitable for the sector
                available_hopping_ops = {}
                targeted_sector = (np.asarray(targeted_tapering_values) == 1)
                for key, value in type_of_commutativities.items():
                    value = np.asarray(value)
                    if np.all(value == targeted_sector):
                        available_hopping_ops[key] = hopping_operators[key]
                _build_one_sector(available_hopping_ops)
                available_entry += len(available_hopping_ops) * len(available_hopping_ops)
        else:
            available_hopping_ops = hopping_operators
            _build_one_sector(available_hopping_ops)
            available_entry = len(available_hopping_ops) * len(available_hopping_ops)

        return q_commutators, w_commutators, m_commutators, v_commutators, available_entry

    def build_eom_matrices(self, excitations_list, q_commutators, w_commutators,
                           m_commutators, v_commutators, available_entry,
                           wave_fn, quantum_instance=None):
        """Compute M, V, Q and W matrices.

        Args:
            excitations_list (list): single excitations list + double excitation list
            q_commutators (dict): key: a string of matrix indices; value:
                                    the commutators for Q matrix
            w_commutators (dict): key: a string of matrix indices; value:
                                    the commutators for W matrix
            m_commutators (dict): key: a string of matrix indices; value:
                                    the commutators for M matrix
            v_commutators (dict): key: a string of matrix indices; value:
                                    the commutators for V matrix
            available_entry (int): number of entries in the matrix
            wave_fn (QuantumCircuit or numpy.ndarray): the circuit generated wave function
            for the ground state energy
            quantum_instance (QuantumInstance): a quantum instance with configured settings

        Returns:
            numpy.ndarray: M matrix
            numpy.ndarray: V matrix
            numpy.ndarray: Q matrix
            numpy.ndarray: W matrix

        Raises:
            AquaError: wrong setting for wave_fn and quantum_instance
        """
        if isinstance(wave_fn, QuantumCircuit) and quantum_instance is None:
            raise AquaError("quantum_instance is required when wavn_fn is a QuantumCircuit.")

        size = len(excitations_list)
        logger.info('EoM matrix size is %sx%s.', size, size)

        # get all to-be-processed index
        if self._is_eom_matrix_symmetric:
            mus, nus = np.triu_indices(size)
        else:
            mus, nus = np.indices((size, size))
            mus = np.asarray(mus.flat)
            nus = np.asarray(nus.flat)

        m_mat = np.zeros((size, size), dtype=complex)
        v_mat = np.zeros((size, size), dtype=complex)
        q_mat = np.zeros((size, size), dtype=complex)
        w_mat = np.zeros((size, size), dtype=complex)
        m_mat_std, v_mat_std, q_mat_std, w_mat_std = 0, 0, 0, 0

        if quantum_instance is not None:

            circuit_names = []
            circuits = []
            for idx, _ in enumerate(mus):
                m_u = mus[idx]
                n_u = nus[idx]

                for op in [q_commutators[m_u][n_u], w_commutators[m_u][n_u],
                           m_commutators[m_u][n_u], v_commutators[m_u][n_u]]:
                    if op is not None and not op.is_empty():
                        curr_circuits = op.construct_evaluation_circuit(
                            wave_function=wave_fn, statevector_mode=quantum_instance.is_statevector)
                        for c in curr_circuits:
                            if c.name not in circuit_names:
                                circuits.append(c)
                                circuit_names.append(c.name)

            result = quantum_instance.execute(circuits)

            # evaluate results
            for idx, _ in enumerate(mus):
                m_u = mus[idx]
                n_u = nus[idx]

                def _get_result(op):
                    mean, std = 0.0, 0.0
                    if op is not None and not op.is_empty():
                        mean, std = \
                            op.evaluate_with_result(
                                result=result,
                                statevector_mode=quantum_instance.is_statevector)
                    return mean, std

                q_mean, q_std = _get_result(q_commutators[m_u][n_u])
                w_mean, w_std = _get_result(w_commutators[m_u][n_u])
                m_mean, m_std = _get_result(m_commutators[m_u][n_u])
                v_mean, v_std = _get_result(v_commutators[m_u][n_u])

                q_mat[m_u][n_u] = q_mean if q_mean != 0.0 else q_mat[m_u][n_u]
                w_mat[m_u][n_u] = w_mean if w_mean != 0.0 else w_mat[m_u][n_u]
                m_mat[m_u][n_u] = m_mean if m_mean != 0.0 else m_mat[m_u][n_u]
                v_mat[m_u][n_u] = v_mean if v_mean != 0.0 else v_mat[m_u][n_u]
                q_mat_std += q_std
                w_mat_std += w_std
                m_mat_std += m_std
                v_mat_std += v_std
        else:
            for idx, _ in enumerate(mus):
                m_u = mus[idx]
                n_u = nus[idx]
                q_mean, q_std = q_commutators[m_u][n_u].evaluate_with_statevector(wave_fn) \
                    if q_commutators[m_u][n_u] is not None else (0.0, 0.0)
                w_mean, w_std = w_commutators[m_u][n_u].evaluate_with_statevector(wave_fn) \
                    if w_commutators[m_u][n_u] is not None else (0.0, 0.0)
                m_mean, m_std = m_commutators[m_u][n_u].evaluate_with_statevector(wave_fn) \
                    if m_commutators[m_u][n_u] is not None else (0.0, 0.0)
                v_mean, v_std = v_commutators[m_u][n_u].evaluate_with_statevector(wave_fn) \
                    if v_commutators[m_u][n_u] is not None else (0.0, 0.0)
                q_mat[m_u][n_u] = q_mean if q_mean != 0.0 else q_mat[m_u][n_u]
                w_mat[m_u][n_u] = w_mean if w_mean != 0.0 else w_mat[m_u][n_u]
                m_mat[m_u][n_u] = m_mean if m_mean != 0.0 else m_mat[m_u][n_u]
                v_mat[m_u][n_u] = v_mean if v_mean != 0.0 else v_mat[m_u][n_u]

        # pylint: disable=unsubscriptable-object
        if self._is_eom_matrix_symmetric:
            q_mat = q_mat + q_mat.T - np.identity(q_mat.shape[0]) * q_mat
            w_mat = w_mat + w_mat.T - np.identity(w_mat.shape[0]) * w_mat
            m_mat = m_mat + m_mat.T - np.identity(m_mat.shape[0]) * m_mat
            v_mat = v_mat + v_mat.T - np.identity(v_mat.shape[0]) * v_mat

        q_mat = np.real(q_mat)
        w_mat = np.real(w_mat)
        m_mat = np.real(m_mat)
        v_mat = np.real(v_mat)

        q_mat_std = q_mat_std / float(available_entry)
        w_mat_std = w_mat_std / float(available_entry)
        m_mat_std = m_mat_std / float(available_entry)
        v_mat_std = v_mat_std / float(available_entry)

        logger.debug("\nQ:=========================\n%s", q_mat)
        logger.debug("\nW:=========================\n%s", w_mat)
        logger.debug("\nM:=========================\n%s", m_mat)
        logger.debug("\nV:=========================\n%s", v_mat)

        return m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std

    @staticmethod
    def compute_excitation_energies(m_mat, v_mat, q_mat, w_mat):
        """Diagonalizing M, V, Q, W matrices for excitation energies.

        Args:
            m_mat (numpy.ndarray): M matrices
            v_mat (numpy.ndarray): V matrices
            q_mat (numpy.ndarray): Q matrices
            w_mat (numpy.ndarray): W matrices

        Returns:
            numpy.ndarray: 1-D vector stores all energy gap to reference state
        """
        logger.debug('Diagonalizing qeom matrices for excited states...')
        a_mat = np.bmat([[m_mat, q_mat], [q_mat.T.conj(), m_mat.T.conj()]])
        b_mat = np.bmat([[v_mat, w_mat], [-w_mat.T.conj(), -v_mat.T.conj()]])
        # pylint: disable=too-many-function-args
        res = linalg.eig(a_mat, b_mat)
        # convert nan value into 0
        res[0][np.where(np.isnan(res[0]))] = 0.0
        # Only the positive eigenvalues are physical. We need to take care
        # though of very small values
        # should an excited state approach ground state. Here the small values
        # may be both negative or
        # positive. We should take just one of these pairs as zero. So to get the values we want we
        # sort the real parts and then take the upper half of the sorted values.
        # Since we may now have
        # small values (positive or negative) take the absolute and then threshold zero.
        logger.debug('... %s', res[0])
        w = np.sort(np.real(res[0]))
        logger.debug('Sorted real parts %s', w)
        w = np.abs(w[len(w) // 2:])
        w[w < 1e-06] = 0
        excitation_energies_gap = w
        return excitation_energies_gap

    @staticmethod
    def _build_single_hopping_operator(index, num_particles, num_orbitals, qubit_mapping,
                                       two_qubit_reduction, z2_symmetries):

        h_1 = np.zeros((num_orbitals, num_orbitals), dtype=complex)
        h_2 = np.zeros((num_orbitals, num_orbitals, num_orbitals, num_orbitals), dtype=complex)
        if len(index) == 2:
            i, j = index
            h_1[i, j] = 4.0
        elif len(index) == 4:
            i, j, k, m = index
            h_2[i, j, k, m] = 16.0
        fer_op = FermionicOperator(h_1, h_2)
        qubit_op = fer_op.mapping(qubit_mapping)
        if two_qubit_reduction:
            qubit_op = Z2Symmetries.two_qubit_reduction(qubit_op, num_particles)

        commutativities = []
        if not z2_symmetries.is_empty():
            for symmetry in z2_symmetries.symmetries:
                symmetry_op = WeightedPauliOperator(paulis=[[1.0, symmetry]])
                commuting = qubit_op.commute_with(symmetry_op)
                anticommuting = qubit_op.anticommute_with(symmetry_op)

                if commuting != anticommuting:  # only one of them is True
                    if commuting:
                        commutativities.append(True)
                    elif anticommuting:
                        commutativities.append(False)
                else:
                    raise AquaError("Symmetry {} is nor commute neither anti-commute "
                                    "to exciting operator.".format(symmetry.to_label()))

        return qubit_op, commutativities

    @staticmethod
    def _build_commutator_rountine(params, operator, z2_symmetries):
        m_u, n_u, left_op, right_op_1, right_op_2 = params
        if left_op is None:
            q_mat_op = None
            w_mat_op = None
            m_mat_op = None
            v_mat_op = None
        else:
            if right_op_1 is None and right_op_2 is None:
                q_mat_op = None
                w_mat_op = None
                m_mat_op = None
                v_mat_op = None
            else:
                if right_op_1 is not None:
                    q_mat_op = commutator(left_op, operator, right_op_1)
                    w_mat_op = commutator(left_op, right_op_1)
                    q_mat_op = None if q_mat_op.is_empty() else q_mat_op
                    w_mat_op = None if w_mat_op.is_empty() else w_mat_op
                else:
                    q_mat_op = None
                    w_mat_op = None

                if right_op_2 is not None:
                    m_mat_op = commutator(left_op, operator, right_op_2)
                    v_mat_op = commutator(left_op, right_op_2)
                    m_mat_op = None if m_mat_op.is_empty() else m_mat_op
                    v_mat_op = None if v_mat_op.is_empty() else v_mat_op
                else:
                    m_mat_op = None
                    v_mat_op = None

                if not z2_symmetries.is_empty():
                    if q_mat_op is not None and not q_mat_op.is_empty():
                        q_mat_op = z2_symmetries.taper(q_mat_op)
                    if w_mat_op is not None and not w_mat_op.is_empty():
                        w_mat_op = z2_symmetries.taper(w_mat_op)
                    if m_mat_op is not None and not m_mat_op.is_empty():
                        m_mat_op = z2_symmetries.taper(m_mat_op)
                    if v_mat_op is not None and not v_mat_op.is_empty():
                        v_mat_op = z2_symmetries.taper(v_mat_op)

        return m_u, n_u, q_mat_op, w_mat_op, m_mat_op, v_mat_op
