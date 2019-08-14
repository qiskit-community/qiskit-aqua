# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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
from qiskit.aqua.operators import (WeightedPauliOperator, Z2Symmetries, TPBGroupedWeightedPauliOperator,
                                   op_converter, commutator)

from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
from qiskit.chemistry import FermionicOperator

logger = logging.getLogger(__name__)


class QEquationOfMotion:

    def __init__(self, operator, num_orbitals, num_particles,
                 qubit_mapping=None, two_qubit_reduction=False,
                 active_occupied=None, active_unoccupied=None,
                 is_eom_matrix_symmetric=True, se_list=None, de_list=None,
                 z2_symmetries=None, untapered_op=None):
        """Constructor.

        Args:
            operator (WeightedPauliOperator): qubit operator
            num_orbitals (int):  total number of spin orbitals
            num_particles (list, int): number of particles, if it is a list, the first number is alpha and the second
                                        number if beta.
            qubit_mapping (str): qubit mapping type
            two_qubit_reduction (bool): two qubit reduction is applied or not
            active_occupied (list): list of occupied orbitals to include, indices are
                                    0 to n where n is num particles // 2
            active_unoccupied (list): list of unoccupied orbitals to include, indices are
                                    0 to m where m is (num_orbitals - num particles) // 2
            is_eom_matrix_symmetric (bool): is EoM matrix symmetric
            se_list ([list]): single excitation list, overwrite the setting in active space
            de_list ([list]): double excitation list, overwrite the setting in active space
            z2_symmetries (Z2Symmetries): represent the Z2 symmetries
            untapered_op (WeightedPauliOperator): if the operator is tapered, we need untapered operator
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
            logger.info("Use user-specified single excitation list: {}".format(self._se_list))

        if de_list is None:
            self._de_list = de_list_default
        else:
            self._de_list = de_list
            logger.info("Use user-specified double excitation list: {}".format(self._de_list))

        self._z2_symmetries = z2_symmetries if z2_symmetries is not None else Z2Symmetries([], [], [])
        self._untapered_op = untapered_op if untapered_op is not None else operator

        self._is_eom_matrix_symmetric = is_eom_matrix_symmetric

    def calculate_excited_states(self, wave_fn, excitations_list=None, quantum_instance=None):
        """Calculate energy gap of excited states from the reference state.

        Args:
            wave_fn (QuantumCircuit | numpy.ndarray): wavefunction of reference state
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
                            "the wave_fn is reused and set in initial_statevector for faster simulation.")
                temp_quantum_instance.set_config(initial_statevector=initial_statevector)
                wave_fn = QuantumCircuit(q)
        else:
            temp_quantum_instance = None

        # this is required to assure paulis mode is there regardless how you compute VQE
        # it might be slow if you calculate vqe through matrix mode and then convert it back to paulis
        self._operator = op_converter.to_weighted_pauli_operator(self._operator)
        self._untapered_op = op_converter.to_weighted_pauli_operator(self._untapered_op)

        excitations_list = self._de_list + self._se_list if excitations_list is None else excitations_list

        # build all hopping operators
        hopping_operators, type_of_commutativities = self.build_hopping_operators(excitations_list)
        # build all commutators
        q_commutators, w_commutators, m_commutators, v_commutators, available_entry = self.build_all_commutators(
            excitations_list, hopping_operators, type_of_commutativities)
        # build qeom matrices (the step involves quantum)
        m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std = \
            self.build_eom_matrices(excitations_list, q_commutators, w_commutators,
                                    m_commutators, v_commutators, available_entry,
                                    wave_fn, temp_quantum_instance)
        excitation_energies_gap = self.compute_excitation_energies(m_mat, v_mat, q_mat, w_mat)

        logger.info('Net excited state values (gap to reference state): {}'.format(excitation_energies_gap))

        eom_matrices = {'m_mat': m_mat, 'v_mat': v_mat, 'q_mat': q_mat, 'w_mat': w_mat,
                        'm_mat_std': m_mat_std, 'v_mat_std': v_mat_std, 'q_mat_std': q_mat_std, 'w_mat_std': w_mat_std}

        return excitation_energies_gap, eom_matrices

    def build_hopping_operators(self, excitations_list):

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
        for idx in range(len(mus)):
            mu = mus[idx]
            nu = nus[idx]
            for excitations in [excitations_list[mu], excitations_list[nu], list(reversed(excitations_list[nu]))]:
                key = '_'.join([str(x) for x in excitations])
                if key not in hopping_operators:
                    to_be_executed_list.append(excitations)
                    hopping_operators[key] = None
                    type_of_commutativities[key] = None

        result = parallel_map(QEquationOfMotion._build_single_hopping_operator,
                              to_be_executed_list,
                              task_args=(self._num_particles, self._num_orbitals, self._qubit_mapping,
                                         self._two_qubit_reduction, self._z2_symmetries),
                              num_processes=aqua_globals.num_processes)

        for excitations, res in zip(to_be_executed_list, result):
            key = '_'.join([str(x) for x in excitations])
            hopping_operators[key] = res[0]
            type_of_commutativities[key] = res[1]

        return hopping_operators, type_of_commutativities

    def build_all_commutators(self, excitations_list, hopping_operators, type_of_commutativities):

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
            for idx in range(len(mus)):
                mu = mus[idx]
                nu = nus[idx]
                left_op = available_hopping_ops.get('_'.join([str(x) for x in excitations_list[mu]]), None)
                right_op_1 = available_hopping_ops.get('_'.join([str(x) for x in excitations_list[nu]]), None)
                right_op_2 = available_hopping_ops.get(
                    '_'.join([str(x) for x in reversed(excitations_list[nu])]), None)
                to_be_computed_list.append((mu, nu, left_op, right_op_1, right_op_2))

            if logger.isEnabledFor(logging.INFO):
                logger.info("Building all commutators:")
                TextProgressBar(sys.stderr)
            results = parallel_map(QEquationOfMotion._build_commutator_rountine,
                                   to_be_computed_list,
                                   task_args=(self._untapered_op, self._z2_symmetries))
            for result in results:
                mu, nu, q_mat_op, w_mat_op, m_mat_op, v_mat_op = result
                q_commutators[mu][nu] = op_converter.to_tpb_grouped_weighted_pauli_operator(q_mat_op, TPBGroupedWeightedPauliOperator.sorted_grouping) if q_mat_op is not None else q_commutators[mu][nu]
                w_commutators[mu][nu] = op_converter.to_tpb_grouped_weighted_pauli_operator(w_mat_op, TPBGroupedWeightedPauliOperator.sorted_grouping) if w_mat_op is not None else w_commutators[mu][nu]
                m_commutators[mu][nu] = op_converter.to_tpb_grouped_weighted_pauli_operator(m_mat_op, TPBGroupedWeightedPauliOperator.sorted_grouping) if m_mat_op is not None else m_commutators[mu][nu]
                v_commutators[mu][nu] = op_converter.to_tpb_grouped_weighted_pauli_operator(v_mat_op, TPBGroupedWeightedPauliOperator.sorted_grouping) if v_mat_op is not None else v_commutators[mu][nu]

        available_entry = 0
        if not self._z2_symmetries.is_empty():
            for targeted_tapering_values in itertools.product([1, -1], repeat=len(self._z2_symmetries.symmetries)):
                logger.info("In sector: ({})".format(','.join([str(x) for x in targeted_tapering_values])))
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
        """
        Compute M, V, Q and W matrices.

        Args:
            excitations_list (list): single excitations list + double excitation list
            wave_fn (QuantumCircuit or numpy.ndarray): the circuit generated wave function for the ground state energy
            q_commutators (dict):
            w_commutators (dict):
            m_commutators (dict):
            v_commutators (dict):
            available_entry (int):
            quantum_instance (QuantumInstance): a quantum instance with configured settings

        Returns:
            numpy.ndarray: M matrix
            numpy.ndarray: V matrix
            numpy.ndarray: Q matrix
            numpy.ndarray: W matrix

        Raises:
            ValueError: wrong setting for wave_fn and quantum_instance
        """
        if isinstance(wave_fn, QuantumCircuit) and quantum_instance is None:
            raise ValueError("quantum_instance is required when wavn_fn is a QuantumCircuit.")

        size = len(excitations_list)
        logger.info('EoM matrix size is {}x{}.'.format(size, size))

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
            for idx in range(len(mus)):
                mu = mus[idx]
                nu = nus[idx]

                for op in [q_commutators[mu][nu], w_commutators[mu][nu], m_commutators[mu][nu], v_commutators[mu][nu]]:
                    if op is not None and not op.is_empty():
                        curr_circuits = op.construct_evaluation_circuit(
                            wave_function=wave_fn, statevector_mode=quantum_instance.is_statevector)
                        for c in curr_circuits:
                            if c.name not in circuit_names:
                                circuits.append(c)
                                circuit_names.append(c.name)

            result = quantum_instance.execute(circuits)

            # evaluate results
            for idx in range(len(mus)):
                mu = mus[idx]
                nu = nus[idx]

                def _get_result(op):
                    mean, std = 0.0, 0.0
                    if op is not None and not op.is_empty():
                        mean, std = op.evaluate_with_result(result=result,
                                                            statevector_mode=quantum_instance.is_statevector)
                    return mean, std

                q_mean, q_std = _get_result(q_commutators[mu][nu])
                w_mean, w_std = _get_result(w_commutators[mu][nu])
                m_mean, m_std = _get_result(m_commutators[mu][nu])
                v_mean, v_std = _get_result(v_commutators[mu][nu])

                q_mat[mu][nu] = q_mean if q_mean != 0.0 else q_mat[mu][nu]
                w_mat[mu][nu] = w_mean if w_mean != 0.0 else w_mat[mu][nu]
                m_mat[mu][nu] = m_mean if m_mean != 0.0 else m_mat[mu][nu]
                v_mat[mu][nu] = v_mean if v_mean != 0.0 else v_mat[mu][nu]
                q_mat_std += q_std
                w_mat_std += w_std
                m_mat_std += m_std
                v_mat_std += v_std
        else:
            for idx in range(len(mus)):
                mu = mus[idx]
                nu = nus[idx]
                q_mean, q_std = q_commutators[mu][nu].evaluate_with_statevector(wave_fn) \
                    if q_commutators[mu][nu] is not None else (0.0, 0.0)
                w_mean, w_std = w_commutators[mu][nu].evaluate_with_statevector(wave_fn) \
                    if w_commutators[mu][nu] is not None else (0.0, 0.0)
                m_mean, m_std = m_commutators[mu][nu].evaluate_with_statevector(wave_fn) \
                    if m_commutators[mu][nu] is not None else (0.0, 0.0)
                v_mean, v_std = v_commutators[mu][nu].evaluate_with_statevector(wave_fn) \
                    if v_commutators[mu][nu] is not None else (0.0, 0.0)
                q_mat[mu][nu] = q_mean if q_mean != 0.0 else q_mat[mu][nu]
                w_mat[mu][nu] = w_mean if w_mean != 0.0 else w_mat[mu][nu]
                m_mat[mu][nu] = m_mean if m_mean != 0.0 else m_mat[mu][nu]
                v_mat[mu][nu] = v_mean if v_mean != 0.0 else v_mat[mu][nu]

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

        logger.debug("\nQ:=========================\n{}".format(q_mat))
        logger.debug("\nW:=========================\n{}".format(w_mat))
        logger.debug("\nM:=========================\n{}".format(m_mat))
        logger.debug("\nV:=========================\n{}".format(v_mat))

        return m_mat, v_mat, q_mat, w_mat, m_mat_std, v_mat_std, q_mat_std, w_mat_std

    @staticmethod
    def compute_excitation_energies(m_mat, v_mat, q_mat, w_mat):
        """
        Diagonalizing M, V, Q, W matrices for excitation energies.

        Args:
            m_mat (numpy.ndarray): M
            v_mat (numpy.ndarray): V
            q_mat (numpy.ndarray): Q
            w_mat (numpy.ndarray): W

        Returns:
            numpy.ndarray: 1-D vector stores all energy gap to reference state
        """
        logger.debug('Diagonalizing qeom matrices for excited states...')
        a_mat = np.bmat([[m_mat, q_mat], [q_mat.T.conj(), m_mat.T.conj()]])
        b_mat = np.bmat([[v_mat, w_mat], [-w_mat.T.conj(), -v_mat.T.conj()]])
        res = linalg.eig(a_mat, b_mat)
        # convert nan value into 0
        res[0][np.where(np.isnan(res[0]))] = 0.0
        # Only the positive eigenvalues are physical. We need to take care though of very small values
        # should an excited state approach ground state. Here the small values may be both negative or
        # positive. We should take just one of these pairs as zero. So to get the values we want we
        # sort the real parts and then take the upper half of the sorted values. Since we may now have
        # small values (positive or negative) take the absolute and then threshold zero.
        logger.debug('... {}'.format(res[0]))
        w = np.sort(np.real(res[0]))
        logger.debug('Sorted real parts {}'.format(w))
        w = np.abs(w[len(w) // 2:])
        w[w < 1e-06] = 0
        excitation_energies_gap = w
        return excitation_energies_gap

    @staticmethod
    def _build_single_hopping_operator(index, num_particles, num_orbitals, qubit_mapping,
                                       two_qubit_reduction, z2_symmetries):

        h1 = np.zeros((num_orbitals, num_orbitals), dtype=complex)
        h2 = np.zeros((num_orbitals, num_orbitals, num_orbitals, num_orbitals), dtype=complex)
        if len(index) == 2:
            i, j = index
            h1[i, j] = 4.0
        elif len(index) == 4:
            i, j, k, m = index
            h2[i, j, k, m] = 16.0
        fer_op = FermionicOperator(h1, h2)
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
        mu, nu, left_op, right_op_1, right_op_2 = params
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

        return mu, nu, q_mat_op, w_mat_op, m_mat_op, v_mat_op
