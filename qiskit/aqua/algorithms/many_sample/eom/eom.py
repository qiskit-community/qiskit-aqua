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
import copy
import os
import platform
import itertools
import functools
import psutil
import json
import sys

import numpy as np
from scipy import linalg
from qiskit import QuantumCircuit, compile as q_compile
from qiskit.tools import parallel_map
#from qiskit.quantum_info import Pauli
from qiskit.aqua import AquaError, Operator
from qiskit.aqua.utils import find_regs_by_name
#from qiskit.aqua.operator import construct_evaluation_circuit
from qiskit.chemistry.aqua_extensions.components.variational_forms.uccsd import UCCSD
from qiskit.chemistry.fermionic_operator import FermionicOperator

from qiskit.aqua import build_logging_config
from .a_matrix_tools import make_cal_circuits, generate_A_matrix, remove_measurement_errors_all
from scipy.stats import linregress

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter(build_logging_config(logging.INFO)['formatters']['f']['format'])
handler.setFormatter(formatter)
logger.addHandler(handler)

def generate_qobj_for_error_mitigation(circuits, backend, a_matrix_circuits=None, num_points=3, shots=1, **kwargs):

    num_circuits = len(circuits)
    qobj = q_compile(circuits, backend, shots=shots, **kwargs)
    temp1 = copy.deepcopy(qobj.experiments)
    for _ in range(num_points - 1):
        qobj.experiments.extend(copy.deepcopy(temp1))
        # qobj.experiments.extend(temp2)
    if len(qobj.experiments) // num_circuits != num_points:
        raise ValueError("Too many circuits")
    for j, experiment in enumerate(qobj.experiments):
        suffix = "_{}".format(j // num_circuits + 1) if j // num_circuits != 0 else None
        if suffix is None:
            continue
        else:
            experiment.header.name += suffix
            for inst in experiment.instructions:
                if inst.name in ['u1', 'u2', 'u3', 'cx']:
                    inst.name += suffix

    if a_matrix_circuits is not None:
        qobj_a = q_compile(a_matrix_circuits, backend, shots=shots, **kwargs)
        qobj_a.experiments += qobj.experiments
    else:
        qobj_a = qobj

    return qobj_a

def extrapolation(stretch, means, stds):

    # if len(stretch) == 1:
    #     return means[0], stds[0]
    #
    # stretch_matrix = np.zeros((len(stretch) - 1, len(stretch) - 1))
    # for i in range(stretch_matrix.shape[0]):
    #     for j in range(stretch_matrix.shape[1]):
    #         stretch_matrix[j, i] = stretch[i + 1] ** j
    #
    # current = np.zeros(len(stretch) - 1)
    # current[0] = 1
    # stretch_coeff = np.linalg.solve(stretch_matrix, current)
    #
    # avg_mitigated = np.dot(stretch_coeff, means[1:])
    # std_dev_mitigated = np.dot(np.absolute(stretch_coeff), stds[1:])

    slope, avg_mitigated, rvalue, pvalue, stderr = linregress(stretch, means)
    slope, std_dev_mitigated, rvalue, pvalue, stderr = linregress(stretch, stds)
    # logger.info("Before mitigation: ({:.6f}, {:.6f})".format(means[0], stds[0]))
    # logger.info("After  mitigation: ({:.6f}, {:.6f})".format(avg_mitigated, std_dev_mitigated))
    return avg_mitigated, std_dev_mitigated


def combine_dict(orig_dict, new_dict):
    # beware the orig_dict will be modified rather than using as a copy

    for k, v in new_dict.items():
        if k in orig_dict:
            orig_dict[k] += v
        else:
            orig_dict[k] = v


class EquationOfMotion:

    def __init__(self, operator, operator_mode='matrix',
                 num_orbitals=0, num_particles=0, qubit_mapping=None, two_qubit_reduction=False,
                 active_occupied=None, active_unoccupied=None,
                 is_eom_matrix_symmetric=True, se_list=None, de_list=None,
                 cliffords=None, sq_list=None, tapering_values=None, symmetries=None,
                 untapered_op=None, mitigate=False, stretch=[0.0, 1.0, 2.0], load_commutators=False):

        """Constructor.

        Args:
            operator (Operator): qubit operator
            operator_mode (str): operator mode, used for eval of operator
            num_orbitals (int):  total number of spin orbitals
            num_particles (int): total number of particles
            qubit_mapping (str): qubit mapping type
            two_qubit_reduction (bool): two qubit reduction is applied or not
            active_occupied (list): list of occupied orbitals to include, indices are
                                    0 to n where n is num particles // 2
            active_unoccupied (list): list of unoccupied orbitals to include, indices are
                                    0 to m where m is (num_orbitals - num particles) // 2
            is_eom_matrix_symmetric (bool): is EoM matrix symmetric
            se_list ([list]): single excitation list, overwrite the setting in active space
            de_list ([list]): double excitation list, overwrite the setting in active space
            cliffords ([Operator]): list of unitary Clifford transformation
            sq_list ([int]): position of the single-qubit operators that anticommute
                            with the cliffords
            tapering_values ([int]): array of +/- 1 used to select the subspace. Length
                                    has to be equal to the length of cliffords and sq_list
            symmetries ([Pauli]): represent the Z2 symmetries
            untapered_op (Operator): if the operator is tapered, we need untapered operator
                                    to build element of EoM matrix
        """
        self._operator = operator
        self._operator_mode = operator_mode

        self._num_orbitals = num_orbitals
        self._num_particles = num_particles
        self._qubit_mapping = qubit_mapping
        self._two_qubit_reduction = two_qubit_reduction
        self._active_occupied = active_occupied
        self._active_unoccupied = active_unoccupied

        self._mitigate = mitigate

        if se_list is None or de_list is None:
            se_list_default, de_list_default = UCCSD.compute_excitation_lists(self._num_particles, self._num_qubits,
                                                                          self._active_occupied, self._active_unoccupied)
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

        # if qubit operator is tapered.
        self._cliffords = cliffords
        self._sq_list = sq_list
        self._tapering_values = tapering_values
        self._symmetries = symmetries
        if self._cliffords is not None and self._sq_list is not None and \
                self._tapering_values is not None and self._symmetries is not None:
            self._qubit_tapering = True
        else:
            self._qubit_tapering = False
        self._untapered_op = untapered_op if untapered_op is not None else operator

        self._is_eom_matrix_symmetric = is_eom_matrix_symmetric
        self._num_processes = psutil.cpu_count(logical=False) if platform.system() != "Windows" else 1
        self._ret = {}
        self._stretch = stretch
        self._num_points = len(stretch)
        self._load_commutators = load_commutators

        self._exp_results = {}

    def calculate_excited_states(self, wave_fn, quantum_instance=None):
        """Calcuate energy gap of excited states from the reference state.

        Args:
            wave_fn (QuantumCircuit or numpy.ndarray): wavefunction of reference state
            quantum_instance (QuantumInstance): a quantum instance with configured settings

        Returns:
            list: energy gaps to the reference state
            dict: information of eom matrices

        Raises:
            ValueError: wrong setting for wave_fn and quantum_instance
        """
        if isinstance(wave_fn, QuantumCircuit):
            if quantum_instance is None:
                raise ValueError("quantum_instance is required when wavn_fn is a QuantumCircuit.")
            temp_quantum_instance = copy.deepcopy(quantum_instance)
            if temp_quantum_instance.is_statevector and temp_quantum_instance.noise_config == {}:
                initial_statevector = quantum_instance.execute(wave_fn).get_statevector(wave_fn)
                logger.info("Under noise-free and statevector simulation, "
                            "the wave_fn is reused and set in initial_statevector for faster simulation.")
                temp_quantum_instance.set_config(initial_statevector=initial_statevector)
                #q = find_regs_by_name(wave_fn, 'q')
                #wave_fn = QuantumCircuit(q)
                wave_fn = initial_statevector
        else:
            temp_quantum_instance = None

        # this is required to assure paulis mode is there regardless how you compute VQE
        # it might be slow if you calculate vqe through matrix mode and then convert it back to paulis
        self._operator.to_paulis()
        self._untapered_op.to_paulis()

        m_mat, v_mat, q_mat, w_mat,\
            m_std, v_std, q_std, w_std = self.build_eom_matrices(self._de_list + self._se_list,
                                                                 wave_fn, temp_quantum_instance)
        excitation_energies_gap = self.compute_excitation_energies(m_mat, v_mat, q_mat, w_mat)

        logger.info('Net excited state values (gap to reference state): {}'.format(excitation_energies_gap))

        eom_matrices = {}
        eom_matrices['m_mat'] = m_mat
        eom_matrices['v_mat'] = v_mat
        eom_matrices['q_mat'] = q_mat
        eom_matrices['w_mat'] = w_mat
        eom_matrices['m_mat_std'] = m_std
        eom_matrices['v_mat_std'] = v_std
        eom_matrices['q_mat_std'] = q_std
        eom_matrices['w_mat_std'] = w_std

        return excitation_energies_gap, eom_matrices, self._exp_results

    def build_eom_matrices(self, excitations_list, wave_fn, quantum_instance=None, mitigate = False):
        """
        Compute M, V, Q and W matrices.

        Args:
            excitations_list (list): single excitations list + double excitation list
            wave_fn (QuantumCircuit or numpy.ndarray): the circuit generated wave function for the ground state energy
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

        # build all hopping operators
        hopping_operators = {}
        type_of_commutivity = {}
        for idx in range(len(mus)):
            mu = mus[idx]
            nu = nus[idx]
            for excitations in [excitations_list[mu], excitations_list[nu], list(reversed(excitations_list[nu]))]:
                key = '_'.join([str(x) for x in excitations])
                if key not in hopping_operators:
                    hopping_operators[key], type_of_commutivity[key] = \
                        EquationOfMotion._build_hopping_operator(excitations, self._num_particles,
                                                                 self._num_orbitals, self._qubit_mapping,
                                                                 self._two_qubit_reduction, self._symmetries)

        # build all commutators
        def _build_all_commutators(available_hopping_ops):
            from .eom import EquationOfMotion

            if self._load_commutators:
                # load from disk
                logger.info("Loading commutator from disk.")
                for idx in range(len(mus)):
                    mu = mus[idx]
                    nu = nus[idx]
                    file_prefix = self._prefix + "_{}_{}".format(mu, nu)

                    if os.path.exists(file_prefix + "_q"):
                        with open(file_prefix + "_q") as f:
                            q_commutators[mu][nu] = Operator.load_from_dict(json.load(f))
                    else:
                        q_commutators[mu][nu] = None

                    if os.path.exists(file_prefix + "_w"):
                        with open(file_prefix + "_w") as f:
                            w_commutators[mu][nu] = Operator.load_from_dict(json.load(f))
                    else:
                        w_commutators[mu][nu] = None

                    if os.path.exists(file_prefix + "_m"):
                        with open(file_prefix + "_m") as f:
                            m_commutators[mu][nu] = Operator.load_from_dict(json.load(f))
                    else:
                        m_commutators[mu][nu] = None

                    if os.path.exists(file_prefix + "_v"):
                        with open(file_prefix + "_v") as f:
                            v_commutators[mu][nu] = Operator.load_from_dict(json.load(f))
                    else:
                        v_commutators[mu][nu] = None
            else:
                # compute it
                to_be_computed_list = []
                for idx in range(len(mus)):
                    mu = mus[idx]
                    nu = nus[idx]
                    left_op = available_hopping_ops.get('_'.join([str(x) for x in excitations_list[mu]]), None)
                    right_op_1 = available_hopping_ops.get('_'.join([str(x) for x in excitations_list[nu]]), None)
                    right_op_2 = available_hopping_ops.get('_'.join([str(x) for x in reversed(excitations_list[nu])]), None)
                    to_be_computed_list.append((mu, nu, left_op, right_op_1, right_op_2))

                results = parallel_map(EquationOfMotion._build_commutator_rountine,
                                       to_be_computed_list,
                                       task_args=(self._untapered_op, self._cliffords,
                                                  self._sq_list, self._tapering_values))
                for result in results:
                    mu, nu, q_mat_op, w_mat_op, m_mat_op, v_mat_op = result
                    q_commutators[mu][nu] = q_mat_op if q_mat_op is not None else q_commutators[mu][nu]
                    w_commutators[mu][nu] = w_mat_op if w_mat_op is not None else w_commutators[mu][nu]
                    m_commutators[mu][nu] = m_mat_op if m_mat_op is not None else m_commutators[mu][nu]
                    v_commutators[mu][nu] = v_mat_op if v_mat_op is not None else v_commutators[mu][nu]



        def _calculate_eom_elements(q_commutators, w_commutators,
                                    m_commutators, v_commutators):
            m_mat = np.zeros((size, size), dtype=complex)
            v_mat = np.zeros((size, size), dtype=complex)
            q_mat = np.zeros((size, size), dtype=complex)
            w_mat = np.zeros((size, size), dtype=complex)
            m_mat_std, v_mat_std, q_mat_std, w_mat_std = 0, 0, 0, 0
            means = {}
            stds = {}

            if mitigate:

                # collect all paulis
                paulis = {}
                paulis_lut = {}
                for idx in range(len(mus)):
                    mu = mus[idx]
                    nu = nus[idx]
                    for op in [q_commutators[mu][nu], w_commutators[mu][nu], m_commutators[mu][nu], v_commutators[mu][nu]]:
                        if op is None:
                            continue
                        if self._operator_mode == 'grouped_paulis':
                            op.to_grouped_paulis()
                            for p in op.grouped_paulis:
                                if p[0][1] not in paulis:
                                    paulis[p[0][1]] = (p[0], len(paulis))
                        else:
                            for p in op.paulis:
                                if p[1] not in paulis:
                                    paulis[p[1]] = (p, len(paulis))

                    if self._operator_mode == 'grouped_paulis':
                        paulis_lut["{}_{}_q".format(mu, nu)] = [paulis[p[0][1]][1] for p in q_commutators[mu][nu].grouped_paulis] if q_commutators[mu][nu] is not None else []
                        paulis_lut["{}_{}_w".format(mu, nu)] = [paulis[p[0][1]][1] for p in w_commutators[mu][nu].grouped_paulis] if w_commutators[mu][nu] is not None else []
                        paulis_lut["{}_{}_m".format(mu, nu)] = [paulis[p[0][1]][1] for p in m_commutators[mu][nu].grouped_paulis] if m_commutators[mu][nu] is not None else []
                        paulis_lut["{}_{}_v".format(mu, nu)] = [paulis[p[0][1]][1] for p in v_commutators[mu][nu].grouped_paulis] if v_commutators[mu][nu] is not None else []
                    else:
                        paulis_lut["{}_{}_q".format(mu, nu)] = [paulis[p[1]][1] for p in
                                                                q_commutators[mu][nu].paulis] if q_commutators[mu][
                                                                                                     nu] is not None else []
                        paulis_lut["{}_{}_w".format(mu, nu)] = [paulis[p[1]][1] for p in
                                                                w_commutators[mu][nu].paulis] if w_commutators[mu][
                                                                                                     nu] is not None else []
                        paulis_lut["{}_{}_m".format(mu, nu)] = [paulis[p[1]][1] for p in
                                                                m_commutators[mu][nu].paulis] if m_commutators[mu][
                                                                                                     nu] is not None else []
                        paulis_lut["{}_{}_v".format(mu, nu)] = [paulis[p[1]][1] for p in
                                                                v_commutators[mu][nu].paulis] if v_commutators[mu][
                                                                                                     nu] is not None else []

                temp_paulis = [[1.0, p[0][1]] for p in paulis.values()]
                temp_op = Operator(paulis=temp_paulis)
                circuits = temp_op.construct_evaluation_circuit('paulis', wave_fn, quantum_instance.backend)
                logger.info("Total number of circuits: {}".format(len(circuits)))

                q = find_regs_by_name(circuits[0], 'q')
                c = find_regs_by_name(circuits[0], 'c', qreg=False)
                a_matrix_circuits = make_cal_circuits(list(range(temp_op.num_qubits)), q, c)
                qobj = generate_qobj_for_error_mitigation(circuits, quantum_instance.backend, a_matrix_circuits,
                                                          num_points=self._num_points, shots=quantum_instance.run_config['shots'])

                job = quantum_instance.backend.run(qobj)
                result = job.result(timeout=None)

                # perform error mitigation over measurements
                a_matrix_counts = {qc.name: result.get_counts(qc) for qc in a_matrix_circuits}
                # build a matrix
                a_matrix = generate_A_matrix(a_matrix_counts, a_matrix_circuits, list(range(op.num_qubits)),
                                             shots=quantum_instance.run_config['shots'])

                pauli_counts = {}  # paulis-counts pair
                for idx in range(self._num_points):
                    suffix = "_{}".format(idx + 1) if idx != 0 else None
                    sub_circuits = copy.deepcopy(circuits)
                    for qc in sub_circuits:
                        if suffix:
                            qc.name += suffix
                        pauli_counts[qc.name] = result.get_counts(qc)


                for idx in range(len(mus)):
                    mu = mus[idx]
                    nu = nus[idx]
                    means["{}_{}_q".format(mu, nu)] = np.zeros((self._num_points))
                    stds["{}_{}_q".format(mu, nu)] = np.zeros((self._num_points))

                    means["{}_{}_w".format(mu, nu)] = np.zeros((self._num_points))
                    stds["{}_{}_w".format(mu, nu)] = np.zeros((self._num_points))

                    means["{}_{}_m".format(mu, nu)] = np.zeros((self._num_points))
                    stds["{}_{}_m".format(mu, nu)] = np.zeros((self._num_points))

                    means["{}_{}_v".format(mu, nu)] = np.zeros((self._num_points))
                    stds["{}_{}_v".format(mu, nu)] = np.zeros((self._num_points))

                # calibrate counts based on a matrix and calculate the expectation of different pulse lengths
                for m in range(self._num_points):
                    suffix = "_{}".format(m + 1) if m != 0 else None
                    sub_circuits = copy.deepcopy(circuits)
                    for qc_idx in range(len(sub_circuits)):
                        if suffix:
                            sub_circuits[qc_idx].name += suffix
                    cal_result = remove_measurement_errors_all(pauli_counts, sub_circuits,
                                                               list(range(temp_op.num_qubits)),
                                                               a_matrix, shots=quantum_instance.run_config['shots'], method=1,
                                                               data_format='counts')
                    # evaluate results
                    for idx in range(len(mus)):
                        mu = mus[idx]
                        nu = nus[idx]

                        def _get_result(op, circuits, cat):
                            if circuits is not None and circuits != [] and op is not None and not op.is_empty():
                                sub_circuits = copy.deepcopy(circuits)
                                for qc in sub_circuits:
                                    if suffix:
                                        qc.name += suffix
                                mean, std = op.evaluate_with_result(self._operator_mode, sub_circuits,
                                                                    quantum_instance.backend, cal_result)

                                means["{}_{}_{}".format(mu, nu, cat)][m] = mean.real
                                stds["{}_{}_{}".format(mu, nu, cat)][m] = std.real
                            else:
                                means["{}_{}_{}".format(mu, nu, cat)][m] = 0.0
                                stds["{}_{}_{}".format(mu, nu, cat)][m] = 0.0


                        _get_result(q_commutators[mu][nu], [circuits[c_idx] for c_idx in paulis_lut["{}_{}_q".format(mu, nu)]], "q")
                        _get_result(w_commutators[mu][nu], [circuits[c_idx] for c_idx in paulis_lut["{}_{}_w".format(mu, nu)]], "w")
                        _get_result(m_commutators[mu][nu], [circuits[c_idx] for c_idx in paulis_lut["{}_{}_m".format(mu, nu)]], "m")
                        _get_result(v_commutators[mu][nu], [circuits[c_idx] for c_idx in paulis_lut["{}_{}_v".format(mu, nu)]], "v")

                # get all expectation, start error mitigation
                logger.info("EOM elements: (the last one is the mitigated value)")
                for idx in range(len(mus)):
                    mu = mus[idx]
                    nu = nus[idx]

                    new_mean, new_std = extrapolation(self._stretch, means["{}_{}_q".format(mu, nu)], stds["{}_{}_q".format(mu, nu)])
                    means["{}_{}_q".format(mu, nu)] = np.append(means["{}_{}_q".format(mu, nu)], new_mean)
                    stds["{}_{}_q".format(mu, nu)] = np.append(stds["{}_{}_q".format(mu, nu)], new_std)
                    logger.info("Q({}, {}):\t{}".format(mu, nu, means["{}_{}_q".format(mu, nu)]))
                    q_mat[mu][nu] = new_mean if new_mean != 0.0 else q_mat[mu][nu]
                    q_mat_std += new_std

                    new_mean, new_std = extrapolation(self._stretch, means["{}_{}_w".format(mu, nu)], stds["{}_{}_w".format(mu, nu)])
                    means["{}_{}_w".format(mu, nu)] = np.append(means["{}_{}_w".format(mu, nu)], new_mean)
                    stds["{}_{}_w".format(mu, nu)] = np.append(stds["{}_{}_w".format(mu, nu)], new_std)
                    logger.info("W({}, {}):\t{}".format(mu, nu, means["{}_{}_w".format(mu, nu)]))
                    w_mat[mu][nu] = new_mean if new_mean != 0.0 else w_mat[mu][nu]
                    w_mat_std += new_std

                    new_mean, new_std = extrapolation(self._stretch, means["{}_{}_m".format(mu, nu)], stds["{}_{}_m".format(mu, nu)])
                    means["{}_{}_m".format(mu, nu)] = np.append(means["{}_{}_m".format(mu, nu)], new_mean)
                    stds["{}_{}_m".format(mu, nu)] = np.append(stds["{}_{}_m".format(mu, nu)], new_std)
                    logger.info("M({}, {}):\t{}".format(mu, nu, means["{}_{}_m".format(mu, nu)]))
                    m_mat[mu][nu] = new_mean if new_mean != 0.0 else m_mat[mu][nu]
                    m_mat_std += new_std

                    new_mean, new_std = extrapolation(self._stretch, means["{}_{}_v".format(mu, nu)], stds["{}_{}_v".format(mu, nu)])
                    means["{}_{}_v".format(mu, nu)] = np.append(means["{}_{}_v".format(mu, nu)], new_mean)
                    stds["{}_{}_v".format(mu, nu)] = np.append(stds["{}_{}_v".format(mu, nu)], new_std)
                    logger.info("V({}, {}):\t{}".format(mu, nu, means["{}_{}_v".format(mu, nu)]))
                    v_mat[mu][nu] = new_mean if new_mean != 0.0 else v_mat[mu][nu]
                    v_mat_std += new_std

            else:
                for idx in range(len(mus)):
                    mu = mus[idx]
                    nu = nus[idx]
                    #print(m_commutators[mu][nu].paulis)
                    q_mat_mu_nu = q_commutators[mu][nu]._eval_directly(
                        wave_fn) if q_commutators[mu][nu] is not None else 0.0
                    w_mat_mu_nu = w_commutators[mu][nu]._eval_directly(
                        wave_fn) if w_commutators[mu][nu] is not None else 0.0
                    m_mat_mu_nu = m_commutators[mu][nu]._eval_directly(
                        wave_fn) if m_commutators[mu][nu] is not None else 0.0
                    v_mat_mu_nu = v_commutators[mu][nu]._eval_directly(
                        wave_fn) if v_commutators[mu][nu] is not None else 0.0
                    q_mat[mu][nu] = q_mat_mu_nu if q_mat_mu_nu != 0.0 else q_mat[mu][nu]
                    w_mat[mu][nu] = w_mat_mu_nu if w_mat_mu_nu != 0.0 else w_mat[mu][nu]
                    m_mat[mu][nu] = m_mat_mu_nu if m_mat_mu_nu != 0.0 else m_mat[mu][nu]
                    v_mat[mu][nu] = v_mat_mu_nu if v_mat_mu_nu != 0.0 else v_mat[mu][nu]

            if self._is_eom_matrix_symmetric:
                q_mat = q_mat + q_mat.T - np.identity(q_mat.shape[0]) * q_mat
                w_mat = w_mat + w_mat.T - np.identity(w_mat.shape[0]) * w_mat
                m_mat = m_mat + m_mat.T - np.identity(m_mat.shape[0]) * m_mat
                v_mat = v_mat + v_mat.T - np.identity(v_mat.shape[0]) * v_mat

            return q_mat, w_mat, m_mat, v_mat, q_mat_std, w_mat_std, m_mat_std, v_mat_std, means, stds

        # start to calculate eom matrix
        available_entry = 0
        if self._qubit_tapering:
            for targeted_tapering_values in itertools.product([1, -1], repeat=len(self._symmetries)):
                logger.info("In sector: ({})".format(','.join([str(x) for x in targeted_tapering_values])))
                # remove the excited operators which are not suitable for the sector
                available_hopping_ops = {}
                targeted_sector = (np.asarray(targeted_tapering_values) == 1)
                for key, value in type_of_commutivity.items():
                    value = np.asarray(value)
                    if np.all(value == targeted_sector):
                        available_hopping_ops[key] = hopping_operators[key]
                _build_all_commutators(available_hopping_ops)
                print("=========  ({})  ========".format(','.join([str(x) for x in targeted_tapering_values])))
                for k, v in available_hopping_ops.items():
                    print("{}:\n{}".format(k, v.print_operators()))
                available_entry += len(available_hopping_ops) * len(available_hopping_ops)

        else:
            available_hopping_ops = hopping_operators
            _build_all_commutators(available_hopping_ops)
            available_entry = len(available_hopping_ops) * len(available_hopping_ops)

        q_mat, w_mat, m_mat, v_mat, q_mat_std, w_mat_std, m_mat_std, v_mat_std, means, stds = _calculate_eom_elements(
            q_commutators, w_commutators, m_commutators, v_commutators)

        for key, value in means.items():
            self._exp_results['eom_mean_{}'.format(key)] = value
        for key, value in stds.items():
            self._exp_results['eom_std_{}'.format(key)] = value

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
    def _build_hopping_operator(index, num_particles, num_orbitals, qubit_mapping,
                                two_qubit_reduction, symmetries=None):

        def check_commutativity(op_1, op_2, anti=False):
            com = op_1 * op_2 - op_2 * op_1 if not anti else op_1 * op_2 + op_2 * op_1
            com.zeros_coeff_elimination()
            return True if com.is_empty() else False

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
            qubit_op = qubit_op.two_qubit_reduced_operator(num_particles)

        type_of_commutivity = []
        if symmetries is not None:
            for symmetry in symmetries:
                symmetry_op = Operator(paulis=[[1.0, symmetry]])
                commuting = check_commutativity(symmetry_op, qubit_op)
                anticommuting = check_commutativity(symmetry_op, qubit_op, anti=True)
                if commuting != anticommuting:  # only one of them is True
                    if commuting:
                        type_of_commutivity.append(True)
                    elif anticommuting:
                        type_of_commutivity.append(False)
                else:
                    raise AquaError(
                        "Symmetry {} is nor commute neither anti-commute to exciting operator.".format(symmetry.to_label()))

        return qubit_op, type_of_commutivity

    @staticmethod
    def _build_commutator_rountine(params, operator,
                                   cliffords, sq_list, tapering_values):
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
                logger.info('Building commutator at ({}, {}).'.format(mu, nu))
                if right_op_1 is not None:
                    q_mat_op = EquationOfMotion.commutator(left_op, operator, right_op_1, cliffords=cliffords,
                                                           sq_list=sq_list, tapering_values=tapering_values)
                    w_mat_op = EquationOfMotion.commutator(left_op, right_op_1, cliffords=cliffords,
                                                           sq_list=sq_list, tapering_values=tapering_values)
                    q_mat_op = None if q_mat_op.is_empty() else q_mat_op
                    w_mat_op = None if w_mat_op.is_empty() else w_mat_op
                else:
                    q_mat_op = None
                    w_mat_op = None

                if right_op_2 is not None:
                    m_mat_op = EquationOfMotion.commutator(left_op, operator, right_op_2, cliffords=cliffords,
                                                           sq_list=sq_list, tapering_values=tapering_values)
                    v_mat_op = EquationOfMotion.commutator(left_op, right_op_2, cliffords=cliffords,
                                                           sq_list=sq_list, tapering_values=tapering_values)
                    m_mat_op = None if m_mat_op.is_empty() else m_mat_op
                    v_mat_op = None if v_mat_op.is_empty() else v_mat_op
                else:
                    m_mat_op = None
                    v_mat_op = None

        return mu, nu, q_mat_op, w_mat_op, m_mat_op, v_mat_op

    def compute_excitation_energies(self, m_mat, v_mat, q_mat, w_mat):
        """
        Diagonalizing M, V, Q, W matrices for excitation energies.

        Args:
            m_mat (numpy.ndarray): M
            v_mat (numpy.ndarray): V
            q_mat (numpy.ndarray): Q
            w_mat (numpy.ndarray): W
        Returns:
            numpy.ndarray: 1-D vector stores all excited energy gap to reference state
        """
        logger.debug('Diagonalizing eom matrices for excited states...')
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
        # the computed values are the delta from the ground state energy to the excited states
        # excitation_energies += ground_state_energy
        return excitation_energies_gap

    @staticmethod
    def commutator(op_a, op_b, op_c=None, cliffords=None, sq_list=None, tapering_values=None):
        """
        Compute commutator of op_a and op_b or the symmetric double commutator of op_a, op_b and op_c.

        See McWeeny chapter 13.6 Equation of motion methods (page 479)

        res = 0.5 * (2*A*B*C + 2*C*B*A - B*A*C - C*A*B - A*C*B - B*C*A)

        Args:
            op_a: operator a
            op_b: operator b
            op_c: operator c

        Returns:
            Operator: the commutator

        Note:
            For the final chop, the original codes only contain the paulis with real coefficient.
        """
        if op_c is None:
            op_ab = op_a * op_b
            op_ba = op_b * op_a
            res = op_ab - op_ba
        else:
            op_ab = op_a * op_b
            op_ba = op_b * op_a
            op_ac = op_a * op_c
            op_ca = op_c * op_a

            op_abc = op_ab * op_c
            op_cba = op_c * op_ba
            op_bac = op_ba * op_c
            op_cab = op_c * op_ab
            op_acb = op_ac * op_b
            op_bca = op_b * op_ca

            tmp = (op_bac + op_cab + op_acb + op_bca)
            tmp.scaling_coeff(0.5)
            res = op_abc + op_cba - tmp

        if cliffords is not None and sq_list is not None and tapering_values is not None and not res.is_empty():
            res = Operator.qubit_tapering(res, cliffords, sq_list, tapering_values)

        res.chop(1e-12)
        res.zeros_coeff_elimination()
        return res
