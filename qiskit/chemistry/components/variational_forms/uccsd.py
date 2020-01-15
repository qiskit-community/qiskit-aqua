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
This trial wavefunction is a Unitary Coupled-Cluster Single and Double excitations
variational form.
For more information, see https://arxiv.org/abs/1805.04340
"""

from typing import Optional, Union, List
import logging
import sys

import numpy as np
from qiskit.aqua.utils.validation import validate_min, validate_in_set
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.aqua import aqua_globals
from qiskit.aqua.components.initial_states import InitialState
from qiskit.aqua.operators import WeightedPauliOperator, Z2Symmetries
from qiskit.aqua.components.variational_forms import VariationalForm
from qiskit.chemistry.fermionic_operator import FermionicOperator

logger = logging.getLogger(__name__)


class UCCSD(VariationalForm):
    """
        This trial wavefunction is a Unitary Coupled-Cluster Single and Double excitations
        variational form.
        For more information, see https://arxiv.org/abs/1805.04340
    """

    def __init__(self, num_qubits: int,
                 depth: int,
                 num_orbitals: int,
                 num_particles: Union[List[int], int],
                 active_occupied: Optional[List[int]] = None,
                 active_unoccupied: Optional[List[int]] = None,
                 initial_state: Optional[InitialState] = None,
                 qubit_mapping: str = 'parity',
                 two_qubit_reduction: bool = True,
                 num_time_slices: int = 1,
                 shallow_circuit_concat: bool = True,
                 z2_symmetries: Optional[Z2Symmetries] = None) -> None:
        """Constructor.

        Args:
            num_qubits: number of qubits, has a min. value of 1.
            depth: number of replica of basic module, has a min. value of 1.
            num_orbitals: number of spin orbitals, has a min. value of 1.
            num_particles: number of particles, if it is a list,
                            the first number is alpha and the second number if beta.
            active_occupied: list of occupied orbitals to consider as active space
            active_unoccupied: list of unoccupied orbitals to consider as active space
            initial_state: An initial state object.
            qubit_mapping: qubit mapping type.
            two_qubit_reduction: two qubit reduction is applied or not.
            num_time_slices: parameters for dynamics, has a min. value of 1.
            shallow_circuit_concat: indicate whether to use shallow (cheap) mode for
                                           circuit concatenation
            z2_symmetries: represent the Z2 symmetries, including symmetries,
                            sq_paulis, sq_list, tapering_values, and cliffords
         Raises:
             ValueError: Computed qubits do not match actual value
        """
        validate_min('num_qubits', num_qubits, 1)
        validate_min('depth', depth, 1)
        validate_min('num_orbitals', num_orbitals, 1)
        if isinstance(num_particles, list) and len(num_particles) != 2:
            raise ValueError('Num particles value {}. Number of values allowed is 2'.format(
                num_particles))
        validate_in_set('qubit_mapping', qubit_mapping,
                        {'jordan_wigner', 'parity', 'bravyi_kitaev'})
        validate_min('num_time_slices', num_time_slices, 1)
        super().__init__()

        self._z2_symmetries = Z2Symmetries([], [], [], []) \
            if z2_symmetries is None else z2_symmetries

        self._num_qubits = num_orbitals if not two_qubit_reduction else num_orbitals - 2
        self._num_qubits = self._num_qubits if self._z2_symmetries.is_empty() \
            else self._num_qubits - len(self._z2_symmetries.sq_list)
        if self._num_qubits != num_qubits:
            raise ValueError('Computed num qubits {} does not match actual {}'
                             .format(self._num_qubits, num_qubits))
        self._depth = depth
        self._num_orbitals = num_orbitals
        if isinstance(num_particles, list):
            self._num_alpha = num_particles[0]
            self._num_beta = num_particles[1]
        else:
            logger.info("We assume that the number of alphas and betas are the same.")
            self._num_alpha = num_particles // 2
            self._num_beta = num_particles // 2

        self._num_particles = [self._num_alpha, self._num_beta]

        if sum(self._num_particles) > self._num_orbitals:
            raise ValueError('# of particles must be less than or equal to # of orbitals.')

        self._initial_state = initial_state
        self._qubit_mapping = qubit_mapping
        self._two_qubit_reduction = two_qubit_reduction
        self._num_time_slices = num_time_slices
        self._shallow_circuit_concat = shallow_circuit_concat

        self._single_excitations, self._double_excitations = \
            UCCSD.compute_excitation_lists([self._num_alpha, self._num_beta], self._num_orbitals,
                                           active_occupied, active_unoccupied)

        self._hopping_ops, self._num_parameters = self._build_hopping_operators()
        self._excitation_pool = None
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

        self._logging_construct_circuit = True
        self._support_parameterized_circuit = True

    @property
    def single_excitations(self):
        """
        Getter of single excitation list
        Returns:
            list[list[int]]: single excitation list
        """
        return self._single_excitations

    @property
    def double_excitations(self):
        """
        Getter of double excitation list
        Returns:
            list[list[int]]: double excitation list
        """
        return self._double_excitations

    @property
    def excitation_pool(self):
        """
        Getter of full list of available excitations (called the pool)
        Returns:
            list[WeightedPauliOperator]: excitation pool
        """
        return self._excitation_pool

    def _build_hopping_operators(self):
        if logger.isEnabledFor(logging.DEBUG):
            TextProgressBar(sys.stderr)

        results = parallel_map(UCCSD._build_hopping_operator,
                               self._single_excitations + self._double_excitations,
                               task_args=(self._num_orbitals,
                                          self._num_particles, self._qubit_mapping,
                                          self._two_qubit_reduction, self._z2_symmetries),
                               num_processes=aqua_globals.num_processes)
        hopping_ops = []
        s_e_list = []
        d_e_list = []
        for op, index in results:
            if op is not None and not op.is_empty():
                hopping_ops.append(op)
                if len(index) == 2:  # for double excitation
                    s_e_list.append(index)
                else:  # for double excitation
                    d_e_list.append(index)

        self._single_excitations = s_e_list
        self._double_excitations = d_e_list

        num_parameters = len(hopping_ops) * self._depth
        return hopping_ops, num_parameters

    @staticmethod
    def _build_hopping_operator(index, num_orbitals, num_particles, qubit_mapping,
                                two_qubit_reduction, z2_symmetries):

        h_1 = np.zeros((num_orbitals, num_orbitals))
        h_2 = np.zeros((num_orbitals, num_orbitals, num_orbitals, num_orbitals))
        if len(index) == 2:
            i, j = index
            h_1[i, j] = 1.0
            h_1[j, i] = -1.0
        elif len(index) == 4:
            i, j, k, m = index
            h_2[i, j, k, m] = 1.0
            h_2[m, k, j, i] = -1.0

        dummpy_fer_op = FermionicOperator(h1=h_1, h2=h_2)
        qubit_op = dummpy_fer_op.mapping(qubit_mapping)
        if two_qubit_reduction:
            qubit_op = Z2Symmetries.two_qubit_reduction(qubit_op, num_particles)

        if not z2_symmetries.is_empty():
            symm_commuting = True
            for symmetry in z2_symmetries.symmetries:
                symmetry_op = WeightedPauliOperator(paulis=[[1.0, symmetry]])
                symm_commuting = qubit_op.commute_with(symmetry_op)
                if not symm_commuting:
                    break
            qubit_op = z2_symmetries.taper(qubit_op) if symm_commuting else None

        if qubit_op is None:
            logger.debug('Excitation (%s) is skipped since it is not commuted '
                         'with symmetries', ','.join([str(x) for x in index]))
        return qubit_op, index

    def manage_hopping_operators(self):
        """
        Triggers the adaptive behavior of this UCCSD instance.

        This function is used by the Adaptive VQE algorithm. It stores the full list of available
        hopping operators in a so called "excitation pool" and clears the previous list to be empty.
        Furthermore, the depth is asserted to be 1 which is required by the Adaptive VQE algorithm.
        """
        # store full list of excitations as pool
        self._excitation_pool = self._hopping_ops.copy()

        # check depth parameter
        if self._depth != 1:
            logger.warning('The depth of the variational form was not 1 but %i which does not work \
                    in the adaptive VQE algorithm. Thus, it has been reset to 1.')
            self._depth = 1

        # reset internal excitation list to be empty
        self._hopping_ops = []
        self._num_parameters = len(self._hopping_ops) * self._depth
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

    def push_hopping_operator(self, excitation):
        """
        Pushes a new hopping operator.

        Args:
            excitation (WeightedPauliOperator): the new hopping operator to be added
        """
        self._hopping_ops.append(excitation)
        self._num_parameters = len(self._hopping_ops) * self._depth
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

    def pop_hopping_operator(self):
        """
        Pops the hopping operator that was added last.
        """
        self._hopping_ops.pop()
        self._num_parameters = len(self._hopping_ops) * self._depth
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

    def construct_circuit(self, parameters, q=None):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (Union(numpy.ndarray, list[Parameter], ParameterVector)): circuit parameters
            q (QuantumRegister, optional): Quantum Register for the circuit.

        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`

        Raises:
            ValueError: the number of parameters is incorrect.
        """
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        if q is None:
            q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        if logger.isEnabledFor(logging.DEBUG) and self._logging_construct_circuit:
            logger.debug("Evolving hopping operators:")
            TextProgressBar(sys.stderr)
            self._logging_construct_circuit = False

        num_excitations = len(self._hopping_ops)
        results = parallel_map(UCCSD._construct_circuit_for_one_excited_operator,
                               [(self._hopping_ops[index % num_excitations], parameters[index])
                                for index in range(self._depth * num_excitations)],
                               task_args=(q, self._num_time_slices),
                               num_processes=aqua_globals.num_processes)
        for qc in results:
            if self._shallow_circuit_concat:
                circuit.data += qc.data
            else:
                circuit += qc

        return circuit

    @staticmethod
    def _construct_circuit_for_one_excited_operator(qubit_op_and_param, qr, num_time_slices):
        qubit_op, param = qubit_op_and_param
        # TODO: need to put -1j in the coeff of pauli since the Parameter
        # does not support complex number, but it can be removed if Parameter supports complex
        qubit_op = qubit_op * -1j
        qc = qubit_op.evolve(state_in=None, evo_time=param,
                             num_time_slices=num_time_slices,
                             quantum_registers=qr)
        return qc

    @property
    def preferred_init_points(self):
        """Getter of preferred initial points based on the given initial state."""
        if self._initial_state is None:
            return None
        else:
            bitstr = self._initial_state.bitstr
            if bitstr is not None:
                return np.zeros(self._num_parameters, dtype=np.float)
            else:
                return None

    @staticmethod
    def compute_excitation_lists(num_particles, num_orbitals, active_occ_list=None,
                                 active_unocc_list=None, same_spin_doubles=True):
        """
        Computes single and double excitation lists

        Args:
            num_particles (Union(list, int)): number of particles, if it is a tuple,
                                        the first number is alpha and the second number if beta.
            num_orbitals (int): Total number of spin orbitals
            active_occ_list (list): List of occupied orbitals to include, indices are
                             0 to n where n is max(num_alpha, num_beta)
            active_unocc_list (list): List of unoccupied orbitals to include, indices are
                               0 to m where m is num_orbitals // 2 - min(num_alpha, num_beta)
            same_spin_doubles (bool): True to include alpha,alpha and beta,beta double excitations
                               as well as alpha,beta pairings. False includes only alpha,beta

        Returns:
            list: Single excitation list
            list: Double excitation list

        Raises:
            ValueError: invalid setting of number of particles
            ValueError: invalid setting of number of orbitals
        """

        if isinstance(num_particles, list):
            num_alpha = num_particles[0]
            num_beta = num_particles[1]
        else:
            logger.info("We assume that the number of alphas and betas are the same.")
            num_alpha = num_particles // 2
            num_beta = num_particles // 2

        num_particles = num_alpha + num_beta

        if num_particles < 2:
            raise ValueError('Invalid number of particles {}'.format(num_particles))
        if num_orbitals < 4 or num_orbitals % 2 != 0:
            raise ValueError('Invalid number of orbitals {}'.format(num_orbitals))
        if num_orbitals <= num_particles:
            raise ValueError('No unoccupied orbitals')

        # convert the user-defined active space for alpha and beta respectively
        active_occ_list_alpha = []
        active_occ_list_beta = []
        active_unocc_list_alpha = []
        active_unocc_list_beta = []

        if active_occ_list is not None:
            active_occ_list = \
                [i if i >= 0 else i + max(num_alpha, num_beta) for i in active_occ_list]
            for i in active_occ_list:
                if i < num_alpha:
                    active_occ_list_alpha.append(i)
                else:
                    raise ValueError(
                        'Invalid index {} in active active_occ_list {}'.format(i, active_occ_list))
                if i < num_beta:
                    active_occ_list_beta.append(i)
                else:
                    raise ValueError(
                        'Invalid index {} in active active_occ_list {}'.format(i, active_occ_list))
        else:
            active_occ_list_alpha = list(range(0, num_alpha))
            active_occ_list_beta = list(range(0, num_beta))

        if active_unocc_list is not None:
            active_unocc_list = [i + min(num_alpha, num_beta) if i >=
                                 0 else i + num_orbitals // 2 for i in active_unocc_list]
            for i in active_unocc_list:
                if i >= num_alpha:
                    active_unocc_list_alpha.append(i)
                else:
                    raise ValueError('Invalid index {} in active active_unocc_list {}'
                                     .format(i, active_unocc_list))
                if i >= num_beta:
                    active_unocc_list_beta.append(i)
                else:
                    raise ValueError('Invalid index {} in active active_unocc_list {}'
                                     .format(i, active_unocc_list))
        else:
            active_unocc_list_alpha = list(range(num_alpha, num_orbitals // 2))
            active_unocc_list_beta = list(range(num_beta, num_orbitals // 2))

        logger.debug('active_occ_list_alpha %s', active_occ_list_alpha)
        logger.debug('active_unocc_list_alpha %s', active_unocc_list_alpha)

        logger.debug('active_occ_list_beta %s', active_occ_list_beta)
        logger.debug('active_unocc_list_beta %s', active_unocc_list_beta)

        single_excitations = []
        double_excitations = []

        beta_idx = num_orbitals // 2
        for occ_alpha in active_occ_list_alpha:
            for unocc_alpha in active_unocc_list_alpha:
                single_excitations.append([occ_alpha, unocc_alpha])

        for occ_beta in [i + beta_idx for i in active_occ_list_beta]:
            for unocc_beta in [i + beta_idx for i in active_unocc_list_beta]:
                single_excitations.append([occ_beta, unocc_beta])

        for occ_alpha in active_occ_list_alpha:
            for unocc_alpha in active_unocc_list_alpha:
                for occ_beta in [i + beta_idx for i in active_occ_list_beta]:
                    for unocc_beta in [i + beta_idx for i in active_unocc_list_beta]:
                        double_excitations.append([occ_alpha, unocc_alpha, occ_beta, unocc_beta])

        if same_spin_doubles and \
                len(active_occ_list_alpha) > 1 and len(active_unocc_list_alpha) > 1:
            for i, occ_alpha in enumerate(active_occ_list_alpha[:-1]):
                for j, unocc_alpha in enumerate(active_unocc_list_alpha[:-1]):
                    for occ_alpha_1 in active_occ_list_alpha[i + 1:]:
                        for unocc_alpha_1 in active_unocc_list_alpha[j + 1:]:
                            double_excitations.append([occ_alpha, unocc_alpha,
                                                       occ_alpha_1, unocc_alpha_1])

            up_active_occ_list = [i + beta_idx for i in active_occ_list_beta]
            up_active_unocc_list = [i + beta_idx for i in active_unocc_list_beta]
            for i, occ_beta in enumerate(up_active_occ_list[:-1]):
                for j, unocc_beta in enumerate(up_active_unocc_list[:-1]):
                    for occ_beta_1 in up_active_occ_list[i + 1:]:
                        for unocc_beta_1 in up_active_unocc_list[j + 1:]:
                            double_excitations.append([occ_beta, unocc_beta,
                                                       occ_beta_1, unocc_beta_1])

        logger.debug('single_excitations (%s) %s', len(single_excitations), single_excitations)
        logger.debug('double_excitations (%s) %s', len(double_excitations), double_excitations)

        return single_excitations, double_excitations
