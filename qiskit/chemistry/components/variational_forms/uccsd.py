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
Also, for more information on the tapering see: https://arxiv.org/abs/1701.08213
And for singlet q-UCCD (full) and paired q-UCCD see: https://arxiv.org/abs/1911.10864
"""

from typing import Optional, Union, List
import logging
import sys
import collections
import copy

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
    And for the singlet q-UCCD (full) and paired q-UCCD) see: https://arxiv.org/abs/1911.10864
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
                 z2_symmetries: Optional[Z2Symmetries] = None,
                 method_singles: str = 'both',
                 method_doubles: str = 'ucc',
                 excitation_type: str = 'sd',
                 same_spin_doubles: bool = True,
                 force_no_tap_excitation: bool = False) -> None:
        """Constructor.

        Args:
            num_qubits: number of qubits, has a min. value of 1.
            depth: number of replica of basic module, has a min. value of 1.
            num_orbitals: number of spin orbitals, has a min. value of 1.
            num_particles: number of particles, if it is a list,
                            the first number is alpha and the second number if beta.
            active_occupied: list of occupied orbitals to consider as active space.
            active_unoccupied: list of unoccupied orbitals to consider as active space.
            initial_state: An initial state object.
            qubit_mapping: qubit mapping type.
            two_qubit_reduction: two qubit reduction is applied or not.
            num_time_slices: parameters for dynamics, has a min. value of 1.
            shallow_circuit_concat: indicate whether to use shallow (cheap) mode for
                                           circuit concatenation.
            z2_symmetries: represent the Z2 symmetries, including symmetries,
                            sq_paulis, sq_list, tapering_values, and cliffords.
            method_singles: specify the single excitation considered. 'alpha', 'beta',
                                'both' only alpha or beta spin-orbital single excitations or
                                both (all of them)
            method_doubles: specify the single excitation considered. 'ucc' (conventional
                                ucc), succ (singlet ucc), succ_full (singlet ucc full)
            excitation_type: specify the excitation type 'sd', 's', 'd' respectively
                                for single and double, only single, only double excitations.
            same_spin_doubles: enable double excitations of the same spin.
            force_no_tap_excitation: keep all the excitation regardless if tapering is used.

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
        validate_in_set('method_singles', method_singles, {'both', 'alpha', 'beta'})
        validate_in_set('method_doubles', method_doubles, {'ucc', 'pucc', 'succ', 'succ_full'})
        validate_in_set('excitation_type', excitation_type, {'sd', 's', 'd'})
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

        # advanced parameters
        self._method_singles = method_singles
        self._method_doubles = method_doubles
        self._excitation_type = excitation_type
        self.same_spin_doubles = same_spin_doubles
        self._force_no_tap_excitation = force_no_tap_excitation

        self._single_excitations, self._double_excitations = \
            UCCSD.compute_excitation_lists([self._num_alpha, self._num_beta], self._num_orbitals,
                                           active_occupied, active_unoccupied,
                                           same_spin_doubles=self.same_spin_doubles,
                                           method_singles=self._method_singles,
                                           method_doubles=self._method_doubles,
                                           excitation_type=self._excitation_type,)

        self._hopping_ops, self._num_parameters = self._build_hopping_operators()
        self._excitation_pool = None
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

        self._logging_construct_circuit = True
        self._support_parameterized_circuit = True

        self.uccd_singlet = False
        if self._method_doubles == 'succ_full':
            self.uccd_singlet = True
            self._single_excitations, self._double_excitations = \
                UCCSD.compute_excitation_lists([self._num_alpha, self._num_beta],
                                               self._num_orbitals,
                                               active_occupied, active_unoccupied,
                                               same_spin_doubles=self.same_spin_doubles,
                                               method_singles=self._method_singles,
                                               method_doubles=self._method_doubles,
                                               excitation_type=self._excitation_type,
                                               )
        if self.uccd_singlet:
            self._hopping_ops, _ = self._build_hopping_operators()
        else:
            self._hopping_ops, self._num_parameters = self._build_hopping_operators()
            self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

        if self.uccd_singlet:
            self._double_excitations_grouped = \
                UCCSD.compute_excitation_lists_singlet(self._double_excitations, num_orbitals)
            self.num_groups = len(self._double_excitations_grouped)

            logging.debug('Grouped double excitations for singlet ucc')
            logging.debug(self._double_excitations_grouped)

            self._num_parameters = self.num_groups
            self._bounds = [(-np.pi, np.pi) for _ in range(self.num_groups)]

            # this will order the hopping operators
            self.labeled_double_excitations = []
            for i in range(len(self._double_excitations)):
                self.labeled_double_excitations.append((self._double_excitations[i], i))

            order_hopping_op = UCCSD.order_labels_for_hopping_ops(self._double_excitations,
                                                                  self._double_excitations_grouped)
            logging.debug('New order for hopping ops')
            logging.debug(order_hopping_op)

            self._hopping_ops_doubles_temp = []
            self._hopping_ops_doubles = self._hopping_ops[len(self._single_excitations):]
            for i in order_hopping_op:
                self._hopping_ops_doubles_temp.append(self._hopping_ops_doubles[i])

            self._hopping_ops[len(self._single_excitations):] = self._hopping_ops_doubles_temp

        self._logging_construct_circuit = True

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
                               task_args=(self._num_orbitals, self._num_particles,
                                          self._qubit_mapping, self._two_qubit_reduction,
                                          self._z2_symmetries,
                                          self._force_no_tap_excitation),
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
                                two_qubit_reduction, z2_symmetries, force_no_tap_excitation=False):
        """
        Builds a hopping operator given the list of indices (index) that is a single or a double
        excitation.

        Args:
            index (list): a single or double excitation (e.g. double excitation [0,1,2,3] for a 4
                          spin-orbital system)
            num_orbitals (int): number of spin-orbitals
            num_particles (int): number of electrons
            qubit_mapping (str): qubit mapping type
            two_qubit_reduction (bool): reduce the number of qubits by 2 if
                                        parity qubit mapping is used
            z2_symmetries (Z2Symmetries): class that contains the symmetries
                                          of hamiltonian for tapering
            force_no_tap_excitation (bool): prevent tapering from eliminating excitations
        Returns:
            WeightedPauliOperator: qubit_op
            list: index
        """
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
            if not force_no_tap_excitation:
                qubit_op = z2_symmetries.taper(qubit_op) if symm_commuting else None
            else:
                qubit_op = z2_symmetries.taper(qubit_op)

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

        if not self.uccd_singlet:
            list_excitation_operators = [
                (self._hopping_ops[index % num_excitations], parameters[index])
                for index in range(self._depth * num_excitations)]
        else:
            list_excitation_operators = []
            counter = 0
            for i in range(int(self._depth * self.num_groups)):
                for _ in range(len(self._double_excitations_grouped[i % self.num_groups])):
                    list_excitation_operators.append((self._hopping_ops[counter],
                                                      parameters[i]))
                    counter += 1

        results = parallel_map(UCCSD._construct_circuit_for_one_excited_operator,
                               list_excitation_operators,
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
        # TODO: need to put -1j in the coeff of pauli since the Parameter.
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
                                 active_unocc_list=None, same_spin_doubles=True,
                                 method_singles='both', method_doubles='ucc',
                                 excitation_type='sd'):
        """
        Computes single and double excitation lists.

        Args:
            num_particles (Union(list, int)): number of particles, if it is a tuple, the first
                                              number is alpha and the second number if beta.
            num_orbitals (int): Total number of spin orbitals
            active_occ_list (list): List of occupied orbitals to include, indices are
                             0 to n where n is max(num_alpha, num_beta)
            active_unocc_list (list): List of unoccupied orbitals to include, indices are
                               0 to m where m is num_orbitals // 2 - min(num_alpha, num_beta)
            same_spin_doubles (bool): True to include alpha,alpha and beta,beta double excitations
                               as well as alpha,beta pairings. False includes only alpha,beta
            excitation_type (str): choose 'sd', 's', 'd' to compute q-UCCSD, q-UCCS,
                                   q-UCCD excitation lists
            method_singles (str):  specify type of single excitations, 'alpha', 'beta', 'both' only
                                   alpha or beta spin-orbital single excitations or both (all
                                   single excitations)
            method_doubles (str): choose method for double excitations 'ucc' (conventional ucc),
                                  'succ' (singlet ucc), 'succ_full' (singlet ucc full)

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

        beta_idx = num_orbitals // 2

        # making lists of indexes of MOs involved in excitations
        if active_occ_list is not None:
            active_occ_list = [i if i >= 0 else i + max(num_alpha, num_beta) for i in
                               active_occ_list]
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
            active_occ_list_beta = [i + beta_idx for i in range(0, num_beta)]

        if active_unocc_list is not None:
            active_unocc_list = [i + min(num_alpha, num_beta) if i >= 0
                                 else i + num_orbitals // 2 for i in active_unocc_list]
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
            active_unocc_list_beta = [i + beta_idx for i in range(num_beta, num_orbitals // 2)]

        logger.debug('active_occ_list_alpha %s', active_occ_list_alpha)
        logger.debug('active_unocc_list_alpha %s', active_unocc_list_alpha)

        logger.debug('active_occ_list_beta %s', active_occ_list_beta)
        logger.debug('active_unocc_list_beta %s', active_unocc_list_beta)

        single_excitations = []
        double_excitations = []

        # lists of single excitations
        if method_singles == 'alpha ':

            for occ_alpha in active_occ_list_alpha:
                for unocc_alpha in active_unocc_list_alpha:
                    single_excitations.append([occ_alpha, unocc_alpha])

        elif method_singles == 'beta':

            for occ_beta in active_occ_list_beta:
                for unocc_beta in active_unocc_list_beta:
                    single_excitations.append([occ_beta, unocc_beta])
        else:
            for occ_alpha in active_occ_list_alpha:
                for unocc_alpha in active_unocc_list_alpha:
                    single_excitations.append([occ_alpha, unocc_alpha])
            for occ_beta in active_occ_list_beta:
                for unocc_beta in active_unocc_list_beta:
                    single_excitations.append([occ_beta, unocc_beta])
            logger.info('Singles excitations with alphas and betas orbitals are used.')

        # different methods of excitations for double excitations
        if method_doubles in ['ucc', 'succ_full']:

            for occ_alpha in active_occ_list_alpha:
                for unocc_alpha in active_unocc_list_alpha:
                    for occ_beta in active_occ_list_beta:
                        for unocc_beta in active_unocc_list_beta:
                            double_excitations.append(
                                [occ_alpha, unocc_alpha, occ_beta, unocc_beta])
        # pair ucc
        elif method_doubles == 'pucc':
            for occ_alpha in active_occ_list_alpha:
                for unocc_alpha in active_unocc_list_alpha:
                    for occ_beta in active_occ_list_beta:
                        for unocc_beta in active_unocc_list_beta:
                            # makes sure the el. excite from same spatial to same spatial orbitals
                            if occ_beta - occ_alpha == num_orbitals / 2 \
                                    and unocc_beta - unocc_alpha == num_orbitals / 2:
                                double_excitations.append(
                                    [occ_alpha, unocc_alpha, occ_beta, unocc_beta])

        # singlet ucc
        elif method_doubles == 'succ':
            for i in active_occ_list_alpha:
                for i_prime in active_unocc_list_alpha:
                    for j in active_occ_list_beta:
                        for j_prime in active_unocc_list_beta:
                            if j - beta_idx >= i and j_prime - beta_idx >= i_prime:
                                double_excitations.append([i, i_prime, j, j_prime])

            same_spin_doubles = False
            logger.info('Same spin double excitations are forced to be disabled in'
                        'singlet ucc')

        # same spin excitations
        if same_spin_doubles and len(active_occ_list_alpha) > 1 and len(
                active_unocc_list_alpha) > 1:
            for i, occ_alpha in enumerate(active_occ_list_alpha[:-1]):
                for j, unocc_alpha in enumerate(active_unocc_list_alpha[:-1]):
                    for occ_alpha_1 in active_occ_list_alpha[i + 1:]:
                        for unocc_alpha_1 in active_unocc_list_alpha[j + 1:]:
                            double_excitations.append([occ_alpha, unocc_alpha,
                                                       occ_alpha_1, unocc_alpha_1])

            up_active_occ_list = active_occ_list_beta
            up_active_unocc_list = active_unocc_list_beta

            for i, occ_beta in enumerate(up_active_occ_list[:-1]):
                for j, unocc_beta in enumerate(up_active_unocc_list[:-1]):
                    for occ_beta_1 in up_active_occ_list[i + 1:]:
                        for unocc_beta_1 in up_active_unocc_list[j + 1:]:
                            double_excitations.append([occ_beta, unocc_beta,
                                                       occ_beta_1, unocc_beta_1])

        if excitation_type == 's':
            double_excitations = []
        elif excitation_type == 'd':
            single_excitations = []
        else:
            logger.info('Singles and Doubles excitations are used.')

        logger.debug('single_excitations (%s) %s', len(single_excitations), single_excitations)
        logger.debug('double_excitations (%s) %s', len(double_excitations), double_excitations)

        return single_excitations, double_excitations

    # below are all tool functions that serve to group excitations that are controlled by
    # same angle theta in singlet ucc
    @staticmethod
    def compute_excitation_lists_singlet(double_exc, num_orbitals):
        """
        Outputs the list of lists of grouped excitation. A single list inside is controlled by
        the same parameter theta.

        Args:
            double_exc (list): exc.group. [[0,1,2,3], [...]]
            num_orbitals (int): number of molecular orbitals

        Returns:
            list: de_groups grouped excitations
        """
        de_groups = UCCSD.group_excitations_if_same_ao(double_exc, num_orbitals)

        return de_groups

    @staticmethod
    def same_ao_double_excitation_block_spin(de_1, de_2, num_orbitals):
        """
        Regroups the excitations that involve same spatial orbitals
        for example, with labeling.

        2--- ---5
        1--- ---4
        0-o- -o-3

        excitations [0,1,3,5] and [0,2,3,4] are controlled by the same parameter in the full
        singlet UCCSD unlike in usual UCCSD where every excitation is controlled by independent
        parameter.

        Args:
             de_1 (list): double exc in block spin [ from to from to ]
             de_2 (list): double exc in block spin [ from to from to ]
             num_orbitals (int): number of molecular orbitals

        Returns:
             int: says if given excitation involves same spatial orbitals 1 = yes, 0 = no.
        """
        half_active_space = int(num_orbitals / 2)

        de_1_new = copy.copy(de_1)
        de_2_new = copy.copy(de_2)

        count = -1
        for ind in de_1_new:
            count += 1
            if ind >= half_active_space:
                de_1_new[count] = ind % half_active_space
        count = -1
        for ind in de_2_new:
            count += 1
            if ind >= half_active_space:
                de_2_new[count] = ind % half_active_space

        # check if 2 unordered lists are same (involve same AOs)
        if collections.Counter(de_1_new) == collections.Counter(de_2_new):
            # we check that the permutations of terms i,j and k,l in [[i,j][k,l]] [[a,b][c,d]
            # as [i,j] ==? [a,b] or [c,d] and [k,l] ==? ...
            # then only return 0, basically criterion for equivalence of 2 mirror excitations
            return 1
        else:
            return 0

    @staticmethod
    def group_excitations(list_de, num_orbitals):
        """
        Groups the excitations and gives out the remaining ones in the list_de_temp list
        because those excitations are controlled by the same parameter in full singlet UCCSD
        unlike in usual UCCSD where every excitation has its own parameter.

        Args:
            list_de (list): list of the double excitations grouped
            num_orbitals (int): number of spin-orbitals (qubits)

        Returns:
            tuple: list_same_ao_group, list_de_temp, the grouped double_exc
            (that involve same spatial orbitals)
        """
        list_de_temp = copy.copy(list_de)
        list_same_ao_group = []
        de1 = list_de[0]
        counter = 0
        for de2 in list_de:
            if UCCSD.same_ao_double_excitation_block_spin(de1, de2, num_orbitals) == 1:
                counter += 1
                if counter == 1:
                    list_same_ao_group.append(de1)
                    for i in list_de_temp:
                        if i == de1:
                            list_de_temp.remove(de1)
                if de1 != de2:
                    list_same_ao_group.append(de2)
                for i in list_de_temp:
                    if i == de2:
                        list_de_temp.remove(de2)

        return list_same_ao_group, list_de_temp

    @staticmethod
    def group_excitations_if_same_ao(list_de, num_orbitals):
        """
        Define that, given list of double excitations list_de and number of spin-orbitals
        num_orbitals, which excitations involve the same spatial orbitals for full singlet UCCSD.

        Args:
            list_de (list): list of double exc
            num_orbitals (int): number of spin-orbitals

        Returns:
            list: grouped list of excitations
        """
        list_groups = []
        list_same_ao_group, list_de_temp = UCCSD.group_excitations(list_de, num_orbitals)
        list_groups.append(list_same_ao_group)
        while len(list_de_temp) != 0:
            list_same_ao_group, list_de_temp = UCCSD.group_excitations(list_de_temp, num_orbitals)
            list_groups.append(list_same_ao_group)

        return list_groups

    @staticmethod
    def order_labels_for_hopping_ops(double_exc, gde):
        """
        Orders the hopping operators according to the grouped excitations for the full singlet
        UCCSD.

        Args:
            double_exc (list): list of double excitations
            gde (list of lists): list of grouped excitations for full singlet UCCSD

        Returns:
            list: ordered_labels to order hopping ops
        """

        labeled_de = []
        for i, _ in enumerate(double_exc):
            labeled_de.append((double_exc[i], i))

        ordered_labels = []
        for group in gde:
            for exc in group:
                for l_e in labeled_de:
                    if exc == l_e[0]:
                        ordered_labels.append(l_e[1])

        return ordered_labels
