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
And for singlet q-UCCD and paired q-UCCD see: https://arxiv.org/abs/1911.10864
"""

import logging
import sys
import collections
import copy

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.tools import parallel_map
from qiskit.tools.events import TextProgressBar

from qiskit.aqua import aqua_globals
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

    CONFIGURATION = {
        'name': 'UCCSD',
        'description': 'UCCSD Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'uccsd_schema',
            'type': 'object',
            'properties': {
                'depth': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                'num_orbitals': {
                    'type': 'integer',
                    'default': 4,
                    'minimum': 1
                },
                'num_particles': {
                    'type': ['array', 'integer'],
                    'default': [1, 1],
                    'contains': {
                        'type': 'integer'
                    },
                    'minItems': 2,
                    'maxItems': 2
                },
                'active_occupied': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'active_unoccupied': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'qubit_mapping': {
                    'type': 'string',
                    'default': 'parity',
                    'enum': ['jordan_wigner', 'parity', 'bravyi_kitaev']
                },
                'two_qubit_reduction': {
                    'type': 'boolean',
                    'default': True
                },
                'num_time_slices': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                'method_singles': {
                    'type': 'string',
                    'default': 'both',
                    'enum': ['both', 'alpha', 'beta']
                },
                'method_doubles': {
                    'type': 'string',
                    'default': 'ucc',
                    'enum': ['ucc', 'pucc', 'succ', 'succ_full']
                },
                'exc_type': {
                    'type': 'string',
                    'default': 'sd',
                    'enum': ['sd', 's', 'd']
                },
            },
            'additionalProperties': False
        },
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'HartreeFock',
                }
            },
        ],
    }

    def __init__(self, num_qubits, depth, num_orbitals, num_particles,
                 active_occupied=None, active_unoccupied=None, initial_state=None,
                 qubit_mapping='parity', two_qubit_reduction=True, num_time_slices=1,
                 shallow_circuit_concat=True, z2_symmetries=None,
                 method_singles='both', method_doubles='ucc', exc_type='sd',
                 same_spin_doubles=True, force_no_tap_excitation=False):
        """Constructor.

        Args:
            num_qubits (int): number of qubits
            depth (int): number of replica of basic module
            num_orbitals (int): number of spin orbitals
            num_particles (Union(list, int)): number of particles, if it is a list,
                                        the first number is alpha and the second number if beta.
            active_occupied (list): list of occupied orbitals to consider as active space
            active_unoccupied (list): list of unoccupied orbitals to consider as active space
            initial_state (InitialState): An initial state object.
            qubit_mapping (str): qubit mapping type.
            two_qubit_reduction (bool): two qubit reduction is applied or not.
            num_time_slices (int): parameters for dynamics.
            z2_symmetries (Z2Symmetries): represent the Z2 symmetries, including symmetries,
                                          sq_paulis, sq_list, tapering_values, and cliffords
            shallow_circuit_concat (bool): indicate whether to use shallow (cheap) mode for
                                           circuit concatenation
            method_singles (str): 'alpha', 'beta', 'both' only alpha or beta spin-orbital
                            single exc. or both (all)
            method_doubles (str): 'ucc' (conventional ucc), succ (singlet ucc,
                            "https://arxiv.org/abs/1911.10864"
                            Eq.(14) page 5), succ_full (singlet ucc full
                            this implements the "https://arxiv.org/abs/1911.10864" Eq.(16) page 5.)
            exc_type (str): 'sd', 's', 'd', choose q-UCCSD, q-UCCS, q-UCCD
            same_spin_doubles (bool):, enable double excitations of the same spin
            force_no_tap_excitation (bool): keep all the excitation regardless if tapering is used

         Raises:
             ValueError: Computed qubits do not match actual value

        """
        # basic parameters
        self.validate(locals())
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
        self._exc_type = exc_type
        self.same_spin_doubles = same_spin_doubles

        self._single_excitations, self._double_excitations = \
            UCCSD.compute_excitation_lists([self._num_alpha, self._num_beta], self._num_orbitals,
                                           active_occupied, active_unoccupied,
                                           same_spin_doubles=self.same_spin_doubles,
                                           method_singles=self._method_singles,
                                           method_doubles=self._method_doubles,
                                           exc_type=self._exc_type, )

        self._hopping_ops, self._num_parameters = self._build_hopping_operators()
        self._excitation_pool = None
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

        self._logging_construct_circuit = True
        self._support_parameterized_circuit = True

        self.uccd_singlet = False
        if self._method_doubles == 'succ_full':
            # this implements the UCCD0-full "https://arxiv.org/abs/1911.10864" Eq.(16) page 5.
            self.uccd_singlet = True
            self._single_excitations, self._double_excitations = \
                UCCSD.compute_excitation_lists([self._num_alpha, self._num_beta],
                                               self._num_orbitals,
                                               active_occupied, active_unoccupied,
                                               same_spin_doubles=self.same_spin_doubles,
                                               method_singles=self._method_singles,
                                               method_doubles=self._method_doubles,
                                               exc_type=self._exc_type,
                                               )
        if self.uccd_singlet:
            self._hopping_ops, _ = self._build_hopping_operators()
        else:
            self._hopping_ops, self._num_parameters = self._build_hopping_operators()
            self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

        if self.uccd_singlet:
            # logging.debug('Reordered hopping ops')
            # logging.debug('Original hopping ops')
            # logging.debug(self._hopping_ops)
            self._double_excitations_grouped = \
                UCCSD.compute_excitation_lists_singlet(self._double_excitations, num_orbitals)
            self.num_groups = len(self._double_excitations_grouped)

            self._num_parameters = self.num_groups
            self._bounds = [(-np.pi, np.pi) for _ in range(self.num_groups)]

            # this will order the hopping operators (hop hopping_op per excitation)
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

            # logging.debug('Reordered hopping ops')
            # logging.debug(self._hopping_ops)

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

    def _build_hopping_operators(self, excitation_list=None):
        if logger.isEnabledFor(logging.DEBUG):
            TextProgressBar(sys.stderr)

        # change 1: custom excitation list is allowed
        if excitation_list is None:
            results = parallel_map(UCCSD._build_hopping_operator,
                                   self._single_excitations + self._double_excitations,
                                   task_args=(self._num_orbitals,
                                              self._num_particles, self._qubit_mapping,
                                              self._two_qubit_reduction, self._z2_symmetries),
                                   num_processes=aqua_globals.num_processes)
        else:
            results = parallel_map(UCCSD._build_hopping_operator,
                                   excitation_list,
                                   task_args=(self._num_orbitals,
                                              self._num_particles, self._qubit_mapping,
                                              self._two_qubit_reduction, self._z2_symmetries),
                                   num_processes=aqua_globals.num_processes)
        hopping_ops = []
        s_e_list = []
        d_e_list = []
        for hopping_op, index in results:
            if hopping_op is not None and not hopping_op.is_empty():
                hopping_ops.append(hopping_op)
                if len(index) == 2:  # for double excitation
                    s_e_list.append(index)
                else:  # for double excitation
                    d_e_list.append(index)

        self._single_excitations = s_e_list
        self._double_excitations = d_e_list

        num_parameters = len(hopping_ops) * self._depth
        return hopping_ops, num_parameters

    # change 2: added force_no_tap_excitation to taper the excitations ops when necessary only
    # (user should be aware that some excitations are gone because tapering is used)
    @staticmethod
    def _build_hopping_operator(index, num_orbitals, num_particles, qubit_mapping,
                                two_qubit_reduction, z2_symmetries, force_no_tap_excitation=False):

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

        # change 2: option to taper the uccsd excitation ops
        # (by default if tapering is used, the excitations ops are also tapered)
        if force_no_tap_excitation is False:
            if not z2_symmetries.is_empty():
                # symm_commuting = True
                for symmetry in z2_symmetries.symmetries:
                    symmetry_op = WeightedPauliOperator(paulis=[[1.0, symmetry]])
                    symm_commuting = qubit_op.commute_with(symmetry_op)
                    if not symm_commuting:
                        break
                    qubit_op = z2_symmetries.taper(qubit_op) if symm_commuting else None

        if qubit_op is None:
            logger.info('Excitation (%s) is skipped since it is not commuted '
                        'with symmetries (set force_no_tap_excitation'
                        '=True to keep all excitations)',
                        ','.join([str(x) for x in index]))
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

    # change 3: to include singlet
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

        # define the list of operators
        if not self.uccd_singlet:
            # make a list of excited operators
            list_excitation_operators = [
                (self._hopping_ops[index % num_excitations], parameters[index])
                for index in range(self._depth * num_excitations)]

            # assign parameters according to groups
            # reorder the self._hopping_ops
        else:
            list_excitation_operators = []
            # you just count through the operators for exc and assign the parameter if that operator
            # corresponds to the group, of course all operators are in the correct order of groups
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
    def _interleaved_spin_to_block_spin(i_end_act_sp, single_exc_op,
                                        double_exc_op):
        """
        Function that changes the excitation labels from interleaved to block spin orbital
        labeling convention. Transforms the excitation [occ, occ, vir, vir] -> [occ, vir, occ,
        vir], for example for H2 is 631G
        we have 8 qubits with JW mapping so the double excitations [[0,1,2,3],...] (interleaved)
        becomes [[0,1,4,5],...] (block spin).

        Active space

                     4 -   - 5       2 -   - 5
                     2 -   - 3   ->  1 -   - 4
        spin orbital 0 -   - 1       0 -   - 3

        Args:
            i_end_act_sp (int): indice of the last orbital in the AS, if 8 MOs then its index is 7
            single_exc_op (list): [[0,2], [1,3],..]
            double_exc_op (list): [[0,2], [1,3],..]

        Returns:
            list: single_exc_op, list: double_exc_op

        """
        for i, _ in enumerate(double_exc_op):
            # transform to Aqua notation

            # take an excitation
            list_exc = double_exc_op[i]
            # max_index = i_end_act_sp

            # transform each indice to the block spin indice
            for j, _ in enumerate(list_exc):
                list_exc[j] = UCCSD._interleaved_to_block_spin_single_indice(i_end_act_sp,
                                                                             list_exc[j])

            # overwrite the old excitation
            double_exc_op[i] = list_exc

            # permute the indices to Aqua convention
            # occ occ vir vir -> occ vir occ vir
            list_exc = double_exc_op[i]
            list_exc_temp_1 = list_exc[0]
            list_exc_temp_2 = list_exc[1]
            list_exc_temp_3 = list_exc[2]
            list_exc_temp_4 = list_exc[3]
            list_exc[0] = list_exc_temp_1
            list_exc[1] = list_exc_temp_3
            list_exc[2] = list_exc_temp_2
            list_exc[3] = list_exc_temp_4

            # overwrite again
            double_exc_op[i] = list_exc

        # same procedure for single excitations
        for i, _ in enumerate(single_exc_op):
            # transform to Aqua notation
            list_exc = single_exc_op[i]
            max_index = i_end_act_sp
            for k, _ in enumerate(list_exc):
                if list_exc[k] % 2 == 0:
                    list_exc[k] = int(list_exc[k] / 2)
                elif list_exc[k] % 2 != 0:
                    list_exc[k] = int((list_exc[k] + max_index) / 2)
            single_exc_op[i] = list_exc

        return single_exc_op, double_exc_op

    @staticmethod
    def _block_spin_to_interleaved_single_indice(i_end_act_sp, indice):
        """
        It converts an indice in block_spin to interleaved notation.
        i.exc. if 8 orbitals, orbital 4 becomes orbital 1.

        Args:
            i_end_act_sp int: label of the last spin orbital
            indice int: which indice will be converted to interleaved notation

        Returns:
            int: converted_indice, indice converted to interleaved notation
        """

        if indice < (i_end_act_sp + 1) / 2:
            converted_indice = int(indice * 2)
        else:
            converted_indice = int(2 * (indice - (i_end_act_sp + 1) / 2) + 1)

        return converted_indice

    @staticmethod
    def _interleaved_to_block_spin_single_indice(i_end_act_sp, indice):
        """
        It converts an indice in interleaved spin to block spin (Aqua) notation.
        i.exc. if 8 orbitals, orbital 4 becomes orbital 1

        Args:
            i_end_act_sp int: label of the last spin orbital
            indice int: which indice will be converted to blockspin notation

        Returns:
            int: converted_indice, indice converted to blockspin notation
        """
        converted_indice = 0
        if indice % 2 == 0:
            converted_indice = int(indice / 2)
        elif indice % 2 != 0:
            converted_indice = int((indice + i_end_act_sp) / 2)

        return converted_indice

    @staticmethod
    def compute_excitation_lists(num_particles, num_orbitals, active_occ_list=None,
                                 active_unocc_list=None, same_spin_doubles=True,
                                 method_singles='both', method_doubles='ucc',
                                 exc_type='sd'):
        """

        Computes single and double excitation lists

        Args:
            num_particles (list): number of particles, if it is a tuple, the first number is
                                  alpha and the second number if beta.
            num_orbitals (int): Total number of spin orbitals
            active_occ_list (list): List of occupied orbitals to include, indices are
                             0 to n where n is max(num_alpha, num_beta)
            active_unocc_list (list): List of unoccupied orbitals to include, indices are
                               0 to m where m is num_orbitals // 2 - min(num_alpha, num_beta)
            same_spin_doubles (bool): True to include alpha,alpha and beta,beta double excitations
                               as well as alpha,beta pairings. False includes only alpha,beta
            exc_type (str): 'sd', 's', 'd' compute q-UCCSD, q-UCCS, q-UCCD excitation lists
            method_singles (str):  'alpha', 'beta', 'both' only alpha or beta spin-orbital
                            single exc. or both (all)
            method_doubles (str): 'ucc' (conventional ucc), succ (singlet ucc,
                            "https://arxiv.org/abs/1911.10864" Eq.(14) page 5),
                            succ_full (singlet ucc full this implements the
                            "https://arxiv.org/abs/1911.10864" Eq.(14) page 5.)

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

        # Constructing the lists of excitations
        #  single excitation on both or only alpha or beta orbitals
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
        # paired ucc
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
        # singlet ucc (different to full singlet ucc that actually)
        # this implements the "https://arxiv.org/abs/1911.10864" Eq.(14) page 5.
        elif method_doubles == 'succ':

            act_unocc_alpha_block_spin = [
                UCCSD._block_spin_to_interleaved_single_indice(num_orbitals - 1, i)
                for i in active_unocc_list_alpha]
            act_unocc_beta_block_spin = [
                UCCSD._block_spin_to_interleaved_single_indice(num_orbitals - 1, i)
                for i in active_unocc_list_beta]
            act_occ_alpha_block_spin = [
                UCCSD._block_spin_to_interleaved_single_indice(num_orbitals - 1, i)
                for i in active_occ_list_alpha]
            act_occ_beta_block_spin = [
                UCCSD._block_spin_to_interleaved_single_indice(num_orbitals - 1, i)
                for i in active_occ_list_beta]

            # build double excitations in the succ way in interleaved notations
            for i in act_occ_alpha_block_spin:
                for i_prime in act_unocc_alpha_block_spin:
                    for j in act_occ_beta_block_spin:
                        for j_prime in act_unocc_beta_block_spin:
                            if j > i and j_prime > i_prime:
                                double_excitations.append([i, j, i_prime, j_prime])

            # translate the interleaved excitations (occ, occ, vir, vir) with inter labeling
            # back to block spin (occ, vir, occ, vir)
            _, double_excitations = UCCSD._interleaved_spin_to_block_spin(num_orbitals - 1, [],
                                                                          double_excitations)
        # same spin excitations
        if same_spin_doubles and len(active_occ_list_alpha) > 1 and len(
                active_unocc_list_alpha) > 1:
            for i, occ_alpha in enumerate(active_occ_list_alpha[:-1]):
                for j, unocc_alpha in enumerate(active_unocc_list_alpha[:-1]):
                    for occ_alpha_1 in active_occ_list_alpha[i + 1:]:
                        for unocc_alpha_1 in active_unocc_list_alpha[j + 1:]:
                            double_excitations.append([occ_alpha, unocc_alpha,
                                                       occ_alpha_1, unocc_alpha_1])

            # TODO look into this line if there is any problems
            up_active_occ_list = active_occ_list_beta
            up_active_unocc_list = active_unocc_list_beta

            for i, occ_beta in enumerate(up_active_occ_list[:-1]):
                for j, unocc_beta in enumerate(up_active_unocc_list[:-1]):
                    for occ_beta_1 in up_active_occ_list[i + 1:]:
                        for unocc_beta_1 in up_active_unocc_list[j + 1:]:
                            double_excitations.append([occ_beta, unocc_beta,
                                                       occ_beta_1, unocc_beta_1])

        # if one wants only single or double excitations
        if exc_type == 's':
            double_excitations = []
        elif exc_type == 'd':
            single_excitations = []
        else:
            logger.info('Singles and Doubles excitations are used.')

        logger.debug('single_excitations (%s) %s', len(single_excitations), single_excitations)
        logger.debug('double_excitations (%s) %s', len(double_excitations), double_excitations)

        return single_excitations, double_excitations

    # below are all tool functions that serve to group excitations that are controlled by
    # same angle theta in singlet ucc ("https://arxiv.org/abs/1911.10864")
    @staticmethod
    def compute_excitation_lists_singlet(double_exc, num_orbitals):
        """
        Outputs the list of lists of grouped excitation. A single list inside is controlled by
        the same parameter theta

        Args:
            double_exc (list): exc.group. [[0,1,2,3], [...]]
            num_orbitals (int): number of molecular orbitals

        Returns:
            list: de_groups grouped excitations
        """
        de_groups = UCCSD.group_excitations_if_same_ao(double_exc, num_orbitals)
        return de_groups

    @staticmethod
    def same_ao_double_excitation_block_spin(de_1, de_2, nmo):
        """
        Regroups the excitations that involve same spatial orbitals
        for example, with labeling

        2--- ---5
        1--- ---4
        0-o- -o-3

        excitations [0,1,3,5] and [0,2,3,4] are controlled by the same parameter in the full
        singlet UCCSD unlike in usual UCCSD where every excitation is controlled by independent
        parameter

        Args:
             de_1 (list): double exc in block spin [ from to from to ]
             de_2 (list): double exc in block spin [ from to from to ]
             nmo (int): number of molecular orbitals

        Returns:
             int: says if given excitation involves same spatial orbitals 1 = yes, 0 = no.
        """
        # for the RHF, may have to adapt to UHF
        half_active_space = int(nmo / 2)

        # writing the indices of the orbitals for 2 double excitations
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

        # the collections bastard actually does not cop[y but modifies the thing
        # so it goes out of the scope of function
        # I created with copy a separate object on which I operate\

        # first we check if 2 unordered lists are same (involve same AOs)
        if collections.Counter(de_1_new) == collections.Counter(de_2_new):
            # we check that the permutations of terms i,j and k,l in [[i,j][k,l]] [[a,b][c,d]
            # as [i,j] ==? [a,b] or [c,d] and [k,l] ==? ...
            # then only return 0
            # basically criterion for equivalence of 2 mirror excitations
            return 1
        else:
            return 0

    @staticmethod
    def group_excitations(list_de, nmo):
        """
        Groups the excitations and gives out the remaining ones in the list_de_temp list
        because those excitations are controlled by the same parameter in full singlet UCCSD
        unlike in usual UCCSD where every excitation has its own parameter.

        Args:
            list_de (list): list of the double excitations grouped
            nmo (int): number of spin-orbitals (qubits)
        Returns:
            tuple: list_same_ao_group, list_de_temp, the grouped double_exc
            (that involve same spatial orbitals)

        """
        list_de_temp = copy.copy(list_de)
        list_same_ao_group = []
        de1 = list_de[0]
        counter = 0
        for de2 in list_de:
            if UCCSD.same_ao_double_excitation_block_spin(de1, de2, nmo) == 1:
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
    def group_excitations_if_same_ao(list_de, nmo):
        """
        Define that, given list of double excitations list_de and number of spin-orbitals nmo,
        which excitations involve the same spatial orbitals for full singlet UCCSD

        Args:
            list_de (list): list of double exc
            nmo (int): number of spin-orbitals

        Returns:
            list: grouped list of excitations
        """
        list_groups = []
        list_same_ao_group, list_de_temp = UCCSD.group_excitations(list_de, nmo)
        list_groups.append(list_same_ao_group)
        while len(list_de_temp) != 0:
            list_same_ao_group, list_de_temp = UCCSD.group_excitations(list_de_temp, nmo)
            list_groups.append(list_same_ao_group)

        return list_groups

    @staticmethod
    def order_labels_for_hopping_ops(double_exc, gde):
        """
        Orders the hopping operators according to the grouped excitations for the
        full singlet UCCSD

        Args:
            double_exc (list): list of double excitations
            gde (list of lists): list of grouped excitations for full singlet UCCSD

        Returns:
            list: ordered_labels to order hopping ops
        """
        # import collections
        # making a labeled list
        labeled_de = []
        for i, _ in enumerate(double_exc):
            labeled_de.append((double_exc[i], i))

        # ordered labels
        ordered_labels = []
        for group in gde:
            for exc in group:
                for l_e in labeled_de:
                    if exc == l_e[0]:
                        # if collections.Counter(exc) == collections.Counter(l_e[0]):
                        ordered_labels.append(l_e[1])

        return ordered_labels
