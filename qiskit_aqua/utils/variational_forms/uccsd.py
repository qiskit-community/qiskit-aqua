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
"""
    This trial wavefunction is a Unitary Coupled-Cluster Single and Double excitations
    variational form.
    For more information, see https://arxiv.org/abs/1805.04340
"""

import logging

import numpy as np
from qiskit import QuantumRegister, QuantumCircuit

from qiskit_aqua.utils.variational_forms import VariationalForm

try:
    from qiskit_aqua_chemistry.fermionic_operator import FermionicOperator
except ImportError:
    raise ImportWarning('UCCSD can be only used with qiskit_aqua_chemistry lib. \
        If you would like to use it for other purposes, please install qiskit_aqua_chemistry first.')

logger = logging.getLogger(__name__)


class VarFormUCCSD(VariationalForm):
    """
        This trial wavefunction is a Unitary Coupled-Cluster Single and Double excitations
        variational form.
        For more information, see https://arxiv.org/abs/1805.04340
    """

    UCCSD_CONFIGURATION = {
        'name': 'UCCSD',
        'description': 'UCCSD Variational Form',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
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
                    'type': 'integer',
                    'default': 2,
                    'minimum': 1
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
                    'oneOf': [
                        {'enum': ['jordan_wigner', 'parity', 'bravyi_kitaev']}
                    ]
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
            },
            'additionalProperties': False
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.UCCSD_CONFIGURATION.copy())
        self._num_qubits = 0
        self._depth = 0
        self._num_orbitals = 0
        self._num_particles = 0
        self._single_excitations = None
        self._double_excitations = None
        self._qubit_mapping = None
        self._initial_state = None
        self._two_qubit_reduction = False
        self._num_time_slices = 1
        self._num_parameters = 0
        self._bounds = None

    def init_args(self, num_qubits, depth, num_orbitals, num_particles,
                  active_occupied=None, active_unoccupied=None, initial_state=None,
                  qubit_mapping='parity', two_qubit_reduction=False, num_time_slices=1):
        """
        Args:
            - num_orbitals (int): number of spin orbitals
            - depth (int): number of replica of basic module
            - num_particles (int): number of particles
            - active_occupied (list): list of occupied orbitals to consider as active space
            - active_unoccupied (list): list of unoccupied orbitals to consider as active space
            - initial_state (InitialState): An initial state object.
            - qubit_mapping (str): qubit mapping type.
            - two_qubit_reduction (bool): two qubit reduction is applied or not.
            - num_time_slices (int): parameters for dynamics.
        """
        self._num_qubits = num_orbitals if not two_qubit_reduction else num_orbitals - 2
        if self._num_qubits != num_qubits:
            raise ValueError('Computed num qubits {} does not match actual {}'.format(self._num_qubits, num_qubits))
        self._num_orbitals = num_orbitals
        self._depth = depth
        self._num_particles = num_particles

        self._initial_state = initial_state
        self._qubit_mapping = qubit_mapping
        self._two_qubit_reduction = two_qubit_reduction
        self._num_time_slices = num_time_slices

        self._single_excitations, self._double_excitations = \
            VarFormUCCSD.compute_excitation_lists(num_particles, num_orbitals, active_occupied, active_unoccupied)

        self._num_parameters = (len(self._single_excitations) + len(self._double_excitations)) * self._depth
        self._bounds = [(-np.pi, np.pi) for _ in range(self._num_parameters)]

    def construct_circuit(self, parameters):
        """
        Construct the variational form, given its parameters.

        Args:
            parameters (numpy.ndarray) : circuit parameters

        Returns:
            QuantumCircuit: a quantum circuit with given `parameters`

        Raises:
            ValueError: the number of parameters is incorrect.
        """
        if len(parameters) != self._num_parameters:
            raise ValueError('The number of parameters has to be {}'.format(self._num_parameters))

        q = QuantumRegister(self._num_qubits, name='q')
        if self._initial_state is not None:
            circuit = self._initial_state.construct_circuit('circuit', q)
        else:
            circuit = QuantumCircuit(q)

        param_idx = 0
        two_d_zeros = np.zeros((self._num_orbitals, self._num_orbitals))
        four_d_zeros = np.zeros((self._num_orbitals, self._num_orbitals,
                                 self._num_orbitals, self._num_orbitals))
        dummpy_fer_op = FermionicOperator(h1=two_d_zeros, h2=four_d_zeros)
        for d in range(self._depth):
            for s_e_qubits in self._single_excitations:
                h1 = two_d_zeros.copy()
                h2 = four_d_zeros.copy()
                h1[s_e_qubits[0], s_e_qubits[1]] = 1.0
                h1[s_e_qubits[1], s_e_qubits[0]] = -1.0
                dummpy_fer_op.h1 = h1
                dummpy_fer_op.h2 = h2

                qubitOp = dummpy_fer_op.mapping(self._qubit_mapping)
                qubitOp = qubitOp.two_qubit_reduced_operator(
                    self._num_particles) if self._two_qubit_reduction else qubitOp
                circuit.extend(qubitOp.evolve(
                    None, parameters[param_idx] * -1j, 'circuit', self._num_time_slices, q))
                param_idx += 1

            for d_e_qubits in self._double_excitations:
                h1 = two_d_zeros.copy()
                h2 = four_d_zeros.copy()
                h2[d_e_qubits[0], d_e_qubits[1], d_e_qubits[2], d_e_qubits[3]] = 1.0
                h2[d_e_qubits[3], d_e_qubits[2], d_e_qubits[1], d_e_qubits[0]] = -1.0
                dummpy_fer_op.h1 = h1
                dummpy_fer_op.h2 = h2

                qubitOp = dummpy_fer_op.mapping(self._qubit_mapping)
                qubitOp = qubitOp.two_qubit_reduced_operator(
                    self._num_particles) if self._two_qubit_reduction else qubitOp
                circuit.extend(qubitOp.evolve(
                    None, parameters[param_idx] * -1j, 'circuit', self._num_time_slices, q))
                param_idx += 1

        return circuit

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
    def compute_excitation_lists(num_particles, num_orbitals, active_occ_list=None, active_unocc_list=None):
        """
        Computes single and double excitation lists
        Args:
            num_particles: Total number of particles
            num_orbitals:  Total number of spin orbitals
            active_occ_list: List of occupied orbitals to include, indices are
                             0 to n where n is num particles // 2
            active_unocc_list: List of unoccupied orbitals to include, indices are
                               0 to m where m is (num_orbitals - num particles) // 2

        Returns:
            Single and double excitation lists
        """
        if num_particles < 2 or num_particles % 2 != 0:
            raise ValueError('Invalid number of particles {}'.format(num_particles))
        if num_orbitals < 4 or num_orbitals % 2 != 0:
            raise ValueError('Invalid number of orbitals {}'.format(num_orbitals))
        if num_orbitals <= num_particles:
            raise ValueError('No unoccupied orbitals')
        if active_occ_list is not None:
            active_occ_list = [i if i >= 0 else i + num_particles // 2 for i in active_occ_list]
            for i in active_occ_list:
                if i >= num_particles // 2:
                    raise ValueError('Invalid index {} in active active_occ_list {}'.format(i, active_occ_list))
        if active_unocc_list is not None:
            active_unocc_list = [i + num_particles // 2 if i >=
                                 0 else i + num_orbitals // 2 for i in active_unocc_list]
            for i in active_unocc_list:
                if i < 0 or i >= num_orbitals // 2:
                    raise ValueError('Invalid index {} in active active_unocc_list {}'.format(i, active_unocc_list))

        if active_occ_list is None or len(active_occ_list) <= 0:
            active_occ_list = [i for i in range(0, num_particles // 2)]

        if active_unocc_list is None or len(active_unocc_list) <= 0:
            active_unocc_list = [i for i in range(num_particles // 2, num_orbitals // 2)]

        single_excitations = []
        double_excitations = []

        logger.debug('active_occ_list {}'.format(active_occ_list))
        logger.debug('active_unocc_list {}'.format(active_unocc_list))

        beta_idx = num_orbitals // 2
        for occ_alpha in active_occ_list:
            for unocc_alpha in active_unocc_list:
                single_excitations.append([occ_alpha, unocc_alpha])

        for occ_beta in [i + beta_idx for i in active_occ_list]:
            for unocc_beta in [i + beta_idx for i in active_unocc_list]:
                single_excitations.append([occ_beta, unocc_beta])

        for occ_alpha in active_occ_list:
            for unocc_alpha in active_unocc_list:
                for occ_beta in [i + beta_idx for i in active_occ_list]:
                    for unocc_beta in [i + beta_idx for i in active_unocc_list]:
                        double_excitations.append([occ_alpha, unocc_alpha, occ_beta, unocc_beta])

        logger.debug('single_excitations {}'.format(single_excitations))
        logger.debug('double_excitations {}'.format(double_excitations))

        return single_excitations, double_excitations
