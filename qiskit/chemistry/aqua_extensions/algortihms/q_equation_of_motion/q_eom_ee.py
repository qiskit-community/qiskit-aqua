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

import numpy as np

from qiskit.aqua import QuantumAlgorithm, AquaError
from qiskit.aqua.algorithms import ExactEigensolver
from .q_equation_of_motion import QEquationOfMotion

logger = logging.getLogger(__name__)


class QEomEE(ExactEigensolver):

    CONFIGURATION = {
        'name': 'Q_EOM_EE',
        'description': 'Q_EOM with ExactEigensolver Algorithm to find the reference state',
        'classical': True,
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'qeom_vqe_schema',
            'type': 'object',
            'properties': {
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
                'active_occupied': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'active_unoccupied': {
                    'type': ['array', 'null'],
                    'default': None
                },
                'is_eom_matrix_symmetric': {
                    'type': 'boolean',
                    'default': True
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy', 'excited_states']
    }

    def __init__(self, operator, aux_operators=None,
                 num_orbitals=4, num_particles=2, qubit_mapping='parity',
                 two_qubit_reduction=True, active_occupied=None, active_unoccupied=None,
                 is_eom_matrix_symmetric=True, se_list=None, de_list=None,
                 z2_symmetries=None, untapered_op=None):
        """
        Args:
            operator (BaseOperator): qubit operator
            aux_operators ([BaseOperator]): Auxiliary operators to be evaluated at each eigenvalue
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
            untapered_op (BaseOperator): if the operator is tapered, we need untapered operator
                                     to build element of EoM matrix
        """
        self.validate(locals())
        super().__init__(operator, 1, aux_operators)

        self.qeom = QEquationOfMotion(operator, num_orbitals, num_particles, qubit_mapping,
                                      two_qubit_reduction, active_occupied, active_unoccupied,
                                      is_eom_matrix_symmetric, se_list, de_list,
                                      z2_symmetries, untapered_op)

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance.

        Args:
            params (dict): parameters dictionary
            algo_input (EnergyInput): EnergyInput instance
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        eom_vqe_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        num_orbitals = eom_vqe_params.get('num_orbitals')
        num_particles = eom_vqe_params.get('num_particles')
        qubit_mapping = eom_vqe_params.get('qubit_mapping')
        two_qubit_reduction = eom_vqe_params.get('two_qubit_reduction')
        active_occupied = eom_vqe_params.get('active_occupied')
        active_unoccupied = eom_vqe_params.get('active_unoccupied')

        return cls(operator, aux_operators=algo_input.aux_ops, num_orbitals=num_orbitals,
                   num_particles=num_particles,
                   qubit_mapping=qubit_mapping, two_qubit_reduction=two_qubit_reduction,
                   active_occupied=active_occupied, active_unoccupied=active_unoccupied)

    def _run(self):
        super()._run()
        wave_fn = self._ret['eigvecs'][0]
        excitation_energies_gap, eom_matrices = self.qeom.calculate_excited_states(wave_fn)
        excitation_energies = excitation_energies_gap + self._ret['energy']
        all_energies = np.concatenate(([self._ret['energy']], excitation_energies))
        self._ret['energy_gap'] = excitation_energies_gap
        self._ret['energies'] = all_energies
        self._ret['eom_matrices'] = eom_matrices
        return self._ret
