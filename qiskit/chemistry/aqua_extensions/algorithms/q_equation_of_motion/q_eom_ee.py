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

""" QEomEE algorithm """

import logging

import numpy as np

from qiskit.aqua import QuantumAlgorithm, AquaError
from qiskit.aqua.algorithms import ExactEigensolver
from .q_equation_of_motion import QEquationOfMotion

logger = logging.getLogger(__name__)


class QEomEE(ExactEigensolver):
    """ QEomEE algorithm """
    CONFIGURATION = {
        'name': 'QEomEE',
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
                }
            },
            'additionalProperties': False
        },
        'problems': ['excited_states']
    }

    def __init__(self, operator, num_orbitals, num_particles, qubit_mapping='parity',
                 two_qubit_reduction=True, active_occupied=None, active_unoccupied=None,
                 is_eom_matrix_symmetric=True, se_list=None, de_list=None,
                 z2_symmetries=None, untapered_op=None, aux_operators=None):
        """
        Args:
            operator (BaseOperator): qubit operator
            num_orbitals (int):  total number of spin orbitals
            num_particles (Union(list, int)): number of particles, if it is a list,
                                        the first number is alpha and the second
                                        number if beta.
            qubit_mapping (str): qubit mapping type
            two_qubit_reduction (bool): two qubit reduction is applied or not
            active_occupied (list): list of occupied orbitals to include, indices are
                                    0 to n where n is num particles // 2
            active_unoccupied (list): list of unoccupied orbitals to include, indices are
                                    0 to m where m is (num_orbitals - num particles) // 2
            is_eom_matrix_symmetric (bool): is EoM matrix symmetric
            se_list (list[list]): single excitation list, overwrite the setting in active space
            de_list (list[list]): double excitation list, overwrite the setting in active space
            z2_symmetries (Z2Symmetries): represent the Z2 symmetries
            untapered_op (BaseOperator): if the operator is tapered, we need untapered operator
                                         to build element of EoM matrix
            aux_operators (list[BaseOperator]): Auxiliary operators to be evaluated at
                                                each eigenvalue
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
        Returns:
            QEomEE: Newly created instance
        Raises:
             AquaError: EnergyInput instance is required
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        q_eom_ee_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        num_orbitals = q_eom_ee_params.get('num_orbitals')
        num_particles = q_eom_ee_params.get('num_particles')
        qubit_mapping = q_eom_ee_params.get('qubit_mapping')
        two_qubit_reduction = q_eom_ee_params.get('two_qubit_reduction')
        active_occupied = q_eom_ee_params.get('active_occupied')
        active_unoccupied = q_eom_ee_params.get('active_unoccupied')

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
