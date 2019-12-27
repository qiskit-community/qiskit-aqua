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

""" QEomVQE algorithm """

import logging

import numpy as np
from qiskit.aqua.utils.validation import validate
from qiskit.aqua.algorithms import VQE
from .q_equation_of_motion import QEquationOfMotion

logger = logging.getLogger(__name__)


class QEomVQE(VQE):
    """ QEomVQE algorithm """

    _INPUT_SCHEMA = {
        '$schema': 'http://json-schema.org/draft-07/schema#',
        'id': 'qeom_vqe_schema',
        'type': 'object',
        'properties': {
            'initial_point': {
                'type': ['array', 'null'],
                "items": {
                    "type": "number"
                },
                'default': None
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
            'max_evals_grouped': {
                'type': 'integer',
                'default': 1
            }
        },
        'additionalProperties': False
    }

    def __init__(self, operator, var_form, optimizer, num_orbitals, num_particles,
                 initial_point=None, max_evals_grouped=1, callback=None,
                 auto_conversion=True, qubit_mapping='parity',
                 two_qubit_reduction=True, is_eom_matrix_symmetric=True,
                 active_occupied=None, active_unoccupied=None,
                 se_list=None, de_list=None, z2_symmetries=None,
                 untapered_op=None, aux_operators=None):
        """
        Args:
            operator (BaseOperator): qubit operator
            var_form (VariationalForm): parametrized variational form.
            optimizer (Optimizer): the classical optimization algorithm.
            num_orbitals (int):  total number of spin orbitals
            num_particles (Union(list, int)): number of particles, if it is a list,
                                              the first number is
                                              alpha and the second number if beta.
            initial_point (numpy.ndarray): optimizer initial point, 1-D vector
            max_evals_grouped (int): max number of evaluations performed simultaneously
            callback (Callable): a callback that can access the intermediate data during
                                 the optimization.
                                 Internally, four arguments are provided as follows
                                 the index of evaluation, parameters of variational form,
                                 evaluated mean, evaluated standard deviation.
            auto_conversion (bool): an automatic conversion for operator and aux_operators into
                                    the type which is most suitable for the backend.

                                    - non-aer statevector_simulator: MatrixOperator
                                    - aer statevector_simulator: WeightedPauliOperator
                                    - qasm simulator or real backend:
                                        TPBGroupedWeightedPauliOperator
            qubit_mapping (str): qubit mapping type
            two_qubit_reduction (bool): two qubit reduction is applied or not
            is_eom_matrix_symmetric (bool): is EoM matrix symmetric
            active_occupied (list): list of occupied orbitals to include, indices are
                                    0 to n where n is num particles // 2
            active_unoccupied (list): list of unoccupied orbitals to include, indices are
                                      0 to m where m is (num_orbitals - num particles) // 2
            se_list (list[list]): single excitation list, overwrite the setting in active space
            de_list (list[list]): double excitation list, overwrite the setting in active space
            z2_symmetries (Z2Symmetries): represent the Z2 symmetries
            untapered_op (BaseOperator): if the operator is tapered, we need untapered operator
                                         during building element of EoM matrix
            aux_operators (list[BaseOperator]): Auxiliary operators to be
                                                evaluated at each eigenvalue
        """
        validate(locals(), self._INPUT_SCHEMA)
        super().__init__(operator.copy(), var_form, optimizer, initial_point=initial_point,
                         max_evals_grouped=max_evals_grouped, aux_operators=aux_operators,
                         callback=callback, auto_conversion=auto_conversion)

        self.qeom = QEquationOfMotion(operator, num_orbitals, num_particles,
                                      qubit_mapping, two_qubit_reduction, active_occupied,
                                      active_unoccupied,
                                      is_eom_matrix_symmetric, se_list, de_list,
                                      z2_symmetries, untapered_op)

    def _run(self):
        super()._run()
        self._quantum_instance.circuit_summary = True
        opt_params = self._ret['opt_params']
        logger.info("opt params:\n%s", opt_params)
        wave_fn = self._var_form.construct_circuit(opt_params)
        excitation_energies_gap, eom_matrices = self.qeom.calculate_excited_states(
            wave_fn, quantum_instance=self._quantum_instance)
        excitation_energies = excitation_energies_gap + self._ret['energy']
        all_energies = np.concatenate(([self._ret['energy']], excitation_energies))
        self._ret['energy_gap'] = excitation_energies_gap
        self._ret['energies'] = all_energies
        self._ret['eom_matrices'] = eom_matrices
        return self._ret
