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

from qiskit.aqua import QuantumAlgorithm, AquaError
from qiskit.aqua import PluggableType, get_pluggable_class, Pluggable
from qiskit.aqua.algorithms import VQE
from .q_equation_of_motion import QEquationOfMotion

logger = logging.getLogger(__name__)


class QEomVQE(VQE):
    """ QEomVQE algorithm """
    CONFIGURATION = {
        'name': 'QEomVQE',
        'description': 'Q_EOM with VQE Algorithm to find the reference state',
        'input_schema': {
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
        },
        'problems': ['excited_states'],
        'depends': [
            {'pluggable_type': 'optimizer',
             'default': {
                 'name': 'L_BFGS_B'
             }
             },
            {'pluggable_type': 'variational_form',
             'default': {
                 'name': 'RYRZ'
             }
             },
        ],
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
        self.validate(locals())
        super().__init__(operator.copy(), var_form, optimizer, initial_point=initial_point,
                         max_evals_grouped=max_evals_grouped, aux_operators=aux_operators,
                         callback=callback, auto_conversion=auto_conversion)

        self.qeom = QEquationOfMotion(operator, num_orbitals, num_particles,
                                      qubit_mapping, two_qubit_reduction, active_occupied,
                                      active_unoccupied,
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
            QEomVQE: Newly created instance
        Raises:
             AquaError: EnergyInput instance is required
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        q_eom_vqe_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        initial_point = q_eom_vqe_params.get('initial_point')
        max_evals_grouped = q_eom_vqe_params.get('max_evals_grouped')
        num_orbitals = q_eom_vqe_params.get('num_orbitals')
        num_particles = q_eom_vqe_params.get('num_particles')
        qubit_mapping = q_eom_vqe_params.get('qubit_mapping')
        two_qubit_reduction = q_eom_vqe_params.get('two_qubit_reduction')
        active_occupied = q_eom_vqe_params.get('active_occupied')
        active_unoccupied = q_eom_vqe_params.get('active_unoccupied')

        # Set up variational form, we need to add computed num qubits, and initial state to params
        var_form_params = params.get(Pluggable.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = operator.num_qubits
        var_form = get_pluggable_class(PluggableType.VARIATIONAL_FORM,
                                       var_form_params['name']).init_params(params)

        # Set up optimizer
        opt_params = params.get(Pluggable.SECTION_KEY_OPTIMIZER)
        optimizer = get_pluggable_class(PluggableType.OPTIMIZER,
                                        opt_params['name']).init_params(params)

        return cls(operator, var_form, optimizer,
                   initial_point=initial_point, max_evals_grouped=max_evals_grouped,
                   aux_operators=algo_input.aux_ops, num_orbitals=num_orbitals,
                   num_particles=num_particles,
                   qubit_mapping=qubit_mapping, two_qubit_reduction=two_qubit_reduction,
                   active_occupied=active_occupied, active_unoccupied=active_unoccupied)

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
