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
The Variational Quantum Eigensolver algorithm.
See https://arxiv.org/abs/1304.3061
"""

import time
import logging

import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister

from qiskit_aqua import QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_optimizer_instance, get_variational_form_instance, get_initial_state_instance

logger = logging.getLogger(__name__)


class VQE(QuantumAlgorithm):
    """
    The Variational Quantum Eigensolver algorithm.
    See https://arxiv.org/abs/1304.3061
    """

    VQE_CONFIGURATION = {
        'name': 'VQE',
        'description': 'VQE Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'vqe_schema',
            'type': 'object',
            'properties': {
                'operator_mode': {
                    'type': 'string',
                    'default': 'matrix',
                    'oneOf': [
                        {'enum': ['matrix', 'paulis', 'grouped_paulis']}
                    ]
                },
                'initial_point': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy', 'ising'],
        'depends': ['optimizer', 'variational_form', 'initial_state'],
        'defaults': {
            'optimizer': {
                'name': 'L_BFGS_B'
            },
            'variational_form': {
                'name': 'RYRZ'
            },
            'initial_state': {
                'name': 'ZERO'
            }
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.VQE_CONFIGURATION.copy())
        self._operator = None
        self._operator_mode = 'matrix'
        self._var_form = None
        self._optimizer = None
        self._opt_init_point = None
        self._aux_operators = None
        self._ret = {}
        self._eval_count = 0
        self._eval_time = 0

    def init_params(self, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance

        Args:
            params (dict): parameters dictionary
            algo_input (EnergyInput): EnergyInput instance
        """
        if algo_input is None:
            raise AlgorithmError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        vqe_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        operator_mode = vqe_params.get('operator_mode')
        initial_point = vqe_params.get('initial_point')

        # Set up initial state, we need to add computed num qubits to params
        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        init_state_params['num_qubits'] = operator.num_qubits
        init_state = get_initial_state_instance(init_state_params['name'])
        init_state.init_params(init_state_params)

        # Set up variational form, we need to add computed num qubits, and initial state to params
        var_form_params = params.get(QuantumAlgorithm.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = operator.num_qubits
        var_form_params['initial_state'] = init_state
        var_form = get_variational_form_instance(var_form_params['name'])
        var_form.init_params(var_form_params)

        # Set up optimizer
        opt_params = params.get(QuantumAlgorithm.SECTION_KEY_OPTIMIZER)
        optimizer = get_optimizer_instance(opt_params['name'])
        optimizer.init_params(opt_params)

        if 'statevector' not in self._backend and operator_mode == 'matrix':
            logger.debug('Qasm simulation does not work on {} mode, changing \
                            the operator_mode to paulis'.format(operator_mode))
            operator_mode = 'paulis'

        self.init_args(operator, operator_mode, var_form, optimizer,
                       opt_init_point=initial_point, aux_operators=algo_input.aux_ops)
        logger.info(self.print_setting())

    def init_args(self, operator, operator_mode, var_form, optimizer,
                  opt_init_point=None, aux_operators=[]):
        """
        Args:
            operator (Operator): Qubit operator
            operator_mode (str): operator mode, used for eval of operator
            var_form (VariationalForm) : parametrized variational form.
            optimizer (Optimizer) : the classical optimization algorithm.
            opt_init_point (str) : optimizer initial point.
            aux_operators ([Operator]): Auxiliary operators to be evaluated at each eigenvalue
        """
        self._operator = operator
        self._operator_mode = operator_mode
        self._var_form = var_form
        self._optimizer = optimizer
        self._opt_init_point = opt_init_point
        self._aux_operators = aux_operators
        self._ret = {}
        if opt_init_point is None:
            self._opt_init_point = var_form.preferred_init_points

    @property
    def setting(self):
        ret = "Algorithm: {}\n".format(self._configuration['name'])
        params = ""
        for key, value in self.__dict__.items():
            if key != "_configuration" and key[0] == "_":
                if "opt_init_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                else:
                    params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    def print_setting(self):
        """
        Preparing the setting of VQE into a string.

        Returns:
            str: the formatted setting of VQE
        """
        ret = "\n"
        ret += "==================== Setting of {} ============================\n".format(self.configuration['name'])
        ret += "{}".format(self.setting)
        ret += "===============================================================\n"
        ret += "{}".format(self._var_form.setting)
        ret += "===============================================================\n"
        ret += "{}".format(self._optimizer.setting)
        ret += "===============================================================\n"
        return ret

    def _solve(self):
        opt_params, opt_val = self.find_minimum_eigenvalue()
        self._ret['eigvals'] = np.asarray([opt_val])
        self._ret['opt_params'] = opt_params
        qc = self._var_form.construct_circuit(self._ret['opt_params'])
        if 'statevector' in self._backend:
            ret = self.execute(qc)
            self._ret['eigvecs'] = np.asarray([ret.get_statevector(qc)])
        else:
            c = ClassicalRegister(self._operator.num_qubits, name='c')
            q = qc.get_qregs()['q']
            qc.add(c)
            qc.measure(q, c)
            ret = self.execute(qc)
            self._ret['eigvecs'] = np.asarray([ret.get_counts(qc)])


    def _get_ground_state_energy(self):
        if 'eigvals' not in self._ret:
            self._solve()
        self._ret['energy'] = self._ret['eigvals'][0]

    def _eval_aux_ops(self, threshold=1e-12):
        if 'opt_params' not in self._ret:
            self._get_ground_state_energy()
        wavefn_circuit = self._var_form.construct_circuit(self._ret['opt_params'])
        values = []
        for operator in self._aux_operators:
            mean, std = 0.0, 0.0
            if not operator.is_empty():
                mean, std = operator.eval(self._operator_mode, wavefn_circuit,
                                          self._backend, self._execute_config, self._qjob_config)
                mean = mean.real if abs(mean.real) > threshold else 0.0
                std = std.real if abs(std.real) > threshold else 0.0
            values.append((mean, std))
        if len(values) > 0:
            aux_op_vals = np.empty([1, len(self._aux_operators), 2])
            aux_op_vals[0, :] = np.asarray(values)
            self._ret['aux_ops'] = aux_op_vals

    def run(self):
        """
        Runs the algorithm to compute the minimum eigenvalue

        Returns:
            Dictionary of results
        """
        self._operator.enable_summarize_circuits()
        self._eval_count = 0
        self._solve()
        self._get_ground_state_energy()
        self._eval_aux_ops()
        self._ret['eval_count'] = self._eval_count
        self._ret['eval_time'] = self._eval_time
        return self._ret

    # This is the objective function to be passed to the optimizer that is uses for evaluation
    def _energy_evaluation(self, parameters):
        """
        Evaluate energy at given parameters for the variational form.

        Args:
            parameters (numpy.ndarray) : parameters for variational form.

        Returns:
            Energy of the hamiltonian.
        """
        input_circuit = self._var_form.construct_circuit(parameters)
        mean_energy, std_energy = self._operator.eval(self._operator_mode, input_circuit,
                                                     self._backend, self._execute_config, self._qjob_config)
        self._eval_count += 1

        self._operator.disable_summarize_circuits()
        logger.debug('Energy evaluation {} returned {}'.format(self._eval_count, np.real(mean_energy)))
        return np.real(mean_energy)

    def find_minimum_eigenvalue(self, initial_point=None):
        """Determine minimum energy state.

        Args:
            initial_point (numpy.ndarray[float]) : initial point, or None
                if not provided.

        Returns:
            Optimized variational parameters, and corresponding minimum eigenvalue.

        Raises:
            ValueError:

        """
        initial_point = initial_point if initial_point is not None else self._opt_init_point

        nparms = self._var_form.num_parameters
        bounds = self._var_form.parameter_bounds

        if initial_point is not None and len(initial_point) != nparms:
            raise ValueError('Initial point size {} and parameter size {} mismatch'.format(len(initial_point), nparms))
        if len(bounds) != nparms:
            raise ValueError('Variational form bounds size does not match parameter size')
        # If *any* value is *equal* in bounds array to None then the problem does *not* have bounds
        problem_has_bounds = not np.any(np.equal(bounds, None))
        # Check capabilities of the optimizer
        if problem_has_bounds:
            if not self._optimizer.is_bounds_supported:
                raise ValueError('Problem has bounds but optimizer does not support bounds')
        else:
            if self._optimizer.is_bounds_required:
                raise ValueError('Problem does not have bounds but optimizer requires bounds')
        if initial_point is not None:
            if not self._optimizer.is_initial_point_supported:
                raise ValueError('Optimizer does not support initial point')
        else:
            if self._optimizer.is_initial_point_required:
                low = [(l if l is not None else -2 * np.pi) for (l, u) in bounds]
                high = [(u if u is not None else 2 * np.pi) for (l, u) in bounds]
                initial_point = self.random.uniform(low, high)

        start = time.time()
        logger.info('Starting optimizer bounds={}\ninitial point={}'.format(bounds, initial_point))
        sol, opt, nfev = self._optimizer.optimize(self._var_form.num_parameters, self._energy_evaluation,
                                                  variable_bounds=bounds, initial_point=initial_point)
        if nfev is not None:
            self._eval_count = self._eval_count if self._eval_count >= nfev else nfev
        self._eval_time = time.time() - start
        logger.info('Optimization complete in {}s found {} num evals {}'.format(
            self._eval_time, opt, self._eval_count))

        return sol, opt
