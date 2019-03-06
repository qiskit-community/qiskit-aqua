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

import logging
import functools

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit

from qiskit.aqua.algorithms.adaptive.vqalgorithm import VQAlgorithm
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.utils.backend_utils import is_aer_statevector_backend

logger = logging.getLogger(__name__)


class VQE(VQAlgorithm):
    """
    The Variational Quantum Eigensolver algorithm.

    See https://arxiv.org/abs/1304.3061
    """

    CONFIGURATION = {
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
                },
                'batch_mode': {
                    'type': 'boolean',
                    'default': False
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy', 'ising'],
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

    def __init__(self, operator, var_form, optimizer, operator_mode='matrix',
                 initial_point=None, batch_mode=False, aux_operators=None, callback=None):
        """Constructor.

        Args:
            operator (Operator): Qubit operator
            operator_mode (str): operator mode, used for eval of operator
            var_form (VariationalForm): parametrized variational form.
            optimizer (Optimizer): the classical optimization algorithm.
            initial_point (numpy.ndarray): optimizer initial point.
            batch_mode (bool): evaluate parameter sets in parallel.
            aux_operators (list of Operator): Auxiliary operators to be evaluated at each eigenvalue
            callback (Callable): a callback that can access the intermediate data during the optimization.
                                 Internally, four arguments are provided as follows
                                 the index of evaluation, parameters of variational form,
                                 evaluated mean, evaluated standard devation.
        """
        self.validate(locals())
        super().__init__(var_form=var_form,
                         optimizer=optimizer,
                         cost_fn=self._energy_evaluation,
                         initial_point=initial_point,
                         batch_mode=batch_mode,
                         callback=callback)
        if initial_point is None:
            self._initial_point = var_form.preferred_init_points
        self._operator = operator
        self._operator_mode = operator_mode
        if aux_operators is None:
            self._aux_operators = []
        else:
            self._aux_operators = [aux_operators] if not isinstance(aux_operators, list) else aux_operators
        logger.info(self.print_settings())

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance.

        Args:
            params (dict): parameters dictionary
            algo_input (EnergyInput): EnergyInput instance

        Returns:
            VQE: vqe object
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        vqe_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        operator_mode = vqe_params.get('operator_mode')
        initial_point = vqe_params.get('initial_point')
        batch_mode = vqe_params.get('batch_mode')

        # Set up variational form, we need to add computed num qubits
        # Pass all parameters so that Variational Form can create its dependents
        var_form_params = params.get(Pluggable.SECTION_KEY_VAR_FORM)
        var_form_params['num_qubits'] = operator.num_qubits
        var_form = get_pluggable_class(PluggableType.VARIATIONAL_FORM,
                                       var_form_params['name']).init_params(params)

        # Set up optimizer
        opt_params = params.get(Pluggable.SECTION_KEY_OPTIMIZER)
        optimizer = get_pluggable_class(PluggableType.OPTIMIZER,
                                        opt_params['name']).init_params(params)

        return cls(operator, var_form, optimizer, operator_mode=operator_mode,
                   initial_point=initial_point, batch_mode=batch_mode,
                   aux_operators=algo_input.aux_ops)

    @property
    def setting(self):
        """Prepare the setting of VQE as a string."""
        ret = "Algorithm: {}\n".format(self._configuration['name'])
        params = ""
        for key, value in self.__dict__.items():
            if key != "_configuration" and key[0] == "_":
                if "initial_point" in key and value is None:
                    params += "-- {}: {}\n".format(key[1:], "Random seed")
                else:
                    params += "-- {}: {}\n".format(key[1:], value)
        ret += "{}".format(params)
        return ret

    def print_settings(self):
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

    def construct_circuit(self, parameter, backend=None, use_simulator_operator_mode=False):
        """Generate the circuits.

        Args:
            parameters (numpy.ndarray): parameters for variational form.
            backend (qiskit.BaseBackend): backend object.
            use_simulator_operator_mode (bool): is backend from AerProvider, if True and mode is paulis,
                           single circuit is generated.

        Returns:
            [QuantumCircuit]: the generated circuits with Hamiltonian.
        """
        input_circuit = self._var_form.construct_circuit(parameter)
        if backend is None:
            warning_msg = "Circuits used in VQE depends on the backend type, "
            from qiskit import BasicAer
            if self._operator_mode == 'matrix':
                temp_backend_name = 'statevector_simulator'
            else:
                temp_backend_name = 'qasm_simulator'
            backend = BasicAer.get_backend(temp_backend_name)
            warning_msg += "since operator_mode is '{}', '{}' backend is used.".format(
                self._operator_mode, temp_backend_name)
            logger.warning(warning_msg)
        circuit = self._operator.construct_evaluation_circuit(self._operator_mode,
                                                              input_circuit, backend, use_simulator_operator_mode)
        return circuit

    def _eval_aux_ops(self, threshold=1e-12):
        if 'opt_params' not in self._ret:
            self._get_ground_state_energy()
        wavefn_circuit = self._var_form.construct_circuit(self._ret['opt_params'])
        circuits = []
        values = []
        params = []
        for operator in self._aux_operators:
            if not operator.is_empty():
                temp_circuit = QuantumCircuit() + wavefn_circuit
                circuit = operator.construct_evaluation_circuit(self._operator_mode, temp_circuit,
                                                                self._quantum_instance.backend,
                                                                self._use_simulator_operator_mode)
                params.append(operator.aer_paulis)
            else:
                circuit = None
            circuits.append(circuit)

        if len(circuits) > 0:
            to_be_simulated_circuits = functools.reduce(lambda x, y: x + y, [c for c in circuits if c is not None])
            if self._use_simulator_operator_mode:
                extra_args = {'expectation': {
                    'params': params,
                    'num_qubits': self._operator.num_qubits}
                }
            else:
                extra_args = {}
            result = self._quantum_instance.execute(to_be_simulated_circuits, **extra_args)

            for operator, circuit in zip(self._aux_operators, circuits):
                if circuit is None:
                    mean, std = 0.0, 0.0
                else:
                    mean, std = operator.evaluate_with_result(self._operator_mode,
                                                              circuit, self._quantum_instance.backend,
                                                              result, self._use_simulator_operator_mode)
                mean = mean.real if abs(mean.real) > threshold else 0.0
                std = std.real if abs(std.real) > threshold else 0.0
                values.append((mean, std))
        if len(values) > 0:
            aux_op_vals = np.empty([1, len(self._aux_operators), 2])
            aux_op_vals[0, :] = np.asarray(values)
            self._ret['aux_ops'] = aux_op_vals

    def _run(self):
        """
        Run the algorithm to compute the minimum eigenvalue.

        Returns:
            Dictionary of results
        """
        if not self._quantum_instance.is_statevector and self._operator_mode == 'matrix':
            logger.warning('Qasm simulation does not work on {} mode, changing '
                           'the operator_mode to "paulis"'.format(self._operator_mode))
            self._operator_mode = 'paulis'

        self._use_simulator_operator_mode = \
            is_aer_statevector_backend(self._quantum_instance.backend) \
            and self._operator_mode != 'matrix'

        self._quantum_instance.circuit_summary = True

        self.find_minimum()

        self._ret['energy'] = self.get_optimal_cost()
        self._ret['eigvals'] = np.asarray([self.get_optimal_cost()])
        self._ret['eigvecs'] = np.asarray([self.get_optimal_vector()])
        self._eval_aux_ops()
        return self._ret

    # This is the objective function to be passed to the optimizer that is uses for evaluation
    def _energy_evaluation(self, parameters):
        """
        Evaluate energy at given parameters for the variational form.

        Args:
            parameters (numpy.ndarray): parameters for variational form.

        Returns:
            float or list of float: energy of the hamiltonian of each parameter.
        """
        num_parameter_sets = len(parameters) // self._var_form.num_parameters
        circuits = []
        parameter_sets = np.split(parameters, num_parameter_sets)
        mean_energy = []
        std_energy = []

        for idx in range(len(parameter_sets)):
            parameter = parameter_sets[idx]
            circuit = self.construct_circuit(parameter, self._quantum_instance.backend, self._use_simulator_operator_mode)
            circuits.append(circuit)

        to_be_simulated_circuits = functools.reduce(lambda x, y: x + y, circuits)
        if self._use_simulator_operator_mode:
            extra_args = {'expectation': {
                'params': [self._operator.aer_paulis],
                'num_qubits': self._operator.num_qubits}
            }
        else:
            extra_args = {}
        result = self._quantum_instance.execute(to_be_simulated_circuits, **extra_args)

        for idx in range(len(parameter_sets)):
            mean, std = self._operator.evaluate_with_result(
                self._operator_mode, circuits[idx], self._quantum_instance.backend, result, self._use_simulator_operator_mode)
            mean_energy.append(np.real(mean))
            std_energy.append(np.real(std))
            self._eval_count += 1
            if self._callback is not None:
                self._callback(self._eval_count, parameter_sets[idx], np.real(mean), np.real(std))
            logger.info('Energy evaluation {} returned {}'.format(self._eval_count, np.real(mean)))

        return mean_energy if len(mean_energy) > 1 else mean_energy[0]
