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


from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from qiskit.tools.qi.pauli import Pauli
from functools import reduce
from qiskit_acqua import QuantumAlgorithm, Operator, AlgorithmError, get_optimizer_instance
from qiskit_acqua.vqe.vqe import VQE
import logging


logger = logging.getLogger(__name__)


class QAOA(VQE):
    """
    The Quantum Approximate Optimization Algorithm.
    See https://arxiv.org/abs/1411.4028
    """
    class VarForm:
        def __init__(self, cost_operator, p):
            self.cost_operator = cost_operator
            self.p = p
            self.num_parameters = 2 * p
            self.parameter_bounds = [(0, np.pi)] * p + [(0, 2 * np.pi)] * p
            self.preferred_init_points = [0] * p * 2

            # prepare the mixer operator
            v = np.zeros(self.cost_operator.num_qubits)
            ws = np.eye(self.cost_operator.num_qubits)
            self.mixer_operator = reduce(
                lambda x, y: x + y,
                [
                    Operator([[1, Pauli(v, ws[i, :])]])
                    for i in range(self.cost_operator.num_qubits)
                ]
            )

        def construct_circuit(self, angles):
            if not len(angles) == self.num_parameters:
                raise ValueError('Incorrect number of angles: expecting {}, but {} given.'.format(
                    self.num_parameters, len(angles)
                ))
            q = QuantumRegister(self.cost_operator.num_qubits, name='q')
            qc = QuantumCircuit(q)
            qc.u2(0, np.pi, q)
            for idx in range(self.p):
                beta, gamma = angles[idx], angles[idx + self.p]
                qc += self.cost_operator.evolve(None, gamma, 'circuit', 1, quantum_registers=q)
                qc += self.mixer_operator.evolve(None, beta, 'circuit', 1, quantum_registers=q)
            return qc

    PROP_OPERATOR_MODE = 'operator_mode'
    PROP_P = 'p'
    PROP_INIT_POINT = 'initial_point'

    QAOA_CONFIGURATION = {
        'name': 'QAOA',
        'description': 'Quantum Approximate Optimization Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'qaoa_schema',
            'type': 'object',
            'properties': {
                PROP_OPERATOR_MODE: {
                    'type': 'string',
                    'default': 'paulis',
                    'oneOf': [
                        {'enum': ['matrix', 'paulis', 'grouped_paulis']}
                    ]
                },
                PROP_P: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                PROP_INIT_POINT: {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
            },
            'additionalProperties': False
        },
        'problems': ['ising'],
        'depends': ['optimizer'],
        'defaults': {
            'optimizer': {
                'name': 'COBYLA'
            },
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.QAOA_CONFIGURATION.copy())

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


        qaoa_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        operator_mode = qaoa_params.get(QAOA.PROP_OPERATOR_MODE)
        p = qaoa_params.get(QAOA.PROP_P)
        initial_point = qaoa_params.get(QAOA.PROP_INIT_POINT)

        # Set up optimizer
        opt_params = params.get(QuantumAlgorithm.SECTION_KEY_OPTIMIZER)
        optimizer = get_optimizer_instance(opt_params['name'])
        optimizer.init_params(opt_params)

        if 'statevector' not in self._backend and operator_mode == 'matrix':
            logger.debug('Qasm simulation does not work on {} mode, changing \
                            the operator_mode to paulis'.format(operator_mode))
            operator_mode = 'paulis'

        self.init_args(operator, operator_mode, p, optimizer,
                       opt_init_point=initial_point, aux_operators=algo_input.aux_ops)

    def init_args(self, operator, operator_mode, p, optimizer, opt_init_point=None, aux_operators=[]):
        """
        Args:
            operator (Operator): Qubit operator
            operator_mode (str): operator mode, used for eval of operator
            p (int) : the integer parameter p as specified in https://arxiv.org/abs/1411.4028
            optimizer (Optimizer) : the classical optimization algorithm.
            opt_init_point (str) : optimizer initial point.
        """
        var_form = QAOA.VarForm(operator, p)
        super().init_args(operator, operator_mode, var_form, optimizer, opt_init_point=opt_init_point)
