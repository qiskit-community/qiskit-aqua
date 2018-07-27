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

from qiskit_aqua import QuantumAlgorithm, AlgorithmError, get_optimizer_instance
from qiskit_aqua.vqe.vqe import VQE
from .varform import QAOAVarForm


logger = logging.getLogger(__name__)


class QAOA(VQE):
    """
    The Quantum Approximate Optimization Algorithm.
    See https://arxiv.org/abs/1411.4028
    """

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
                    'default': 'matrix',
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
        var_form = QAOAVarForm(operator, p)
        super().init_args(operator, operator_mode, var_form, optimizer, opt_init_point=opt_init_point)
