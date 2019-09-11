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

""" The Quantum Approximate Optimization Algorithm. """

# pylint: disable=unused-import

import logging
import warnings

from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class
from qiskit.aqua.operators import (WeightedPauliOperator,
                                   TPBGroupedWeightedPauliOperator,
                                   MatrixOperator)
from qiskit.aqua.algorithms.adaptive import VQE
from .var_form import QAOAVarForm

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class QAOA(VQE):
    """
    The Quantum Approximate Optimization Algorithm.

    See https://arxiv.org/abs/1411.4028
    """

    CONFIGURATION = {
        'name': 'QAOA.Variational',
        'description': 'Quantum Approximate Optimization Algorithm',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'qaoa_schema',
            'type': 'object',
            'properties': {
                'p': {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                },
                'initial_point': {
                    'type': ['array', 'null'],
                    "items": {
                        "type": "number"
                    },
                    'default': None
                },
                'max_evals_grouped': {
                    'type': 'integer',
                    'default': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['ising'],
        'depends': [
            {
                'pluggable_type': 'optimizer',
                'default': {
                    'name': 'COBYLA',
                },
            },
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'ZERO',
                },
            },
        ],
    }

    def __init__(self, operator, optimizer, p=1, initial_state=None, mixer=None,
                 initial_point=None, max_evals_grouped=1, aux_operators=None,
                 callback=None, auto_conversion=True):
        """
        Args:
            operator (BaseOperator): Qubit operator
            p (int): the integer parameter p as specified in https://arxiv.org/abs/1411.4028
            initial_state (InitialState): the initial state to prepend the QAOA circuit with
            mixer (BaseOperator): the mixer Hamiltonian to evolve with. Allows support
                              of optimizations in constrained subspaces as
                              specified in https://arxiv.org/abs/1709.03489
            optimizer (Optimizer): the classical optimization algorithm.
            initial_point (numpy.ndarray): optimizer initial point.
            max_evals_grouped (int): max number of evaluations to be performed simultaneously.
            aux_operators (list): aux operators
            callback (Callable): a callback that can access the intermediate
                                 data during the optimization.
                                 Internally, four arguments are provided as follows
                                 the index of evaluation, parameters of variational form,
                                 evaluated mean, evaluated standard deviation.
            auto_conversion (bool): an automatic conversion for operator and aux_operators
                                        into the type which is
                                    most suitable for the backend.
                                    - non-aer statevector_simulator: MatrixOperator
                                    - aer statevector_simulator: WeightedPauliOperator
                                    - qasm simulator or real backend:
                                        TPBGroupedWeightedPauliOperator

        """
        self.validate(locals())
        var_form = QAOAVarForm(operator.copy(), p, initial_state=initial_state,
                               mixer_operator=mixer)
        super().__init__(operator, var_form, optimizer, initial_point=initial_point,
                         max_evals_grouped=max_evals_grouped, aux_operators=aux_operators,
                         callback=callback, auto_conversion=auto_conversion)

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance

        Args:
            params (dict): parameters dictionary
            algo_input (EnergyInput): EnergyInput instance
        Returns:
            QAOA: instance of this class
        Raises:
            AquaError: invalid input
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        operator = algo_input.qubit_op

        qaoa_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        p = qaoa_params.get('p')
        initial_point = qaoa_params.get('initial_point')
        max_evals_grouped = qaoa_params.get('max_evals_grouped')

        init_state_params = params.get(Pluggable.SECTION_KEY_INITIAL_STATE)
        init_state_params['num_qubits'] = operator.num_qubits
        init_state = get_pluggable_class(PluggableType.INITIAL_STATE,
                                         init_state_params['name']).init_params(params)

        # Set up optimizer
        opt_params = params.get(Pluggable.SECTION_KEY_OPTIMIZER)
        optimizer = get_pluggable_class(PluggableType.OPTIMIZER,
                                        opt_params['name']).init_params(params)

        return cls(operator, optimizer, p=p, initial_state=init_state,
                   initial_point=initial_point, max_evals_grouped=max_evals_grouped,
                   aux_operators=algo_input.aux_ops)
