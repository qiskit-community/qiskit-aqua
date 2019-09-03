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
"""
The Quantum Dynamics algorithm.
"""

import logging

from qiskit import QuantumRegister

from qiskit.aqua.algorithms import QuantumAlgorithm
from qiskit.aqua.operators import op_converter
from qiskit.aqua import AquaError, Pluggable, PluggableType, get_pluggable_class

logger = logging.getLogger(__name__)


class EOH(QuantumAlgorithm):
    """
    The Quantum EOH (Evolution of Hamiltonian) algorithm.
    """

    PROP_EVO_TIME = 'evo_time'
    PROP_NUM_TIME_SLICES = 'num_time_slices'
    PROP_EXPANSION_MODE = 'expansion_mode'
    PROP_EXPANSION_ORDER = 'expansion_order'

    CONFIGURATION = {
        'name': 'EOH',
        'description': 'Evolution of Hamiltonian for Quantum Systems',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'EOH_schema',
            'type': 'object',
            'properties': {
                PROP_EVO_TIME: {
                    'type': 'number',
                    'default': 1,
                    'minimum': 0
                },
                PROP_NUM_TIME_SLICES: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 0
                },
                PROP_EXPANSION_MODE: {
                    'type': 'string',
                    'default': 'trotter',
                    'enum': [
                        'trotter',
                        'suzuki'
                    ]
                },
                PROP_EXPANSION_ORDER: {
                    'type': 'integer',
                    'default': 1,
                    'minimum': 1
                }
            },
            'additionalProperties': False
        },
        'problems': ['eoh'],
        'depends': [
            {
                'pluggable_type': 'initial_state',
                'default': {
                    'name': 'ZERO'
                },
            },
        ],
    }

    def __init__(self, operator, initial_state, evo_operator, evo_time=1, num_time_slices=1,
                 expansion_mode='trotter', expansion_order=1):
        self.validate(locals())
        super().__init__()
        self._operator = op_converter.to_weighted_pauli_operator(operator)
        self._initial_state = initial_state
        self._evo_operator = op_converter.to_weighted_pauli_operator(evo_operator)
        self._evo_time = evo_time
        self._num_time_slices = num_time_slices
        self._expansion_mode = expansion_mode
        self._expansion_order = expansion_order
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params (dict): parameters dictionary
            algo_input (EnergyInput): instance
        Returns:
            EOH: and instance of this class
        Raises:
            AquaError: invalid input
        """
        if algo_input is None:
            raise AquaError("EnergyInput instance is required.")

        # For getting the extra operator, caller has
        # to do something like: algo_input.add_aux_op(evo_op)
        operator = algo_input.qubit_op
        aux_ops = algo_input.aux_ops
        if aux_ops is None or len(aux_ops) != 1:
            raise AquaError("EnergyInput, a single aux op is required for evaluation.")
        evo_operator = aux_ops[0]
        if evo_operator is None:
            raise AquaError("EnergyInput, invalid aux op.")

        dynamics_params = params.get(Pluggable.SECTION_KEY_ALGORITHM)
        evo_time = dynamics_params.get(EOH.PROP_EVO_TIME)
        num_time_slices = dynamics_params.get(EOH.PROP_NUM_TIME_SLICES)
        expansion_mode = dynamics_params.get(EOH.PROP_EXPANSION_MODE)
        expansion_order = dynamics_params.get(EOH.PROP_EXPANSION_ORDER)

        # Set up initial state, we need to add computed num qubits to params
        initial_state_params = params.get(Pluggable.SECTION_KEY_INITIAL_STATE)
        initial_state_params['num_qubits'] = operator.num_qubits
        initial_state = get_pluggable_class(PluggableType.INITIAL_STATE,
                                            initial_state_params['name']).init_params(params)

        return cls(operator, initial_state, evo_operator, evo_time=evo_time,
                   num_time_slices=num_time_slices, expansion_mode=expansion_mode,
                   expansion_order=expansion_order)

    def construct_circuit(self):
        """
        Construct the circuit.

        Returns:
            QuantumCircuit: the circuit.
        """
        quantum_registers = QuantumRegister(self._operator.num_qubits, name='q')
        qc = self._initial_state.construct_circuit('circuit', quantum_registers)

        qc += self._evo_operator.evolve(
            evo_time=self._evo_time,
            num_time_slices=self._num_time_slices,
            quantum_registers=quantum_registers,
            expansion_mode=self._expansion_mode,
            expansion_order=self._expansion_order,
        )

        return qc

    def _run(self):
        qc = self.construct_circuit()
        qc_with_op = self._operator.construct_evaluation_circuit(
            wave_function=qc, statevector_mode=self._quantum_instance.is_statevector)
        result = self._quantum_instance.execute(qc_with_op)
        self._ret['avg'], self._ret['std_dev'] = self._operator.evaluate_with_result(
            result=result, statevector_mode=self._quantum_instance.is_statevector)
        return self._ret
