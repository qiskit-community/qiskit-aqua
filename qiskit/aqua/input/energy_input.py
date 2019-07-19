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

from qiskit.aqua import AquaError, Operator
from qiskit.aqua.input import AlgorithmInput


class EnergyInput(AlgorithmInput):

    PROP_KEY_QUBITOP = 'qubit_op'
    PROP_KEY_AUXOPS = 'aux_ops'

    CONFIGURATION = {
        'name': 'EnergyInput',
        'description': 'Energy problem input',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'energy_state_schema',
            'type': 'object',
            'properties': {
                PROP_KEY_QUBITOP: {
                    'type': 'object',
                    'default': {}
                },
                PROP_KEY_AUXOPS: {
                    'type': ['array', 'null'],
                    'default': None
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy', 'excited_states', 'eoh', 'ising']
    }

    def __init__(self, qubit_op, aux_ops=None):
        self.validate(locals())
        super().__init__()
        self._qubit_op = qubit_op
        self._aux_ops = aux_ops if aux_ops is not None else []

    @property
    def qubit_op(self):
        return self._qubit_op

    @qubit_op.setter
    def qubit_op(self, qubit_op):
        self._qubit_op = qubit_op

    @property
    def aux_ops(self):
        return self._aux_ops

    def validate(self, args_dict):
        params = {}
        for key, value in args_dict.items():
            if key == EnergyInput.PROP_KEY_QUBITOP:
                value = value.save_to_dict() if value is not None else {}
            elif key == EnergyInput.PROP_KEY_AUXOPS:
                value = [value[i].save_to_dict() for i in range(len(value))] if value is not None else None

            params[key] = value

        super().validate(params)

    def add_aux_op(self, aux_op):
        self._aux_ops.append(aux_op)

    def has_aux_ops(self):
        return len(self._aux_ops) > 0

    def to_params(self):
        params = {}
        params[EnergyInput.PROP_KEY_QUBITOP] = self._qubit_op.save_to_dict()
        params[EnergyInput.PROP_KEY_AUXOPS] = [self._aux_ops[i].save_to_dict() for i in range(len(self._aux_ops))]
        return params

    @classmethod
    def from_params(cls, params):
        if EnergyInput.PROP_KEY_QUBITOP not in params:
            raise AquaError("Qubit operator is required.")
        qparams = params[EnergyInput.PROP_KEY_QUBITOP]
        qubit_op = Operator.load_from_dict(qparams)
        if EnergyInput.PROP_KEY_AUXOPS in params:
            auxparams = params[EnergyInput.PROP_KEY_AUXOPS]
            aux_ops = [Operator.load_from_dict(auxparams[i]) for i in range(len(auxparams))]
        return cls(qubit_op, aux_ops)
