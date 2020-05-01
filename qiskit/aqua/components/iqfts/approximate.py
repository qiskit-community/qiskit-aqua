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

"""An approximate IQFT."""

from qiskit.aqua.circuits import FourierTransformCircuits as ftc
from . import IQFT


class Approximate(IQFT):
    """An approximate IQFT."""

    CONFIGURATION = {
        'name': 'APPROXIMATE',
        'description': 'Approximate inverse QFT',
        'input_schema': {
            '$schema': 'http://json-schema.org/draft-07/schema#',
            'id': 'aiqft_schema',
            'type': 'object',
            'properties': {
                'degree': {
                    'type': 'integer',
                    'default': 0,
                    'minimum': 0
                },
            },
            'additionalProperties': False
        }
    }

    def __init__(self, num_qubits, degree=0):
        self.validate(locals())
        super().__init__()
        self._num_qubits = num_qubits
        self._degree = degree

    def _build_circuit(self, qubits=None, circuit=None, do_swaps=True):
        return ftc.construct_circuit(
            circuit=circuit,
            qubits=qubits,
            inverse=True,
            approximation_degree=self._degree,
            do_swaps=do_swaps
        )

    def _build_matrix(self):
        raise NotImplementedError
