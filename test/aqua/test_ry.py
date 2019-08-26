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

""" Test RYCRX """

import unittest
from test.aqua.common import QiskitAquaTestCase

import numpy as np
from parameterized import parameterized
from qiskit import BasicAer

from qiskit.aqua import run_algorithm, aqua_globals
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.operators import WeightedPauliOperator


class TestRYCRX(QiskitAquaTestCase):
    """ Test RYCRX """

    def setUp(self):
        super().setUp()
        self.seed = 50
        aqua_globals.random_seed = self.seed
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        qubit_op = WeightedPauliOperator.from_dict(pauli_dict)
        self.algo_input = EnergyInput(qubit_op)

    @parameterized.expand([
        [2, 5],
        [3, 5],
        [4, 5]
    ])
    def test_vqe_var_forms(self, depth, places):
        """ VQE Var Forms test """
        backend = BasicAer.get_backend('statevector_simulator')
        params = {
            'algorithm': {'name': 'VQE'},
            'variational_form': {'name': 'RY',
                                 'depth': depth,
                                 'entanglement': 'sca',
                                 'entanglement_gate': 'crx',
                                 'skip_final_ry': True},
            'backend': {'shots': 1, 'seed_simulator': self.seed, 'seed_transpiler': self.seed}
        }
        result = run_algorithm(params, self.algo_input, backend=backend)
        self.assertAlmostEqual(result['energy'], -1.85727503, places=places)


if __name__ == '__main__':
    unittest.main()
