# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Exact Eigen solver """

import unittest
from test.aqua import QiskitAquaTestCase
import numpy as np
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.operators import WeightedPauliOperator


class TestExactEigensolver(QiskitAquaTestCase):
    """ Test Exact Eigen solver """
    def setUp(self):
        super().setUp()
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        self.qubit_op = WeightedPauliOperator.from_dict(pauli_dict)

    def test_ee(self):
        """ ee test """
        algo = ExactEigensolver(self.qubit_op, k=1, aux_operators=[])
        result = algo.run()
        self.assertAlmostEqual(result['energy'], -1.85727503)
        np.testing.assert_array_almost_equal(result['energies'], [-1.85727503])
        np.testing.assert_array_almost_equal(result['eigvals'], [-1.85727503 + 0j])

    def test_ee_k4(self):
        """ ee k4 test """
        algo = ExactEigensolver(self.qubit_op, k=4, aux_operators=[])
        result = algo.run()
        self.assertAlmostEqual(result['energy'], -1.85727503)
        self.assertEqual(len(result['eigvals']), 4)
        self.assertEqual(len(result['eigvecs']), 4)
        np.testing.assert_array_almost_equal(result['energies'],
                                             [-1.85727503, -1.24458455, -0.88272215, -0.22491125])


if __name__ == '__main__':
    unittest.main()
