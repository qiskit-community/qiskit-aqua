# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test helpers """

import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase
from qiskit.optimization.utils.helpers import NameIndex, init_list_args
from qiskit.optimization import QiskitOptimizationError


class TestHelpers(QiskitOptimizationTestCase):
    """Test helpers."""

    def test_init_list_args(self):
        """ test init list args """
        args = init_list_args(1, [2], None)
        self.assertTupleEqual(args, (1, [2], []))

    def test_name_index1(self):
        """ test name index 1 """
        n_i = NameIndex()
        self.assertEqual(n_i.convert('1'), 0)
        self.assertListEqual(n_i.convert(['2', '3']), [1, 2])
        self.assertEqual(n_i.convert('1'), 0)
        self.assertListEqual(n_i.convert(), [0, 1, 2])
        self.assertListEqual(n_i.convert('1', '3'), [0, 1, 2])
        self.assertListEqual(n_i.convert('1', '2'), [0, 1])

    def test_name_index2(self):
        """ test name index 2 """
        n_i = NameIndex()
        n_i.build(['1', '2', '3'])
        self.assertEqual(n_i.convert('1'), 0)
        self.assertListEqual(n_i.convert(), [0, 1, 2])
        self.assertListEqual(n_i.convert('1', '3'), [0, 1, 2])
        self.assertListEqual(n_i.convert('1', '2'), [0, 1])

    def test_name_index3(self):
        """ test name index 3 """
        n_i = NameIndex()
        with self.assertRaises(QiskitOptimizationError):
            n_i.convert({})
        with self.assertRaises(QiskitOptimizationError):
            n_i.convert(1, 2, 3)


if __name__ == '__main__':
    unittest.main()
