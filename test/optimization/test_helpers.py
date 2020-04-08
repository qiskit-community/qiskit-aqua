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

from qiskit.optimization import QiskitOptimizationError
from qiskit.optimization.utils.helpers import NameIndex, init_list_args


class TestHelpers(QiskitOptimizationTestCase):
    """Test helpers."""

    def test_init_list_args(self):
        """ test init list args """
        args = init_list_args(1, [2], None)
        self.assertTupleEqual(args, (1, [2], []))

    def test_name_index1(self):
        """ test name index 1 """
        nidx = NameIndex()
        nidx.build(['1', '2', '3'])
        self.assertEqual(nidx.convert('1'), 0)
        self.assertListEqual(nidx.convert(['2', '3']), [1, 2])
        self.assertEqual(nidx.convert('1'), 0)
        self.assertListEqual(nidx.convert(), [0, 1, 2])
        self.assertListEqual(nidx.convert('1', '3'), [0, 1, 2])
        self.assertListEqual(nidx.convert('1', '2'), [0, 1])

    def test_name_index2(self):
        """ test name index 2 """
        nidx = NameIndex()
        nidx.build(['1', '2', '3'])
        self.assertEqual(nidx.convert('1'), 0)
        self.assertListEqual(nidx.convert(), [0, 1, 2])
        self.assertListEqual(nidx.convert('1', '3'), [0, 1, 2])
        self.assertListEqual(nidx.convert('1', '2'), [0, 1])

    def test_name_index3(self):
        """ test name index 3 """
        nidx = NameIndex()
        with self.assertRaises(QiskitOptimizationError):
            nidx.convert({})
        with self.assertRaises(QiskitOptimizationError):
            nidx.convert(1, 2, 3)
        nidx.build(['x', 'y', 'z'])
        self.assertEqual(nidx.convert(1), 1)
        with self.assertRaises(QiskitOptimizationError):
            nidx.convert(4)
        self.assertEqual(nidx.convert('z'), 2)
        with self.assertRaises(QiskitOptimizationError):
            nidx.convert('a')
            nidx.convert(1, 2, 3)


if __name__ == '__main__':
    unittest.main()
