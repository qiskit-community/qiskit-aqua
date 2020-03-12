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

from qiskit.optimization.utils.helpers import NameIndex, init_list_args
from test.optimization.common import QiskitOptimizationTestCase


class TestHelpers(QiskitOptimizationTestCase):
    """Test helpers."""

    def setUp(self):
        super().setUp()

    def test_init_list_args(self):
        a = init_list_args(1, [2], None)
        self.assertTupleEqual(a, (1, [2], []))

    def test_name_index1(self):
        a = NameIndex()
        self.assertEqual(a.convert('1'), 0)
        self.assertListEqual(a.convert(['2', '3']), [1, 2])
        self.assertEqual(a.convert('1'), 0)
        self.assertListEqual(a.convert(), [0, 1, 2])
        self.assertListEqual(a.convert('1', '3'), [0, 1, 2])
        self.assertListEqual(a.convert('1', '2'), [0, 1])

    def test_name_index2(self):
        a = NameIndex()
        a.build(['1', '2', '3'])
        self.assertEqual(a.convert('1'), 0)
        self.assertListEqual(a.convert(), [0, 1, 2])
        self.assertListEqual(a.convert('1', '3'), [0, 1, 2])
        self.assertListEqual(a.convert('1', '2'), [0, 1])
