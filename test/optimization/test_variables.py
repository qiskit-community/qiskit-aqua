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

""" Test VariablesInterface """

import numpy as np
import os.path
import tempfile
from cplex import SparsePair
import qiskit.optimization.problems.optimization_problem
from cplex import infinity
from test.optimization.common import QiskitOptimizationTestCase
from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError


class TestVariables(QiskitOptimizationTestCase):
    """Test VariablesInterface."""

    def setUp(self):
        super().setUp()

    def test_type(self):
        op = qiskit.optimization.OptimizationProblem()
        op.variables.type.binary
        op.variables.type['B']

    def test_initial(self):
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(names=["x0", "x1", "x2"])
        # default values for lower_bounds are 0.0
        self.assertAlmostEqual(sum(op.variables.get_lower_bounds()), 0.0)
        # values can be set either one at a time or many at a time
        op.variables.set_lower_bounds(0, 1.0)
        self.assertAlmostEqual(sum(op.variables.get_lower_bounds()), 1.0)
        op.variables.set_lower_bounds([("x1", -1.0), (2, 3.0)])
        self.assertAlmostEqual(sum(op.variables.get_lower_bounds()), 3.0)
        # values can be queried as a sequence in arbitrary order
        self.assertAlmostEqual(op.variables.get_lower_bounds(["x1", "x2", 0])[0], -1.0)
        self.assertAlmostEqual(op.variables.get_lower_bounds(["x1", "x2", 0])[1], 3.0)
        self.assertAlmostEqual(op.variables.get_lower_bounds(["x1", "x2", 0])[2], 1.0)
        # can query the number of variables
        self.assertEqual(op.variables.get_num(), 3)
        op.variables.set_types(0, op.variables.type.binary)
        self.assertEqual(op.variables.get_num_binary(), 1)

    def test_get_num(self):
        op = qiskit.optimization.OptimizationProblem()
        t = op.variables.type
        op.variables.add(types=[t.continuous, t.binary, t.integer])
        self.assertEqual(op.variables.get_num(), 3)

    def test_get_num_continuous(self):
        op = qiskit.optimization.OptimizationProblem()
        t = op.variables.type
        op.variables.add(types=[t.continuous, t.binary, t.integer])
        self.assertEqual(op.variables.get_num_continuous(), 1)

    def test_get_num_integer(self):
        op = qiskit.optimization.OptimizationProblem()
        t = op.variables.type
        op.variables.add(types=[t.continuous, t.binary, t.integer])
        self.assertEqual(op.variables.get_num_integer(), 1)

    def test_get_num_binary(self):
        op = qiskit.optimization.OptimizationProblem()
        t = op.variables.type
        op.variables.add(types=[t.semi_continuous, t.binary, t.integer])
        self.assertEqual(op.variables.get_num_binary(), 1)

    def test_get_num_semicontinuous(self):
        op = qiskit.optimization.OptimizationProblem()
        t = op.variables.type
        op.variables.add(types=[t.semi_continuous, t.semi_integer, t.semi_integer])
        self.assertEqual(op.variables.get_num_semicontinuous(), 1)

    def test_get_num_semiinteger(self):
        op = qiskit.optimization.OptimizationProblem()
        t = op.variables.type
        op.variables.add(types=[t.semi_continuous, t.semi_integer, t.semi_integer])
        self.assertEqual(op.variables.get_num_semiinteger(), 2)

    def test_delete1(self):
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(names=[str(i) for i in range(10)])
        self.assertEqual(op.variables.get_num(), 10)
        op.variables.delete(8)
        self.assertEqual(len(op.variables.get_names()), 9)
        self.assertEqual(op.variables.get_names(), ['0', '1', '2', '3', '4', '5', '6', '7', '9'])
        op.variables.delete([2, "0", 5])
        self.assertEqual(len(op.variables.get_names()), 6)
        self.assertEqual(op.variables.get_names(), ['1', '3', '4', '5', '6', '9'])
        op.variables.delete()
        self.assertEqual(len(op.variables.get_names()), 0)

    def test_lower_bounds(self):
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(names=["x0", "x1", "x2"])
        op.variables.set_lower_bounds(0, 1.0)
        op.variables.get_lower_bounds()
        # [1.0, 0.0, 0.0]
        self.assertEqual(len(op.variables.get_lower_bounds()), 3)
        self.assertAlmostEqual(sum(op.variables.get_lower_bounds()), 1.0)
        op.variables.set_lower_bounds([(2, 3.0), ("x1", -1.0)])
        op.variables.get_lower_bounds()
        # [1.0, -1.0, 3.0]
        self.assertEqual(len(op.variables.get_lower_bounds()), 3)
        self.assertAlmostEqual(sum(op.variables.get_lower_bounds()), 3.0)

    def test_upper_bounds(self):
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(names=["x0", "x1", "x2"])
        op.variables.set_upper_bounds(0, 1.0)
        op.variables.set_upper_bounds([("x1", 10.0), (2, 3.0)])
        # [1.0, 10.0, 3.0]
        self.assertEqual(len(op.variables.get_upper_bounds()), 3)
        self.assertAlmostEqual(sum(op.variables.get_upper_bounds()), 14.0)

    def test_names(self):
        op = qiskit.optimization.OptimizationProblem()
        t = op.variables.type
        op.variables.add(types=[t.continuous, t.binary, t.integer])
        op.variables.set_names(0, "first")
        op.variables.set_names([(2, "third"), (1, "second")])
        self.assertEqual(len(op.variables.get_names()), 3)
        # ['first', 'second', 'third']

    def test_lower_bounds1(self):
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(names=[str(i) for i in range(5)])
        op.variables.set_types(0, op.variables.type.continuous)
        op.variables.set_types([("1", op.variables.type.integer),
                                ("2", op.variables.type.binary),
                                ("3", op.variables.type.semi_continuous),
                                ("4", op.variables.type.semi_integer)])
        self.assertEqual(len(op.variables.get_types()), 5)
        # ['C', 'I', 'B', 'S', 'N']
        self.assertEqual(op.variables.get_types(0), 'C')

    def test_lower_bounds2(self):
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(lb=[1.5 * i for i in range(10)],
                         names=[str(i) for i in range(10)])
        self.assertEqual(op.variables.get_num(), 10)
        self.assertAlmostEqual(op.variables.get_lower_bounds(8), 12.0)
        self.assertEqual(len(op.variables.get_lower_bounds([2, "0", 5])), 3)
        self.assertAlmostEqual(op.variables.get_lower_bounds([2, "0", 5])[0], 3.0)
        self.assertAlmostEqual(op.variables.get_lower_bounds([2, "0", 5])[1], 0.0)
        self.assertAlmostEqual(op.variables.get_lower_bounds([2, "0", 5])[2], 7.5)
        self.assertEqual(len(op.variables.get_lower_bounds()), 10)
        # [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]
        self.assertAlmostEqual(op.variables.get_lower_bounds()[0], 0.0)

    def test_upper_bounds2(self):
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(ub=[(1.5 * i) + 1.0 for i in range(10)],
                         names=[str(i) for i in range(10)])
        self.assertEqual(op.variables.get_num(), 10)
        self.assertAlmostEqual(op.variables.get_upper_bounds(8), 13.0)
        self.assertEqual(len(op.variables.get_upper_bounds([2, "0", 5])), 3)
        self.assertAlmostEqual(op.variables.get_upper_bounds([2, "0", 5])[0], 4.0)
        self.assertEqual(len(op.variables.get_upper_bounds()), 10)
        # [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0, 14.5]
        self.assertAlmostEqual(op.variables.get_upper_bounds()[0], 1.0)

    def test_names2(self):
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(names=['x' + str(i) for i in range(10)])
        self.assertAlmostEqual(op.variables.get_num(), 10)
        self.assertEqual(op.variables.get_names(8), 'x8')
        self.assertEqual(len(op.variables.get_names([2, 0, 5])), 3)
        self.assertEqual(op.variables.get_names([2, 0, 5])[0], 'x2')
        # ['x2', 'x0', 'x5']
        self.assertEqual(len(op.variables.get_names()), 10)
        # ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9']

    def test_types(self):
        op = qiskit.optimization.OptimizationProblem()
        t = op.variables.type
        op.variables.add(names=[str(i) for i in range(5)],
                         types=[t.continuous, t.integer,
                                t.binary, t.semi_continuous, t.semi_integer])
        self.assertEqual(op.variables.get_num(), 5)
        self.assertEqual(op.variables.get_types(3), 'S')
        types = op.variables.get_types([2,0,4])
        # ['B', 'C', 'N']
        self.assertEqual(len(types), 3)
        self.assertEqual(types[0], 'B')
        self.assertEqual(types[1], 'C')
        self.assertEqual(types[2], 'N')

        types = op.variables.get_types()
        #['C', 'I', 'B', 'S', 'N']
        self.assertEqual(len(types), 5)
        self.assertEqual(types[0], 'C')
        self.assertEqual(types[1], 'I')
        self.assertEqual(types[2], 'B')
        self.assertEqual(types[3], 'S')
        self.assertEqual(types[4], 'N')

    def test_cols(self):
        op = qiskit.optimization.OptimizationProblem()
        with self.assertRaises(QiskitOptimizationError):
            op.variables.get_cols()

    def test_obj(self):
        op = qiskit.optimization.OptimizationProblem()
        with self.assertRaises(QiskitOptimizationError):
            op.variables.get_obj()
