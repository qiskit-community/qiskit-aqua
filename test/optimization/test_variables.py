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

import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase

from cplex import infinity

from qiskit.optimization import QuadraticProgram, QiskitOptimizationError


class TestVariables(QiskitOptimizationTestCase):
    """Test VariablesInterface."""

    def test_type(self):
        """ test type """
        op = QuadraticProgram()
        self.assertEqual(op.variables.type.binary, 'B')
        self.assertEqual(op.variables.type['B'], 'binary')

    def test_initial(self):
        """ test initial """
        op = QuadraticProgram()
        op.variables.add(names=["x0", "x1", "x2"])
        self.assertListEqual(op.variables.get_lower_bounds(), [0.0, 0.0, 0.0])
        op.variables.set_lower_bounds(0, 1.0)
        op.variables.set_lower_bounds([("x1", -1.0), (2, 3.0)])
        self.assertListEqual(op.variables.get_lower_bounds(0, "x1"), [1.0, -1.0])
        self.assertListEqual(op.variables.get_lower_bounds(["x1", "x2", 0]), [-1.0, 3.0, 1.0])
        self.assertEqual(op.variables.get_num(), 3)
        op.variables.set_types(0, op.variables.type.binary)
        self.assertEqual(op.variables.get_num_binary(), 1)

    def test_get_num(self):
        """ test get num """
        op = QuadraticProgram()
        typ = op.variables.type
        op.variables.add(types=[typ.continuous, typ.binary, typ.integer])
        self.assertEqual(op.variables.get_num(), 3)

    def test_get_num_continuous(self):
        """ test get num continuous """
        op = QuadraticProgram()
        typ = op.variables.type
        op.variables.add(types=[typ.continuous, typ.binary, typ.integer])
        self.assertEqual(op.variables.get_num_continuous(), 1)

    def test_get_num_integer(self):
        """ test get num integer """
        op = QuadraticProgram()
        typ = op.variables.type
        op.variables.add(types=[typ.continuous, typ.binary, typ.integer])
        self.assertEqual(op.variables.get_num_integer(), 1)

    def test_get_num_binary(self):
        """ test get num binary """
        op = QuadraticProgram()
        typ = op.variables.type
        op.variables.add(types=[typ.semi_continuous, typ.binary, typ.integer])
        self.assertEqual(op.variables.get_num_binary(), 1)

    def test_get_num_semicontinuous(self):
        """ test get num semi continuous """
        op = QuadraticProgram()
        typ = op.variables.type
        op.variables.add(types=[typ.semi_continuous, typ.semi_integer, typ.semi_integer])
        self.assertEqual(op.variables.get_num_semicontinuous(), 1)

    def test_get_num_semiinteger(self):
        """ test get num semi integer """
        op = QuadraticProgram()
        typ = op.variables.type
        op.variables.add(types=[typ.semi_continuous, typ.semi_integer, typ.semi_integer])
        self.assertEqual(op.variables.get_num_semiinteger(), 2)

    def test_add(self):
        """ add test """
        op = QuadraticProgram()
        op.linear_constraints.add(names=["c0", "c1", "c2"])
        op.variables.add(types=[op.variables.type.integer] * 3)
        op.variables.add(
            lb=[-1.0, 1.0, 0.0],
            ub=[100.0, infinity, infinity],
            types=[op.variables.type.integer] * 3,
            names=["0", "1", "2"]
        )
        self.assertListEqual(
            op.variables.get_lower_bounds(),
            [0.0, 0.0, 0.0, -1.0, 1.0, 0.0])
        self.assertListEqual(
            op.variables.get_upper_bounds(),
            [infinity, infinity, infinity, 100.0, infinity, infinity])

    def test_delete(self):
        """ test delete """
        op = QuadraticProgram()
        op.variables.add(names=[str(i) for i in range(10)])
        self.assertEqual(op.variables.get_num(), 10)
        self.assertListEqual(op.variables.get_names(),
                             ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        op.variables.delete(8)
        self.assertListEqual(op.variables.get_names(),
                             ['0', '1', '2', '3', '4', '5', '6', '7', '9'])
        op.variables.delete("1", 3)
        self.assertListEqual(op.variables.get_names(), ['0', '4', '5', '6', '7', '9'])
        op.variables.delete([2, '0', 5])
        self.assertListEqual(op.variables.get_names(), ['4', '6', '7'])
        op.variables.delete()
        self.assertListEqual(op.variables.get_names(), [])

    def test_set_lower_bounds(self):
        """ test set lower bounds """
        op = QuadraticProgram()
        op.variables.add(names=["x0", "x1", "x2"])
        op.variables.set_lower_bounds(0, 1.0)
        self.assertListEqual(op.variables.get_lower_bounds(), [1.0, 0.0, 0.0])
        op.variables.set_lower_bounds([(2, 3.0), ("x1", -1.0)])
        self.assertListEqual(op.variables.get_lower_bounds(), [1.0, -1.0, 3.0])

    def test_set_upper_bounds(self):
        """ test set upper bounds """
        op = QuadraticProgram()
        op.variables.add(names=["x0", "x1", "x2"])
        op.variables.set_upper_bounds(0, 1.0)
        op.variables.set_upper_bounds([("x1", 10.0), (2, 3.0)])
        self.assertListEqual(op.variables.get_upper_bounds(), [1.0, 10.0, 3.0])

    def test_set_names(self):
        """ test set names """
        op = QuadraticProgram()
        typ = op.variables.type
        op.variables.add(types=[typ.continuous, typ.binary, typ.integer])
        op.variables.set_names(0, "first")
        op.variables.set_names([(2, "third"), (1, "second")])
        self.assertListEqual(op.variables.get_names(), ['first', 'second', 'third'])

    def test_set_types(self):
        """ test set types """
        op = QuadraticProgram()
        op.variables.add(names=[str(i) for i in range(5)])
        op.variables.set_types(0, op.variables.type.continuous)
        op.variables.set_types([("1", op.variables.type.integer),
                                ("2", op.variables.type.binary),
                                ("3", op.variables.type.semi_continuous),
                                ("4", op.variables.type.semi_integer)])
        self.assertListEqual(op.variables.get_types(), ['C', 'I', 'B', 'S', 'N'])
        self.assertEqual(op.variables.type[op.variables.get_types(0)], 'continuous')

    def test_get_lower_bounds(self):
        """ test get lower bounds """
        op = QuadraticProgram()
        op.variables.add(lb=[1.5 * i for i in range(10)],
                         names=[str(i) for i in range(10)])
        self.assertEqual(op.variables.get_num(), 10)
        self.assertEqual(op.variables.get_lower_bounds(8), 12.0)
        self.assertListEqual(op.variables.get_lower_bounds('1', 3), [1.5, 3.0, 4.5])
        self.assertListEqual(op.variables.get_lower_bounds([2, "0", 5]), [3.0, 0.0, 7.5])
        self.assertEqual(op.variables.get_lower_bounds(),
                         [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5])

    def test_get_upper_bounds(self):
        """ test get upper bounds """
        op = QuadraticProgram()
        op.variables.add(ub=[(1.5 * i) + 1.0 for i in range(10)],
                         names=[str(i) for i in range(10)])
        self.assertEqual(op.variables.get_num(), 10)
        self.assertEqual(op.variables.get_upper_bounds(8), 13.0)
        self.assertListEqual(op.variables.get_upper_bounds('1', 3), [2.5, 4.0, 5.5])
        self.assertListEqual(op.variables.get_upper_bounds([2, "0", 5]), [4.0, 1.0, 8.5])
        self.assertListEqual(op.variables.get_upper_bounds(),
                             [1.0, 2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5, 13.0, 14.5])

    def test_get_names(self):
        """ test get names """
        op = QuadraticProgram()
        op.variables.add(names=['x' + str(i) for i in range(10)])
        self.assertEqual(op.variables.get_num(), 10)
        self.assertEqual(op.variables.get_names(8), 'x8')
        self.assertListEqual(op.variables.get_names(1, 3), ['x1', 'x2', 'x3'])
        self.assertListEqual(op.variables.get_names([2, 0, 5]), ['x2', 'x0', 'x5'])
        self.assertListEqual(op.variables.get_names(),
                             ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'])

    def test_set_types2(self):
        """ test set types 2 """
        op = QuadraticProgram()
        typ = op.variables.type
        op.variables.add(names=[str(i) for i in range(5)],
                         types=[typ.continuous, typ.integer,
                                typ.binary, typ.semi_continuous, typ.semi_integer])
        self.assertEqual(op.variables.get_num(), 5)
        self.assertEqual(op.variables.get_types(3), 'S')

        types = op.variables.get_types(1, 3)
        self.assertListEqual(types, ['I', 'B', 'S'])

        types = op.variables.get_types([2, 0, 4])
        self.assertListEqual(types, ['B', 'C', 'N'])

        types = op.variables.get_types()
        self.assertEqual(types, ['C', 'I', 'B', 'S', 'N'])

    def test_get_cols(self):
        """ test get cols """
        op = QuadraticProgram()
        with self.assertRaises(NotImplementedError):
            op.variables.get_cols()

    def test_get_obj(self):
        """ test get obj """
        op = QuadraticProgram()
        with self.assertRaises(NotImplementedError):
            op.variables.get_obj()

    def test_get_indices(self):
        """ test get indices """
        op = QuadraticProgram()
        op.variables.add(names=['a', 'b'])
        self.assertEqual(op.variables.get_indices('a'), 0)
        self.assertListEqual(op.variables.get_indices(['a', 'b']), [0, 1])

    def test_add2(self):
        """ test add2 """
        op = QuadraticProgram()
        op.variables.add(names=['x'])
        self.assertEqual(op.variables.get_indices('x'), 0)
        self.assertListEqual(op.variables.get_indices(), [0])
        op.variables.add(names=['y'])
        self.assertEqual(op.variables.get_indices('x'), 0)
        self.assertEqual(op.variables.get_indices('y'), 1)
        self.assertListEqual(op.variables.get_indices(), [0, 1])

    def test_default_bounds(self):
        """  test default bounds """
        op = QuadraticProgram()
        types = ['B', 'I', 'C', 'S', 'N']
        op.variables.add(names=types, types=types)
        self.assertListEqual(op.variables.get_lower_bounds(), [0.0] * 5)
        # the upper bound of binary variable is 1.
        self.assertListEqual(op.variables.get_upper_bounds(), [1.0] + [infinity] * 4)

    def test_empty_names(self):
        """ test empty names """
        op = QuadraticProgram()
        r = op.variables.add(names=['', '', ''])
        self.assertEqual(r, range(0, 3))
        self.assertListEqual(op.variables.get_names(), ['x1', 'x2', 'x3'])
        r = op.variables.add(names=['a', '', 'c'])
        self.assertEqual(r, range(3, 6))
        self.assertListEqual(op.variables.get_names(),
                             ['x1', 'x2', 'x3', 'a', 'x5', 'c'])

    def test_duplicate_names(self):
        """ test duplicate names """
        op = QuadraticProgram()
        op.variables.add(names=['', '', ''])
        with self.assertRaises(QiskitOptimizationError):
            op.variables.add(names=['a', 'a'])
        with self.assertRaises(QiskitOptimizationError):
            op.variables.add(names=['x1'])

    def test_add3(self):
        """ test add 3 """
        op = QuadraticProgram()
        op.variables.add(names=['c'], types=['C'], lb=[-10], ub=[10])
        op.variables.add(names=['b'], types=['B'], lb=[-10], ub=[10])
        op.variables.add(names=['b2'], types=['B'])
        self.assertListEqual(op.variables.get_lower_bounds(), [-10, -10, 0])
        self.assertListEqual(op.variables.get_upper_bounds(), [10, 10, 1])


if __name__ == '__main__':
    unittest.main()
