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

""" Test LinearConstraintInterface """

import unittest
from cplex import SparsePair

from qiskit.optimization import OptimizationProblem
from test.optimization.optimization_test_case import QiskitOptimizationTestCase


class TestLinearConstraints(QiskitOptimizationTestCase):
    """Test LinearConstraintInterface."""

    def setUp(self):
        super().setUp()

    def test_get_num(self):
        """ test get num """
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c1", "c2", "c3"])
        self.assertEqual(op.linear_constraints.get_num(), 3)

    def test_add(self):
        """ test add """
        op = OptimizationProblem()
        op.variables.add(names=["x1", "x2", "x3"])
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=["x1", "x3"], val=[1.0, -1.0]),
                      SparsePair(ind=["x1", "x2"], val=[1.0, 1.0]),
                      SparsePair(ind=["x1", "x2", "x3"], val=[-1.0] * 3),
                      SparsePair(ind=["x2", "x3"], val=[10.0, -2.0])],
            senses=["E", "L", "G", "R"],
            rhs=[0.0, 1.0, -1.0, 2.0],
            range_values=[0.0, 0.0, 0.0, -10.0],
            names=["c0", "c1", "c2", "c3"])
        self.assertListEqual(op.linear_constraints.get_rhs(), [0.0, 1.0, -1.0, 2.0])

    def test_delete(self):
        """ test delete """
        op = OptimizationProblem()
        op.linear_constraints.add(names=[str(i) for i in range(10)])
        self.assertEqual(op.linear_constraints.get_num(), 10)
        op.linear_constraints.delete(8)
        self.assertListEqual(op.linear_constraints.get_names(),
                             ['0', '1', '2', '3', '4', '5', '6', '7', '9'])
        op.linear_constraints.delete("1", 3)
        self.assertListEqual(op.linear_constraints.get_names(), ['0', '4', '5', '6', '7', '9'])
        op.linear_constraints.delete([2, "0", 5])
        self.assertListEqual(op.linear_constraints.get_names(), ['4', '6', '7'])
        op.linear_constraints.delete()
        self.assertListEqual(op.linear_constraints.get_names(), [])

    def test_rhs(self):
        """ test rhs """
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        self.assertListEqual(op.linear_constraints.get_rhs(), [0.0, 0.0, 0.0, 0.0])
        op.linear_constraints.set_rhs("c1", 1.0)
        self.assertListEqual(op.linear_constraints.get_rhs(), [0.0, 1.0, 0.0, 0.0])
        op.linear_constraints.set_rhs([("c3", 2.0), (2, -1.0)])
        self.assertListEqual(op.linear_constraints.get_rhs(), [0.0, 1.0, -1.0, 2.0])

    def test_set_names(self):
        """ test set names """
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        op.linear_constraints.set_names("c1", "second")
        self.assertEqual(op.linear_constraints.get_names(1), 'second')
        op.linear_constraints.set_names([("c3", "last"), (2, "middle")])
        self.assertListEqual(op.linear_constraints.get_names(), ['c0', 'second', 'middle', 'last'])

    def test_set_senses(self):
        """ test set senses """
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        self.assertListEqual(op.linear_constraints.get_senses(), ['E', 'E', 'E', 'E'])
        op.linear_constraints.set_senses("c1", "G")
        self.assertEqual(op.linear_constraints.get_senses(1), 'G')
        op.linear_constraints.set_senses([("c3", "L"), (2, "R")])
        self.assertListEqual(op.linear_constraints.get_senses(), ['E', 'G', 'R', 'L'])

    def test_set_linear_components(self):
        """ test set linear components """
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        op.variables.add(names=["x0", "x1"])
        op.linear_constraints.set_linear_components("c0", [["x0"], [1.0]])
        sp = op.linear_constraints.get_rows("c0")
        self.assertListEqual(sp.ind, [0])
        self.assertListEqual(sp.val, [1.0])
        op.linear_constraints.set_linear_components(
            [("c3", SparsePair(ind=["x1"], val=[-1.0])),
             (2, [[0, 1], [-2.0, 3.0]])]
        )
        sp = op.linear_constraints.get_rows("c3")
        self.assertListEqual(sp.ind, [1])
        self.assertListEqual(sp.val, [-1.0])
        sp = op.linear_constraints.get_rows(2)
        self.assertListEqual(sp.ind, [0, 1])
        self.assertListEqual(sp.val, [-2.0, 3.0])

    def test_set_range_values(self):
        """ test set range values """
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        op.linear_constraints.set_range_values("c1", 1.0)
        self.assertListEqual(op.linear_constraints.get_range_values(), [0.0, 1.0, 0.0, 0.0])
        op.linear_constraints.set_range_values([("c3", 2.0), (2, -1.0)])
        self.assertListEqual(op.linear_constraints.get_range_values(), [0.0, 1.0, -1.0, 2.0])

    def test_set_coeffients(self):
        """ test set coefficients """
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        op.variables.add(names=["x0", "x1"])
        op.linear_constraints.set_coefficients("c0", "x1", 1.0)
        sp = op.linear_constraints.get_rows(0)
        self.assertListEqual(sp.ind, [1])
        self.assertListEqual(sp.val, [1.0])
        op.linear_constraints.set_coefficients([("c2", "x0", 2.0),
                                                ("c2", "x1", -1.0)])
        sp = op.linear_constraints.get_rows("c2")
        self.assertListEqual(sp.ind, [0, 1])
        self.assertListEqual(sp.val, [2.0, -1.0])

    def test_get_rhs(self):
        """ test get rhs """
        op = OptimizationProblem()
        op.linear_constraints.add(rhs=[1.5 * i for i in range(10)],
                                  names=[str(i) for i in range(10)])
        self.assertEqual(op.linear_constraints.get_num(), 10)
        self.assertAlmostEqual(op.linear_constraints.get_rhs(8), 12.0)
        self.assertListEqual(op.linear_constraints.get_rhs([2, "0", 5]), [3.0, 0.0, 7.5])
        self.assertEqual(op.linear_constraints.get_rhs(),
                         [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5])

    def test_get_senses(self):
        """ test get senses """
        op = OptimizationProblem()
        op.linear_constraints.add(
            senses=["E", "G", "L", "R"],
            names=[str(i) for i in range(4)])
        self.assertEqual(op.linear_constraints.get_num(), 4)
        self.assertEqual(op.linear_constraints.get_senses(1), 'G')
        self.assertListEqual(op.linear_constraints.get_senses("1", 3), ['G', 'L', 'R'])
        self.assertListEqual(op.linear_constraints.get_senses([2, "0", 1]), ['L', 'E', 'G'])
        self.assertListEqual(op.linear_constraints.get_senses(), ['E', 'G', 'L', 'R'])

    def test_get_range_values(self):
        """ test get range values """
        op = OptimizationProblem()
        op.linear_constraints.add(
            range_values=[1.5 * i for i in range(10)],
            senses=["R"] * 10,
            names=[str(i) for i in range(10)])
        self.assertEqual(op.linear_constraints.get_num(), 10)
        self.assertEqual(op.linear_constraints.get_range_values(8), 12.0)
        self.assertListEqual(op.linear_constraints.get_range_values("1", 3), [1.5, 3.0, 4.5])
        self.assertListEqual(op.linear_constraints.get_range_values([2, "0", 5]), [3.0, 0.0, 7.5])
        self.assertListEqual(op.linear_constraints.get_range_values(),
                             [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5])

    def test_get_coefficients(self):
        """ test get coefficients """
        op = OptimizationProblem()
        op.variables.add(names=["x0", "x1"])
        op.linear_constraints.add(
            names=["c0", "c1"],
            lin_expr=[[[1], [1.0]], [[0, 1], [2.0, -1.0]]])
        self.assertAlmostEqual(op.linear_constraints.get_coefficients("c0", "x1"), 1.0)
        self.assertListEqual(
            op.linear_constraints.get_coefficients([("c1", "x0"), ("c1", "x1")]), [2.0, -1.0])

    def test_get_rows(self):
        """ test get rows """
        op = OptimizationProblem()
        op.variables.add(names=["x1", "x2", "x3"])
        op.linear_constraints.add(
            names=["c0", "c1", "c2", "c3"],
            lin_expr=[SparsePair(ind=["x1", "x3"], val=[1.0, -1.0]),
                      SparsePair(ind=["x1", "x2"], val=[1.0, 1.0]),
                      SparsePair(ind=["x1", "x2", "x3"], val=[-1.0] * 3),
                      SparsePair(ind=["x2", "x3"], val=[10.0, -2.0])])
        sp = op.linear_constraints.get_rows(0)
        self.assertListEqual(sp.ind, [0, 2])
        self.assertListEqual(sp.val, [1.0, -1.0])

        sp = op.linear_constraints.get_rows(1, 3)
        self.assertListEqual(sp[0].ind, [0, 1])
        self.assertListEqual(sp[0].val, [1.0, 1.0])
        self.assertListEqual(sp[1].ind, [0, 1, 2])
        self.assertListEqual(sp[1].val, [-1.0, -1.0, -1.0])
        self.assertListEqual(sp[2].ind, [1, 2])
        self.assertListEqual(sp[2].val, [10.0, -2.0])

        sp = op.linear_constraints.get_rows(['c2', 0])
        self.assertListEqual(sp[0].ind, [0, 1, 2])
        self.assertListEqual(sp[0].val, [-1.0, -1.0, -1.0])
        self.assertListEqual(sp[1].ind, [0, 2])
        self.assertListEqual(sp[1].val, [1.0, -1.0])

        sp = op.linear_constraints.get_rows()
        self.assertListEqual(sp[0].ind, [0, 2])
        self.assertListEqual(sp[0].val, [1.0, -1.0])
        self.assertListEqual(sp[1].ind, [0, 1])
        self.assertListEqual(sp[1].val, [1.0, 1.0])
        self.assertListEqual(sp[2].ind, [0, 1, 2])
        self.assertListEqual(sp[2].val, [-1.0, -1.0, -1.0])
        self.assertListEqual(sp[3].ind, [1, 2])
        self.assertListEqual(sp[3].val, [10.0, -2.0])

    def test_get_num_nonzeros(self):
        """ test get num non zeros """
        op = OptimizationProblem()
        op.variables.add(names=["x1", "x2", "x3"])
        op.linear_constraints.add(
            names=["c0", "c1", "c2", "c3"],
            lin_expr=[SparsePair(ind=["x1", "x3"], val=[1.0, -1.0]),
                      SparsePair(ind=["x1", "x2"], val=[1.0, 1.0]),
                      SparsePair(ind=["x1", "x2", "x3"], val=[-1.0] * 3),
                      SparsePair(ind=["x2", "x3"], val=[10.0, -2.0])])
        self.assertEqual(op.linear_constraints.get_num_nonzeros(), 9)
        op.linear_constraints.set_coefficients("c0", "x3", 0)
        self.assertEqual(op.linear_constraints.get_num_nonzeros(), 8)

    def test_get_names(self):
        """ test get names """
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c" + str(i) for i in range(10)])
        self.assertEqual(op.linear_constraints.get_num(), 10)
        self.assertEqual(op.linear_constraints.get_names(8), 'c8')
        self.assertListEqual(op.linear_constraints.get_names([2, 0, 5]), ['c2', 'c0', 'c5'])
        self.assertEqual(op.linear_constraints.get_names(),
                         ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'])

    def test_get_histogram(self):
        """ test get histogram """
        op = OptimizationProblem()
        self.assertRaises(NotImplementedError, lambda: op.linear_constraints.get_histogram())
