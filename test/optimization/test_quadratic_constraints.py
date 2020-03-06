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

""" Test QuadraticConstraintInterface """

from cplex import SparsePair, SparseTriple

from qiskit.optimization import QiskitOptimizationError
from qiskit.optimization.problems import OptimizationProblem
from test.optimization.common import QiskitOptimizationTestCase


class TestQuadraticConstraints(QiskitOptimizationTestCase):
    """Test LinearConstraintInterface."""

    def setUp(self):
        super().setUp()

    def test_initial1(self):
        op = OptimizationProblem()
        c1 = op.quadratic_constraints.add(name='c1')
        c2 = op.quadratic_constraints.add(name='c2')
        c3 = op.quadratic_constraints.add(name='c3')
        self.assertEqual(op.quadratic_constraints.get_num(), 3)
        self.assertListEqual(op.quadratic_constraints.get_names(), ['c1', 'c2', 'c3'])
        self.assertListEqual([c1, c2, c3], [0, 1, 2])
        self.assertRaises(QiskitOptimizationError, lambda: op.quadratic_constraints.add(name='c1'))

    def test_initial2(self):
        op = OptimizationProblem()
        op.variables.add(names=['x1', 'x2', 'x3'], types='B' * 3)
        c = op.quadratic_constraints.add(
            lin_expr=SparsePair(ind=['x1', 'x3'], val=[1.0, -1.0]),
            quad_expr=SparseTriple(ind1=['x1', 'x2'], ind2=['x2', 'x3'], val=[1.0, -1.0]),
            sense='E',
            rhs=1.0
        )
        quad = op.quadratic_constraints
        self.assertEqual(quad.get_num(), 1)
        self.assertListEqual(quad.get_names(), ['q0'])
        self.assertListEqual(quad.get_rhs(), [1.0])
        self.assertListEqual(quad.get_senses(), ['E'])
        self.assertListEqual(quad.get_linear_num_nonzeros(), [2])
        self.assertListEqual(quad.get_quad_num_nonzeros(), [2])
        l = quad.get_linear_components()
        self.assertEqual(len(l), 1)
        self.assertTupleEqual(l[0].ind, (0, 2))
        self.assertTupleEqual(l[0].val, (1.0, -1.0))
        q = quad.get_quadratic_components()
        self.assertEqual(len(q), 1)
        self.assertTupleEqual(q[0].ind1, (1, 2))
        self.assertTupleEqual(q[0].ind2, (0, 1))
        self.assertTupleEqual(q[0].val, (1.0, -1.0))

    def test_initial3(self):
        op = OptimizationProblem()
        op.linear_constraints.add(names=[str(i) for i in range(10)])
        self.assertEqual(op.linear_constraints.get_num(), 10)
        op.linear_constraints.delete(8)
        self.assertEqual(len(op.linear_constraints.get_names()), 9)
        self.assertEqual(op.linear_constraints.get_names()[0], '0')
        # ['0', '1', '2', '3', '4', '5', '6', '7', '9']
        op.linear_constraints.delete("1", 3)
        self.assertEqual(len(op.linear_constraints.get_names()), 6)
        # ['0', '4', '5', '6', '7', '9']
        op.linear_constraints.delete([2, "0", 5])
        self.assertEqual(len(op.linear_constraints.get_names()), 3)
        self.assertEqual(op.linear_constraints.get_names()[0], '4')
        # ['4', '6', '7']
        op.linear_constraints.delete()
        self.assertEqual(len(op.linear_constraints.get_names()), 0)
        # []

    def test_rhs1(self):
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        self.assertEqual(len(op.linear_constraints.get_rhs()), 4)
        self.assertAlmostEqual(op.linear_constraints.get_rhs()[0], 0.0)
        # [0.0, 0.0, 0.0, 0.0]
        op.linear_constraints.set_rhs("c1", 1.0)
        self.assertEqual(len(op.linear_constraints.get_rhs()), 4)
        self.assertAlmostEqual(op.linear_constraints.get_rhs()[1], 1.0)
        # [0.0, 1.0, 0.0, 0.0]
        op.linear_constraints.set_rhs([("c3", 2.0), (2, -1.0)])
        self.assertEqual(len(op.linear_constraints.get_rhs()), 4)
        self.assertAlmostEqual(op.linear_constraints.get_rhs()[2], -1.0)
        self.assertAlmostEqual(op.linear_constraints.get_rhs()[3], 2.0)
        # [0.0, 1.0, -1.0, 2.0]

    def test_names(self):
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        op.linear_constraints.set_names("c1", "second")
        self.assertEqual(op.linear_constraints.get_names(1), 'second')
        op.linear_constraints.set_names([("c3", "last"), (2, "middle")])
        op.linear_constraints.get_names()
        self.assertEqual(len(op.linear_constraints.get_names()), 4)
        self.assertEqual(op.linear_constraints.get_names()[0], 'c0')
        self.assertEqual(op.linear_constraints.get_names()[1], 'second')
        self.assertEqual(op.linear_constraints.get_names()[2], 'middle')
        self.assertEqual(op.linear_constraints.get_names()[3], 'last')
        # ['c0', 'second', 'middle', 'last']

    def test_senses1(self):
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        op.linear_constraints.get_senses()
        self.assertEqual(len(op.linear_constraints.get_senses()), 4)
        self.assertEqual(op.linear_constraints.get_senses()[0], 'E')
        # ['E', 'E', 'E', 'E']
        op.linear_constraints.set_senses("c1", "G")
        self.assertEqual(op.linear_constraints.get_senses(1), 'G')
        op.linear_constraints.set_senses([("c3", "L"), (2, "R")])
        # ['E', 'G', 'R', 'L']
        self.assertEqual(op.linear_constraints.get_senses()[0], 'E')
        self.assertEqual(op.linear_constraints.get_senses()[1], 'G')
        self.assertEqual(op.linear_constraints.get_senses()[2], 'R')
        self.assertEqual(op.linear_constraints.get_senses()[3], 'L')

    def test_linear_components(self):
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        op.variables.add(names=["x0", "x1"])
        op.linear_constraints.set_linear_components("c0", [["x0"], [1.0]])
        self.assertEqual(op.linear_constraints.get_rows("c0").ind[0], 0)
        self.assertAlmostEqual(op.linear_constraints.get_rows("c0").val[0], 1.0)
        # SparsePair(ind = [0], val = [1.0])
        op.linear_constraints.set_linear_components([("c3", SparsePair(ind=["x1"], val=[-1.0])),
                                                     (2, [[0, 1], [-2.0, 3.0]])])
        op.linear_constraints.get_rows()
        # [SparsePair(ind = [0], val = [1.0]),
        #  SparsePair(ind = [], val = []),
        #  SparsePair(ind = [0, 1], val = [-2.0, 3.0]),
        #  SparsePair(ind = [1], val = [-1.0])]
        self.assertEqual(op.linear_constraints.get_rows()[0].ind[0], 0)
        self.assertAlmostEqual(op.linear_constraints.get_rows()[0].val[0], 1.0)
        self.assertEqual(op.linear_constraints.get_rows()[2].ind[0], 0)
        self.assertAlmostEqual(op.linear_constraints.get_rows()[2].val[0], -2.0)

    def test_linear_components_ranges(self):
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        op.linear_constraints.set_range_values("c1", 1.0)
        self.assertEqual(len(op.linear_constraints.get_range_values()), 4)
        self.assertAlmostEqual(op.linear_constraints.get_range_values()[0], 0.0)
        # [0.0, 1.0, 0.0, 0.0]
        op.linear_constraints.set_range_values([("c3", 2.0), (2, -1.0)])
        # [0.0, 1.0, -1.0, 2.0]
        self.assertEqual(len(op.linear_constraints.get_range_values()), 4)
        self.assertAlmostEqual(op.linear_constraints.get_range_values()[2], -1.0)
        self.assertAlmostEqual(op.linear_constraints.get_range_values()[3], 2.0)

    def test_rows(self):
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"])
        op.variables.add(names=["x0", "x1"])
        op.linear_constraints.set_coefficients("c0", "x1", 1.0)
        self.assertEqual(op.linear_constraints.get_rows(0).ind[0], 1)
        self.assertAlmostEqual(op.linear_constraints.get_rows(0).val[0], 1.0)
        # SparsePair(ind = [1], val = [1.0])
        op.linear_constraints.set_coefficients([("c2", "x0", 2.0),
                                                ("c2", "x1", -1.0)])
        # SparsePair(ind = [0, 1], val = [2.0, -1.0])
        self.assertEqual(op.linear_constraints.get_rows("c2").ind[0], 0)
        self.assertAlmostEqual(op.linear_constraints.get_rows("c2").val[0], 2.0)
        self.assertEqual(op.linear_constraints.get_rows("c2").ind[1], 1)
        self.assertAlmostEqual(op.linear_constraints.get_rows("c2").val[1], -1.0)

    def test_rhs2(self):
        op = OptimizationProblem()
        op.linear_constraints.add(rhs=[1.5 * i for i in range(10)],
                                  names=[str(i) for i in range(10)])
        self.assertEqual(op.linear_constraints.get_num(), 10)
        self.assertAlmostEqual(op.linear_constraints.get_rhs(8), 12.0)
        self.assertAlmostEqual(op.linear_constraints.get_rhs([2, "0", 5])[0], 3.0)
        self.assertAlmostEqual(op.linear_constraints.get_rhs([2, "0", 5])[1], 0.0)
        self.assertAlmostEqual(op.linear_constraints.get_rhs([2, "0", 5])[2], 7.5)
        # [3.0, 0.0, 7.5]
        self.assertEqual(len(op.linear_constraints.get_rhs()), 10)
        self.assertEqual(sum(op.linear_constraints.get_rhs()), 67.5)
        # [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]

    def test_senses2(self):
        op = OptimizationProblem()
        op.linear_constraints.add(
            senses=["E", "G", "L", "R"],
            names=[str(i) for i in range(4)])
        self.assertEqual(op.linear_constraints.get_num(), 4)
        self.assertEqual(op.linear_constraints.get_senses(1), 'G')
        self.assertEqual(op.linear_constraints.get_senses([2, "0", 1])[0], 'L')
        self.assertEqual(op.linear_constraints.get_senses([2, "0", 1])[1], 'E')
        self.assertEqual(op.linear_constraints.get_senses([2, "0", 1])[2], 'G')
        # ['L', 'E', 'G']
        self.assertEqual(op.linear_constraints.get_senses()[0], 'E')
        self.assertEqual(op.linear_constraints.get_senses()[1], 'G')
        # ['E', 'G', 'L', 'R']

    def test_range_values(self):
        op = OptimizationProblem()
        op.linear_constraints.add(
            range_values=[1.5 * i for i in range(10)],
            senses=["R"] * 10,
            names=[str(i) for i in range(10)])
        self.assertEqual(op.linear_constraints.get_num(), 10)
        self.assertAlmostEqual(op.linear_constraints.get_range_values(8), 12.0)
        self.assertAlmostEqual(sum(op.linear_constraints.get_range_values([2, "0", 5])), 10.5)
        # [3.0, 0.0, 7.5]
        self.assertAlmostEqual(sum(op.linear_constraints.get_range_values()), 67.5)
        # [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5]

    def test_coefficients(self):
        op = OptimizationProblem()
        op.variables.add(names=["x0", "x1"])
        op.linear_constraints.add(
            names=["c0", "c1"],
            lin_expr=[[[1], [1.0]], [[0, 1], [2.0, -1.0]]])
        self.assertAlmostEqual(op.linear_constraints.get_coefficients("c0", "x1"), 1.0)
        self.assertAlmostEqual(op.linear_constraints.get_coefficients(
            [("c1", "x0"), ("c1", "x1")])[0], 2.0)
        self.assertAlmostEqual(op.linear_constraints.get_coefficients(
            [("c1", "x0"), ("c1", "x1")])[1], -1.0)

    def test_rows2(self):
        op = OptimizationProblem()
        op.variables.add(names=["x1", "x2", "x3"])
        op.linear_constraints.add(
            names=["c0", "c1", "c2", "c3"],
            lin_expr=[SparsePair(ind=["x1", "x3"], val=[1.0, -1.0]),
                      SparsePair(ind=["x1", "x2"], val=[1.0, 1.0]),
                      SparsePair(ind=["x1", "x2", "x3"], val=[-1.0] * 3),
                      SparsePair(ind=["x2", "x3"], val=[10.0, -2.0])])
        self.assertEqual(op.linear_constraints.get_rows(0).ind[0], 0)
        self.assertAlmostEqual(op.linear_constraints.get_rows(0).val[0], 1.0)
        self.assertEqual(op.linear_constraints.get_rows(0).ind[1], 2)
        self.assertAlmostEqual(op.linear_constraints.get_rows(0).val[1], -1.0)
        # SparsePair(ind = [0, 2], val = [1.0, -1.0])
        self.assertEqual(op.linear_constraints.get_rows("c2").ind[0], 0)
        self.assertAlmostEqual(op.linear_constraints.get_rows("c2").val[0], -1.0)
        self.assertEqual(op.linear_constraints.get_rows("c2").ind[1], 1)
        self.assertAlmostEqual(op.linear_constraints.get_rows("c2").val[1], -1.0)
        # [SparsePair(ind = [0, 1, 2], val = [-1.0, -1.0, -1.0]),
        #  SparsePair(ind = [0, 2], val = [1.0, -1.0])]
        self.assertEqual(len(op.linear_constraints.get_rows()), 4)
        # [SparsePair(ind = [0, 2], val = [1.0, -1.0]),
        #  SparsePair(ind = [0, 1], val = [1.0, 1.0]),
        #  SparsePair(ind = [0, 1, 2], val = [-1.0, -1.0, -1.0]),
        #  SparsePair(ind = [1, 2], val = [10.0, -2.0])]

    def test_nnz(self):
        op = OptimizationProblem()
        op.variables.add(names=["x1", "x2", "x3"])
        op.linear_constraints.add(names=["c0", "c1", "c2", "c3"],
                                  lin_expr=[SparsePair(ind=["x1", "x3"], val=[1.0, -1.0]),
                                            SparsePair(ind=["x1", "x2"], val=[1.0, 1.0]),
                                            SparsePair(ind=["x1", "x2", "x3"], val=[-1.0] * 3),
                                            SparsePair(ind=["x2", "x3"], val=[10.0, -2.0])])
        self.assertEqual(op.linear_constraints.get_num_nonzeros(), 9)

    def test_names2(self):
        op = OptimizationProblem()
        op.linear_constraints.add(names=["c" + str(i) for i in range(10)])
        self.assertEqual(op.linear_constraints.get_num(), 10)
        self.assertEqual(op.linear_constraints.get_names(8), 'c8')
        self.assertEqual(op.linear_constraints.get_names([2, 0, 5])[0], 'c2')
        # ['c2', 'c0', 'c5']
        self.assertEqual(len(op.linear_constraints.get_names()), 10)
        # ['c0', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9']
