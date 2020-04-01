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

import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase
from cplex import SparsePair, SparseTriple

from qiskit.optimization import QiskitOptimizationError
from qiskit.optimization.problems import OptimizationProblem


class TestQuadraticConstraints(QiskitOptimizationTestCase):
    """Test QuadraticConstraintInterface."""

    def test_initial1(self):
        """ test initial 1"""
        op = OptimizationProblem()
        c_1 = op.quadratic_constraints.add(name='c1')
        c_2 = op.quadratic_constraints.add(name='c2')
        c_3 = op.quadratic_constraints.add(name='c3')
        self.assertEqual(op.quadratic_constraints.get_num(), 3)
        self.assertListEqual(op.quadratic_constraints.get_names(), ['c1', 'c2', 'c3'])
        self.assertListEqual([c_1, c_2, c_3], [0, 1, 2])
        self.assertRaises(QiskitOptimizationError, lambda: op.quadratic_constraints.add(name='c1'))

    def test_initial2(self):
        """ test initial 2"""
        op = OptimizationProblem()
        op.variables.add(names=['x1', 'x2', 'x3'], types='B' * 3)
        _ = op.quadratic_constraints.add(
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
        l_a = quad.get_linear_components()
        self.assertEqual(len(l_a), 1)
        self.assertListEqual(l_a[0].ind, [0, 2])
        self.assertListEqual(l_a[0].val, [1.0, -1.0])
        q = quad.get_quadratic_components()
        self.assertEqual(len(q), 1)
        self.assertListEqual(q[0].ind1, [1, 2])
        self.assertListEqual(q[0].ind2, [0, 1])
        self.assertListEqual(q[0].val, [1.0, -1.0])

    def test_get_num(self):
        """ test get num """
        op = OptimizationProblem()
        op.variables.add(names=['x', 'y'])
        l_a = SparsePair(ind=['x'], val=[1.0])
        q = SparseTriple(ind1=['x'], ind2=['y'], val=[1.0])
        n = 10
        for i in range(n):
            self.assertEqual(op.quadratic_constraints.add(name=str(i),
                                                          lin_expr=l_a, quad_expr=q), i)
        self.assertEqual(op.quadratic_constraints.get_num(), n)

    def test_add(self):
        """ test add """
        op = OptimizationProblem()
        op.variables.add(names=['x', 'y'])
        l_a = SparsePair(ind=['x'], val=[1.0])
        q = SparseTriple(ind1=['x'], ind2=['y'], val=[1.0])
        self.assertEqual(op.quadratic_constraints.add(
            name='my quad', lin_expr=l_a, quad_expr=q, rhs=1.0, sense='G'), 0)

    def test_delete(self):
        """ test delete """
        op = OptimizationProblem()
        q_0 = [op.quadratic_constraints.add(name=str(i)) for i in range(10)]
        self.assertListEqual(q_0, list(range(10)))
        q = op.quadratic_constraints
        self.assertListEqual(q.get_names(), ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
        q.delete(8)
        self.assertListEqual(q.get_names(), ['0', '1', '2', '3', '4', '5', '6', '7', '9'])
        q.delete("1", 3)
        self.assertListEqual(q.get_names(), ['0', '4', '5', '6', '7', '9'])
        q.delete([2, "0", 5])
        self.assertListEqual(q.get_names(), ['4', '6', '7'])
        q.delete()
        self.assertListEqual(q.get_names(), [])

    def test_get_rhs(self):
        """ test get rhs  """
        op = OptimizationProblem()
        op.variables.add(names=[str(i) for i in range(10)])
        q_0 = [op.quadratic_constraints.add(rhs=1.5 * i, name=str(i)) for i in range(10)]
        self.assertListEqual(q_0, list(range(10)))
        q = op.quadratic_constraints
        self.assertEqual(q.get_num(), 10)
        self.assertEqual(q.get_rhs(8), 12.0)
        self.assertListEqual(q.get_rhs('1', 3), [1.5, 3.0, 4.5])
        self.assertListEqual(q.get_rhs([2, '0', 5]), [3.0, 0.0, 7.5])
        self.assertListEqual(q.get_rhs(), [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5])

    def test_get_senses(self):
        """ test get senses """
        op = OptimizationProblem()
        op.variables.add(names=["x0"])
        q = op.quadratic_constraints
        q_0 = [q.add(name=str(i), sense=j) for i, j in enumerate('GGLL')]
        self.assertListEqual(q_0, [0, 1, 2, 3])
        self.assertEqual(q.get_senses(1), 'G')
        self.assertListEqual(q.get_senses('1', 3), ['G', 'L', 'L'])
        self.assertListEqual(q.get_senses([2, '0', 1]), ['L', 'G', 'G'])
        self.assertListEqual(q.get_senses(), ['G', 'G', 'L', 'L'])

    def test_get_linear_num_nonzeros(self):
        """ test get linear num non zeros """
        op = OptimizationProblem()
        op.variables.add(names=[str(i) for i in range(11)], types="B" * 11)
        q = op.quadratic_constraints
        n = 10
        _ = [q.add(name=str(i),
                   lin_expr=[range(i), [1.0 * (j + 1.0) for j in range(i)]])
             for i in range(n)]
        self.assertEqual(q.get_num(), n)
        self.assertEqual(q.get_linear_num_nonzeros(8), 8)
        self.assertListEqual(q.get_linear_num_nonzeros('1', 3), [1, 2, 3])
        self.assertListEqual(q.get_linear_num_nonzeros([2, '0', 5]), [2, 0, 5])
        self.assertListEqual(q.get_linear_num_nonzeros(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    def test_get_linear_components(self):
        """ test get linear components """
        op = OptimizationProblem()
        op.variables.add(names=[str(i) for i in range(4)], types="B" * 4)
        q = op.quadratic_constraints
        z = [q.add(name=str(i),
                   lin_expr=[range(i), [1.0 * (j + 1.0) for j in range(i)]]) for i in range(3)]
        self.assertListEqual(z, [0, 1, 2])
        self.assertEqual(q.get_num(), 3)

        s_p = q.get_linear_components(2)
        self.assertListEqual(s_p.ind, [0, 1])
        self.assertListEqual(s_p.val, [1.0, 2.0])

        s_p = q.get_linear_components('0', 1)
        self.assertListEqual(s_p[0].ind, [])
        self.assertListEqual(s_p[0].val, [])
        self.assertListEqual(s_p[1].ind, [0])
        self.assertListEqual(s_p[1].val, [1.0])

        s_p = q.get_linear_components([1, '0'])
        self.assertListEqual(s_p[0].ind, [0])
        self.assertListEqual(s_p[0].val, [1.0])
        self.assertListEqual(s_p[1].ind, [])
        self.assertListEqual(s_p[1].val, [])

        s_p = q.get_linear_components()
        self.assertListEqual(s_p[0].ind, [])
        self.assertListEqual(s_p[0].val, [])
        self.assertListEqual(s_p[1].ind, [0])
        self.assertListEqual(s_p[1].val, [1.0])
        self.assertListEqual(s_p[2].ind, [0, 1])
        self.assertListEqual(s_p[2].val, [1.0, 2.0])

    def test_get_linear_components2(self):
        """ test get linear components 2 """
        op = OptimizationProblem()
        op.variables.add(names=[str(i) for i in range(11)], types="B" * 11)
        q = op.quadratic_constraints
        _ = [q.add(name=str(i),
                   lin_expr=[range(i), [1.0 * (j + 1.0) for j in range(i)]])
             for i in range(10)]
        s_c = q.get_linear_components(8)
        self.assertListEqual(s_c.ind, [0, 1, 2, 3, 4, 5, 6, 7])
        self.assertListEqual(s_c.val, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

        s_c = q.get_linear_components('1', 3)
        self.assertEqual(len(s_c), 3)
        self.assertListEqual(s_c[0].ind, [0])
        self.assertListEqual(s_c[0].val, [1.0])
        self.assertListEqual(s_c[1].ind, [0, 1])
        self.assertListEqual(s_c[1].val, [1.0, 2.0])
        self.assertListEqual(s_c[2].ind, [0, 1, 2])
        self.assertListEqual(s_c[2].val, [1.0, 2.0, 3.0])

        s_c = q.get_linear_components([2, '0', 5])
        self.assertEqual(len(s_c), 3)
        self.assertListEqual(s_c[0].ind, [0, 1])
        self.assertListEqual(s_c[0].val, [1.0, 2.0])
        self.assertListEqual(s_c[1].ind, [])
        self.assertListEqual(s_c[1].val, [])
        self.assertListEqual(s_c[2].ind, [0, 1, 2, 3, 4])
        self.assertListEqual(s_c[2].val, [1.0, 2.0, 3.0, 4.0, 5.0])

        q.delete(4, 9)
        s_c = q.get_linear_components()
        self.assertEqual(len(s_c), 4)
        self.assertListEqual(s_c[0].ind, [])
        self.assertListEqual(s_c[0].val, [])
        self.assertListEqual(s_c[1].ind, [0])
        self.assertListEqual(s_c[1].val, [1.0])
        self.assertListEqual(s_c[2].ind, [0, 1])
        self.assertListEqual(s_c[2].val, [1.0, 2.0])
        self.assertListEqual(s_c[3].ind, [0, 1, 2])
        self.assertListEqual(s_c[3].val, [1.0, 2.0, 3.0])

    def test_quad_num_nonzeros(self):
        """ test quad num non zeros """
        op = OptimizationProblem()
        op.variables.add(names=[str(i) for i in range(11)])
        q = op.quadratic_constraints
        _ = [q.add(name=str(i),
                   quad_expr=[range(i), range(i), [1.0 * (j + 1.0) for j in range(i)]])
             for i in range(1, 11)]
        self.assertEqual(q.get_num(), 10)
        self.assertEqual(q.get_quad_num_nonzeros(8), 9)
        self.assertListEqual(q.get_quad_num_nonzeros('1', 2), [1, 2, 3])
        self.assertListEqual(q.get_quad_num_nonzeros([2, '1', 5]), [3, 1, 6])
        self.assertListEqual(q.get_quad_num_nonzeros(), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_get_quadratic_components(self):
        """ test get quadratic components """
        op = OptimizationProblem()
        op.variables.add(names=[str(i) for i in range(4)])
        q = op.quadratic_constraints
        z = [q.add(name="q{0}".format(i),
                   quad_expr=[range(i), range(i), [1.0 * (j + 1.0) for j in range(i)]])
             for i in range(1, 3)]
        self.assertListEqual(z, [0, 1])
        self.assertEqual(q.get_num(), 2)

        s_t = q.get_quadratic_components(1)
        self.assertListEqual(s_t.ind1, [0, 1])
        self.assertListEqual(s_t.ind2, [0, 1])
        self.assertListEqual(s_t.val, [1.0, 2.0])

        s_t = q.get_quadratic_components('q1', 1)
        self.assertListEqual(s_t[0].ind1, [0])
        self.assertListEqual(s_t[0].ind2, [0])
        self.assertListEqual(s_t[0].val, [1.0])
        self.assertListEqual(s_t[1].ind1, [0, 1])
        self.assertListEqual(s_t[1].ind2, [0, 1])
        self.assertListEqual(s_t[1].val, [1.0, 2.0])

        s_t = q.get_quadratic_components(['q2', 0])
        self.assertListEqual(s_t[0].ind1, [0, 1])
        self.assertListEqual(s_t[0].ind2, [0, 1])
        self.assertListEqual(s_t[0].val, [1.0, 2.0])
        self.assertListEqual(s_t[1].ind1, [0])
        self.assertListEqual(s_t[1].ind2, [0])
        self.assertListEqual(s_t[1].val, [1.0])

        s_t = q.get_quadratic_components()
        self.assertListEqual(s_t[0].ind1, [0])
        self.assertListEqual(s_t[0].ind2, [0])
        self.assertListEqual(s_t[0].val, [1.0])
        self.assertListEqual(s_t[1].ind1, [0, 1])
        self.assertListEqual(s_t[1].ind2, [0, 1])
        self.assertListEqual(s_t[1].val, [1.0, 2.0])

    def test_get_quadratic_components2(self):
        """ test get quadratic components 2 """
        op = OptimizationProblem()
        op.variables.add(names=[str(i) for i in range(11)])
        q = op.quadratic_constraints
        _ = [q.add(name=str(i),
                   quad_expr=[range(i), range(i), [1.0 * (j + 1.0) for j in range(i)]])
             for i in range(1, 11)]
        s_c = q.get_quadratic_components(8)
        self.assertListEqual(s_c.ind1, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertListEqual(s_c.ind2, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.assertListEqual(s_c.val, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

        s_c = q.get_quadratic_components('1', 3)
        self.assertEqual(len(s_c), 4)
        self.assertListEqual(s_c[0].ind1, [0])
        self.assertListEqual(s_c[0].ind2, [0])
        self.assertListEqual(s_c[0].val, [1.0])
        self.assertListEqual(s_c[1].ind1, [0, 1])
        self.assertListEqual(s_c[1].ind2, [0, 1])
        self.assertListEqual(s_c[1].val, [1.0, 2.0])
        self.assertListEqual(s_c[2].ind1, [0, 1, 2])
        self.assertListEqual(s_c[2].ind2, [0, 1, 2])
        self.assertListEqual(s_c[2].val, [1.0, 2.0, 3.0])
        self.assertListEqual(s_c[3].ind1, [0, 1, 2, 3])
        self.assertListEqual(s_c[3].ind2, [0, 1, 2, 3])
        self.assertListEqual(s_c[3].val, [1.0, 2.0, 3.0, 4.0])

        s_c = q.get_quadratic_components([2, '1', 5])
        self.assertEqual(len(s_c), 3)
        self.assertListEqual(s_c[0].ind1, [0, 1, 2])
        self.assertListEqual(s_c[0].ind2, [0, 1, 2])
        self.assertListEqual(s_c[0].val, [1.0, 2.0, 3.0])
        self.assertListEqual(s_c[1].ind1, [0])
        self.assertListEqual(s_c[1].ind2, [0])
        self.assertListEqual(s_c[1].val, [1.0])
        self.assertListEqual(s_c[2].ind1, [0, 1, 2, 3, 4, 5])
        self.assertListEqual(s_c[2].ind2, [0, 1, 2, 3, 4, 5])
        self.assertListEqual(s_c[2].val, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        q.delete(4, 9)
        s_c = q.get_quadratic_components()
        self.assertEqual(len(s_c), 4)
        self.assertListEqual(s_c[0].ind1, [0])
        self.assertListEqual(s_c[0].ind2, [0])
        self.assertListEqual(s_c[0].val, [1.0])
        self.assertListEqual(s_c[1].ind1, [0, 1])
        self.assertListEqual(s_c[1].ind2, [0, 1])
        self.assertListEqual(s_c[1].val, [1.0, 2.0])
        self.assertListEqual(s_c[2].ind1, [0, 1, 2])
        self.assertListEqual(s_c[2].ind2, [0, 1, 2])
        self.assertListEqual(s_c[2].val, [1.0, 2.0, 3.0])
        self.assertListEqual(s_c[3].ind1, [0, 1, 2, 3])
        self.assertListEqual(s_c[3].ind2, [0, 1, 2, 3])
        self.assertListEqual(s_c[3].val, [1.0, 2.0, 3.0, 4.0])

    def test_get_names(self):
        """ test get names """
        op = OptimizationProblem()
        op.variables.add(names=[str(i) for i in range(11)])
        q = op.quadratic_constraints
        _ = [q.add(name="q" + str(i),
                   quad_expr=[range(i), range(i), [1.0 * (j + 1.0) for j in range(i)]])
             for i in range(1, 11)]
        self.assertEqual(q.get_num(), 10)
        self.assertEqual(q.get_names(8), 'q9')
        self.assertListEqual(q.get_names(1, 3), ['q2', 'q3', 'q4'])
        self.assertListEqual(q.get_names([2, 0, 5]), ['q3', 'q1', 'q6'])
        self.assertListEqual(q.get_names(),
                             ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10'])


if __name__ == '__main__':
    unittest.main()
