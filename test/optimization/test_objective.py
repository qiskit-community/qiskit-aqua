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

""" Test ObjectiveInterface """

import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase
from cplex import SparsePair
from qiskit.optimization import OptimizationProblem


class TestObjective(QiskitOptimizationTestCase):
    """Test ObjectiveInterface"""

    def test_obj_sense(self):
        """ test obj sense """
        op = OptimizationProblem()
        self.assertEqual(op.objective.sense.minimize, 1)
        self.assertEqual(op.objective.sense.maximize, -1)
        self.assertEqual(op.objective.sense[1], 'minimize')
        self.assertEqual(op.objective.sense[-1], 'maximize')

    def test_set_linear0(self):
        """
        op = OptimizationProblem()
        op.variables.add(names=[str(i) for i in range(4)])
        self.assertListEqual(op.objective.get_linear(), [0.0, 0.0, 0.0, 0.0])
        op.objective.set_linear(0, 1.0)
        self.assertListEqual(op.objective.get_linear(), [1.0, 0.0, 0.0, 0.0])
        op.objective.set_linear('3', -1.0)
        self.assertListEqual(op.objective.get_linear(), [1.0, 0.0, 0.0, -1.0])
        op.objective.set_linear([("2", 2.0), (1, 0.5)])
        self.assertListEqual(op.objective.get_linear(), [1.0, 0.5, 2.0, -1.0])
        """
        pass

    def test_set_linear(self):
        """ test set linear """
        op = OptimizationProblem()
        n = 4
        op.variables.add(names=[str(i) for i in range(n)])
        self.assertDictEqual(op.objective.get_linear(), {})
        op.objective.set_linear(0, 1.0)
        self.assertDictEqual(op.objective.get_linear(), {0: 1.0})
        self.assertListEqual(op.objective.get_linear(range(n)), [1.0, 0.0, 0.0, 0.0])
        op.objective.set_linear('3', -1.0)
        self.assertDictEqual(op.objective.get_linear(), {0: 1.0, 3: -1.0})
        op.objective.set_linear([("2", 2.0), (1, 0.5)])
        self.assertDictEqual(op.objective.get_linear(), {0: 1.0, 1: 0.5, 2: 2.0, 3: -1.0})
        self.assertListEqual(op.objective.get_linear(range(n)), [1.0, 0.5, 2.0, -1.0])

    def test_set_empty_quadratic(self):
        """ test set empty quadratic """
        op = OptimizationProblem()
        op.objective.set_quadratic([])
        self.assertRaises(TypeError, lambda: op.objective.set_quadratic())

    def test_set_quadratic(self):
        """ test set quadratic """
        op = OptimizationProblem()
        n = 3
        op.variables.add(names=[str(i) for i in range(n)])
        obj = op.objective
        obj.set_quadratic([SparsePair(ind=[0, 1, 2], val=[1.0, -2.0, 0.5]),
                           ([0, 1], [-2.0, -1.0]),
                           SparsePair(ind=[0, 2], val=[0.5, -3.0])])
        lst = obj.get_quadratic(range(n))
        self.assertListEqual(lst[0].ind, [0, 1, 2])
        self.assertListEqual(lst[0].val, [1.0, -2.0, 0.5])
        self.assertListEqual(lst[1].ind, [0, 1])
        self.assertListEqual(lst[1].val, [-2.0, -1.0])
        self.assertListEqual(lst[2].ind, [0, 2])
        self.assertListEqual(lst[2].val, [0.5, -3.0])

        obj.set_quadratic([1.0, 2.0, 3.0])
        lst = obj.get_quadratic(range(n))
        self.assertListEqual(lst[0].ind, [0])
        self.assertListEqual(lst[0].val, [1.0])
        self.assertListEqual(lst[1].ind, [1])
        self.assertListEqual(lst[1].val, [2.0])
        self.assertListEqual(lst[2].ind, [2])
        self.assertListEqual(lst[2].val, [3.0])

    def test_set_quadratic_coefficients(self):
        """ test set quadratic coefficients """
        op = OptimizationProblem()
        n = 3
        op.variables.add(names=[str(i) for i in range(n)])
        obj = op.objective
        obj.set_quadratic_coefficients(0, 1, 1.0)
        lst = op.objective.get_quadratic(range(n))
        self.assertListEqual(lst[0].ind, [1])
        self.assertListEqual(lst[0].val, [1.0])
        self.assertListEqual(lst[1].ind, [0])
        self.assertListEqual(lst[1].val, [1.0])
        self.assertListEqual(lst[2].ind, [])
        self.assertListEqual(lst[2].val, [])

        obj.set_quadratic_coefficients([(1, 1, 2.0), (0, 2, 3.0)])
        lst = op.objective.get_quadratic(range(n))
        self.assertListEqual(lst[0].ind, [1, 2])
        self.assertListEqual(lst[0].val, [1.0, 3.0])
        self.assertListEqual(lst[1].ind, [0, 1])
        self.assertListEqual(lst[1].val, [1.0, 2.0])
        self.assertListEqual(lst[2].ind, [0])
        self.assertListEqual(lst[2].val, [3.0])

        obj.set_quadratic_coefficients([(0, 1, 4.0), (1, 0, 5.0)])
        lst = op.objective.get_quadratic(range(n))
        self.assertListEqual(lst[0].ind, [1, 2])
        self.assertListEqual(lst[0].val, [5.0, 3.0])
        self.assertListEqual(lst[1].ind, [0, 1])
        self.assertListEqual(lst[1].val, [5.0, 2.0])
        self.assertListEqual(lst[2].ind, [0])
        self.assertListEqual(lst[2].val, [3.0])

    def test_set_senses(self):
        """ test set senses """
        op = OptimizationProblem()
        self.assertEqual(op.objective.sense[op.objective.get_sense()], 'minimize')
        op.objective.set_sense(op.objective.sense.maximize)
        self.assertEqual(op.objective.sense[op.objective.get_sense()], 'maximize')
        op.objective.set_sense(op.objective.sense.minimize)
        self.assertEqual(op.objective.sense[op.objective.get_sense()], 'minimize')

    def test_set_name(self):
        """ test set name """
        op = OptimizationProblem()
        op.objective.set_name('cost')
        self.assertEqual(op.objective.get_name(), 'cost')

    def test_get_linear(self):
        """ test get linear """
        op = OptimizationProblem()
        n = 10
        op.variables.add(names=[str(i) for i in range(n)])
        obj = op.objective
        obj.set_linear([(i, 1.5 * i) for i in range(n)])
        self.assertEqual(op.variables.get_num(), 10)
        self.assertEqual(obj.get_linear(8), 12)
        self.assertListEqual(obj.get_linear('1', 3), [1.5, 3.0, 4.5])
        self.assertListEqual(obj.get_linear([2, '0', 5]), [3.0, 0.0, 7.5])
        self.assertListEqual(obj.get_linear(range(n)),
                             [0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0, 10.5, 12.0, 13.5])

    def test_get_quadratic(self):
        """ test get quadratic """
        op = OptimizationProblem()
        n = 10
        op.variables.add(names=[str(i) for i in range(n)])
        obj = op.objective
        obj.set_quadratic([1.5 * i for i in range(n)])
        s_p = obj.get_quadratic(8)
        self.assertListEqual(s_p.ind, [8])
        self.assertListEqual(s_p.val, [12.0])

        s_p = obj.get_quadratic('1', 3)
        self.assertListEqual(s_p[0].ind, [1])
        self.assertListEqual(s_p[0].val, [1.5])
        self.assertListEqual(s_p[1].ind, [2])
        self.assertListEqual(s_p[1].val, [3.0])
        self.assertListEqual(s_p[2].ind, [3])
        self.assertListEqual(s_p[2].val, [4.5])

        s_p = obj.get_quadratic([3, '1', 5])
        self.assertListEqual(s_p[0].ind, [3])
        self.assertListEqual(s_p[0].val, [4.5])
        self.assertListEqual(s_p[1].ind, [1])
        self.assertListEqual(s_p[1].val, [1.5])
        self.assertListEqual(s_p[2].ind, [5])
        self.assertListEqual(s_p[2].val, [7.5])

        s_p = obj.get_quadratic(range(n))
        for i in range(n):
            self.assertListEqual(s_p[i].ind, [i])
            self.assertListEqual(s_p[i].val, [1.5 * i])

    def test_get_quadratic2(self):
        """ test get quadratic 2 """
        op = OptimizationProblem()
        n = 3
        op.variables.add(names=[str(i) for i in range(n)])
        obj = op.objective
        obj.set_quadratic_coefficients(0, 1, 1.0)
        self.assertEqual(obj.get_quadratic_coefficients('1', 0), 1.0)
        obj.set_quadratic_coefficients([(1, 1, 2.0), (0, 2, 3.0), (1, 0, 5.0)])
        self.assertListEqual(obj.get_quadratic_coefficients([(1, 0), (1, "1"), (2, "0")]),
                             [5.0, 2.0, 3.0])

    def test_get_sense(self):
        """ test get sense """
        op = OptimizationProblem()
        self.assertEqual(op.objective.sense[op.objective.get_sense()], 'minimize')
        op.objective.set_sense(op.objective.sense.maximize)
        self.assertEqual(op.objective.sense[op.objective.get_sense()], 'maximize')
        op.objective.set_sense(op.objective.sense.minimize)
        self.assertEqual(op.objective.sense[op.objective.get_sense()], 'minimize')

    def test_get_name(self):
        """ test get name """
        op = OptimizationProblem()
        op.objective.set_name('cost')
        self.assertEqual(op.objective.get_name(), 'cost')

    def test_get_num_quadratic_variables(self):
        """ test get num quadratic variables """
        op = OptimizationProblem()
        n = 3
        op.variables.add(names=[str(i) for i in range(n)])
        obj = op.objective
        obj.set_quadratic_coefficients(0, 1, 1.0)
        self.assertEqual(obj.get_num_quadratic_variables(), 2)
        obj.set_quadratic([1.0, 0.0, 0.0])
        self.assertEqual(obj.get_num_quadratic_variables(), 1)
        obj.set_quadratic_coefficients([(1, 1, 2.0), (0, 2, 3.0)])
        self.assertEqual(obj.get_num_quadratic_variables(), 3)

    def test_get_num_quadratic_nonzeros(self):
        """ test get num quadratic non zeros """
        op = OptimizationProblem()
        n = 3
        op.variables.add(names=[str(i) for i in range(n)])
        obj = op.objective
        obj.set_quadratic_coefficients(0, 1, 1.0)
        self.assertEqual(obj.get_num_quadratic_nonzeros(), 2)
        obj.set_quadratic_coefficients([(1, 1, 2.0), (0, 2, 3.0)])
        self.assertEqual(obj.get_num_quadratic_nonzeros(), 5)
        obj.set_quadratic_coefficients([(0, 1, 4.0), (1, 0, 0.0)])
        self.assertEqual(obj.get_num_quadratic_nonzeros(), 3)

    def test_offset(self):
        """ test offset """
        op = OptimizationProblem()
        self.assertEqual(op.objective.get_offset(), 0.0)
        op.objective.set_offset(3.14)
        self.assertEqual(op.objective.get_offset(), 3.14)


if __name__ == '__main__':
    unittest.main()
