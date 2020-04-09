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

""" Test QuadraticProgram """

import os.path
import tempfile
import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase
import logging

from qiskit.optimization import QuadraticProgram, QiskitOptimizationError
from qiskit.optimization.problems.quadratic_program import SubstitutionStatus

logger = logging.getLogger(__name__)

_HAS_CPLEX = False
try:
    from cplex import Cplex, SparsePair, SparseTriple, infinity
    _HAS_CPLEX = True
except ImportError:
    logger.info('CPLEX is not installed.')


class TestQuadraticProgram(QiskitOptimizationTestCase):
    """Test QuadraticProgram without the members that have separate test classes
    (VariablesInterface, etc)."""

    def setUp(self):
        super().setUp()
        if not _HAS_CPLEX:
            self.skipTest('CPLEX is not installed.')

        self.resource_file = './test/optimization/resources/op_ip2.lp'

    def test_constructor1(self):
        """ test constructor """
        op = QuadraticProgram()
        self.assertEqual(op.get_problem_name(), '')
        op.variables.add(names=['x1', 'x2', 'x3'])
        self.assertEqual(op.variables.get_num(), 3)

    def test_constructor2(self):
        """ test constructor 2 """
        with self.assertRaises(QiskitOptimizationError):
            _ = QuadraticProgram("unknown")
        # If filename does not exist, an exception is raised.

    def test_constructor_context(self):
        """ test constructor context """
        with QuadraticProgram() as op:
            op.variables.add(names=['x1', 'x2', 'x3'])
            self.assertEqual(op.variables.get_num(), 3)

    def test_end(self):
        """ test end """
        op = QuadraticProgram()
        self.assertIsNone(op.end())

    def test_solve(self):
        """ test solve """
        op = QuadraticProgram()
        self.assertIsNone(op.solve())

    def test_read1(self):
        """ test read 1"""
        op = QuadraticProgram()
        op.read(self.resource_file)
        self.assertEqual(op.variables.get_num(), 3)

    def test_write1(self):
        """ test write 1 """
        op = QuadraticProgram()
        op.variables.add(names=['x1', 'x2', 'x3'])
        file, filename = tempfile.mkstemp(suffix='.lp')
        os.close(file)
        op.write(filename)
        self.assertEqual(os.path.exists(filename), 1)

    def test_write2(self):
        """ test write 2 """
        op1 = QuadraticProgram()
        op1.variables.add(names=['x1', 'x2', 'x3'])
        file, filename = tempfile.mkstemp(suffix='.lp')
        os.close(file)
        op1.write(filename)
        op2 = QuadraticProgram()
        op2.read(filename)
        self.assertEqual(op2.variables.get_num(), 3)

    def test_write3(self):
        """ test write 3 """
        op = QuadraticProgram()
        op.variables.add(names=['x1', 'x2', 'x3'])

        class NoOpStream:
            """ stream """

            def __init__(self):
                self.was_called = False

            def write(self, byt):
                """ write """
                # pylint: disable=unused-argument
                self.was_called = True
                pass

            def flush(self):
                """ flush """
                pass

        stream = NoOpStream()
        op.write_to_stream(stream)
        self.assertEqual(stream.was_called, True)
        with self.assertRaises(QiskitOptimizationError):
            op.write_to_stream("this-is-no-stream")

    def test_write4(self):
        """ test write 4 """
        # Writes a problem as a string in the given file format.
        op = QuadraticProgram()
        op.variables.add(names=['x1', 'x2', 'x3'])
        lp_str = op.write_as_string("lp")
        self.assertGreater(len(lp_str), 0)

    def test_problem_type1(self):
        """ test problem type 1 """
        op = QuadraticProgram()
        op.read(self.resource_file)
        self.assertEqual(op.get_problem_type(), op.problem_type.QP)
        self.assertEqual(op.problem_type[op.get_problem_type()], 'QP')

    def test_problem_type2(self):
        """ test problem type 2"""
        op = QuadraticProgram()
        op.set_problem_type(op.problem_type.LP)
        self.assertEqual(op.get_problem_type(), op.problem_type.LP)
        self.assertEqual(op.problem_type[op.get_problem_type()], 'LP')

    def test_problem_type3(self):
        """ test problem type 3"""
        op = QuadraticProgram()
        self.assertEqual(op.get_problem_type(), op.problem_type.LP)
        op.variables.add(names=['x1', 'x2', 'x3'], types='B' * 3)
        op.objective.set_linear([('x1', 2.0), ('x3', 0.5)])
        self.assertEqual(op.get_problem_type(), op.problem_type.MILP)
        op.objective.set_quadratic([
            SparsePair(ind=[0, 1], val=[2.0, 3.0]),
            SparsePair(ind=[0], val=[3.0]),
            SparsePair(ind=[], val=[])
        ])
        self.assertEqual(op.get_problem_type(), op.problem_type.MIQP)
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=["x1", "x3"], val=[1.0, -1.0]),
                      SparsePair(ind=["x1", "x2"], val=[1.0, 1.0]),
                      SparsePair(ind=["x1", "x2", "x3"], val=[-1.0] * 3),
                      SparsePair(ind=["x2", "x3"], val=[10.0, -2.0])],
            senses=["E", "L", "G", "R"],
            rhs=[0.0, 1.0, -1.0, 2.0],
            range_values=[0.0, 0.0, 0.0, -10.0],
            names=["c0", "c1", "c2", "c3"])
        self.assertEqual(op.get_problem_type(), op.problem_type.MIQP)
        op.quadratic_constraints.add(
            lin_expr=SparsePair(ind=['x1', 'x3'], val=[1.0, -1.0]),
            quad_expr=SparseTriple(ind1=['x1', 'x2'], ind2=['x2', 'x3'], val=[1.0, -1.0]),
            sense='E',
            rhs=1.0
        )
        self.assertEqual(op.get_problem_type(), op.problem_type.MIQCP)

    def test_problem_name(self):
        """ test problem name """
        op = QuadraticProgram()
        op.set_problem_name("test")
        # test
        self.assertEqual(op.get_problem_name(), "test")

    def test_from_and_to_cplex(self):
        """ test from_cplex and to_cplex """
        op = Cplex()
        op.variables.add(names=['x1', 'x2', 'x3'], types='B' * 3)
        op.objective.set_linear([('x1', 2.0), ('x3', 0.5)])
        op.objective.set_quadratic([
            SparsePair(ind=[0, 1], val=[2.0, 3.0]),
            SparsePair(ind=[0], val=[3.0]),
            SparsePair(ind=[], val=[])
        ])
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=["x1", "x3"], val=[1.0, -1.0]),
                      SparsePair(ind=["x1", "x2"], val=[1.0, 1.0]),
                      SparsePair(ind=["x1", "x2", "x3"], val=[-1.0] * 3),
                      SparsePair(ind=["x2", "x3"], val=[10.0, -2.0])],
            senses=["E", "L", "G", "R"],
            rhs=[0.0, 1.0, -1.0, 2.0],
            range_values=[0.0, 0.0, 0.0, -10.0],
            names=["c0", "c1", "c2", "c3"])
        op.quadratic_constraints.add(
            lin_expr=SparsePair(ind=['x1', 'x3'], val=[1.0, -1.0]),
            quad_expr=SparseTriple(ind1=['x1', 'x2'], ind2=['x2', 'x3'], val=[1.0, -1.0]),
            sense='E',
            rhs=1.0
        )
        orig = op.write_as_string()
        op2 = QuadraticProgram()
        op2.from_cplex(op)
        self.assertEqual(op2.write_as_string(), orig)
        op3 = op2.to_cplex()
        self.assertEqual(op3.write_as_string(), orig)

        op.set_problem_name('test')
        orig = op.write_as_string()
        op2 = QuadraticProgram()
        op2.from_cplex(op)
        self.assertEqual(op2.write_as_string(), orig)
        op3 = op2.to_cplex()
        self.assertEqual(op3.write_as_string(), orig)

    def test_substitute_variables_bounds1(self):
        """ test substitute variables bounds 1 """
        op = QuadraticProgram()
        op.set_problem_name('before')
        n = 5
        op.variables.add(names=['x' + str(i) for i in range(n)], types='I' * n,
                         lb=[-2] * n, ub=[4] * n)
        op2, status = op.substitute_variables(constants=SparsePair(ind=['x0'], val=[100]))
        self.assertEqual(status, SubstitutionStatus.infeasible)
        op2, status = op.substitute_variables(
            constants=SparsePair(ind=['x0'], val=[3.0]),
            variables=SparseTriple(ind1=['x1', 'x3'], ind2=['x2', 'x4'], val=[2.0, -2.0])
        )
        self.assertEqual(status, SubstitutionStatus.success)
        self.assertListEqual(op2.variables.get_names(), ['x2', 'x4'])
        self.assertListEqual(op2.variables.get_lower_bounds(), [-1, -2])
        self.assertListEqual(op2.variables.get_upper_bounds(), [2, 1])

    def test_substitute_variables_bounds2(self):
        """ test substitute variables bounds 2 """
        op = QuadraticProgram()
        op.set_problem_name('before')
        n = 5
        op.variables.add(names=['x' + str(i) for i in range(n)], types='I' * n,
                         lb=[0] * n, ub=[infinity] * n)
        op2, status = op.substitute_variables(constants=SparsePair(ind=['x0'], val=[-1]))
        self.assertEqual(status, SubstitutionStatus.infeasible)
        op2, status = op.substitute_variables(
            constants=SparsePair(ind=['x0'], val=[1.0]),
            variables=SparseTriple(ind1=['x1', 'x3'], ind2=['x2', 'x4'], val=[2.0, -2.0])
        )
        self.assertEqual(status, SubstitutionStatus.success)
        self.assertListEqual(op2.variables.get_names(), ['x2', 'x4'])
        self.assertListEqual(op2.variables.get_lower_bounds(), [0, 0])
        self.assertListEqual(op2.variables.get_upper_bounds(), [infinity, 0])

    def test_substitute_variables_obj(self):
        """ test substitute variables objective """
        op = QuadraticProgram()
        op.set_problem_name('before')
        op.variables.add(names=['x1', 'x2', 'x3'], types='I' * 3, lb=[-2] * 3, ub=[4] * 3)
        op.objective.set_linear([('x1', 1.0), ('x2', 2.0)])
        op.objective.set_quadratic_coefficients([
            ('x1', 'x1', 1),
            ('x2', 'x3', 2)
        ])
        op2, status = op.substitute_variables(
            constants=SparsePair(ind=['x1'], val=[3]),
            variables=SparseTriple(ind1=['x2'], ind2=['x3'], val=[-2])
        )
        self.assertEqual(status, SubstitutionStatus.success)
        self.assertListEqual(op2.variables.get_names(), ['x3'])
        self.assertEqual(op2.objective.get_offset(), 7.5)
        self.assertListEqual(op2.objective.get_linear(), [-4])
        self.assertEqual(op2.objective.get_quadratic_coefficients(0, 0), -8)
        self.assertEqual(op.objective.get_sense(), op2.objective.get_sense())

    def test_substitute_variables_lin_cst1(self):
        """ test substitute variables linear constraints 1 """
        op = QuadraticProgram()
        n = 5
        op.variables.add(names=['x' + str(i) for i in range(n)], types='I' * n,
                         lb=[-10] * n, ub=[14] * n)
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=['x0'], val=[1.0]),
                      SparsePair(ind=['x1'], val=[1.0]),
                      SparsePair(ind=['x2'], val=[1.0]),
                      SparsePair(ind=['x3'], val=[1.0]),
                      SparsePair(ind=['x4'], val=[1.0])],
            senses=["L", "E", "G", "R", "R"],
            rhs=[-1.0, 1.0, 1.0, 2.0, 2.0],
            range_values=[0.0, 0.0, 0.0, 10.0, -10.0],
            names=["c0", "c1", "c2", "c3", "c4"])
        _, status = op.substitute_variables(constants=SparsePair(ind=['x0'], val=[3]))
        self.assertEqual(status, SubstitutionStatus.infeasible)
        _, status = op.substitute_variables(constants=SparsePair(ind=['x1'], val=[3]))
        self.assertEqual(status, SubstitutionStatus.infeasible)
        _, status = op.substitute_variables(constants=SparsePair(ind=['x1'], val=[1]))
        self.assertEqual(status, SubstitutionStatus.success)
        _, status = op.substitute_variables(constants=SparsePair(ind=['x2'], val=[-1]))
        self.assertEqual(status, SubstitutionStatus.infeasible)
        _, status = op.substitute_variables(constants=SparsePair(ind=['x3'], val=[1.99]))
        self.assertEqual(status, SubstitutionStatus.infeasible)
        _, status = op.substitute_variables(constants=SparsePair(ind=['x3'], val=[2]))
        self.assertEqual(status, SubstitutionStatus.success)
        _, status = op.substitute_variables(constants=SparsePair(ind=['x3'], val=[12]))
        self.assertEqual(status, SubstitutionStatus.success)
        _, status = op.substitute_variables(constants=SparsePair(ind=['x3'], val=[12.01]))
        self.assertEqual(status, SubstitutionStatus.infeasible)
        _, status = op.substitute_variables(constants=SparsePair(ind=['x4'], val=[-8.01]))
        self.assertEqual(status, SubstitutionStatus.infeasible)
        _, status = op.substitute_variables(constants=SparsePair(ind=['x4'], val=[-8]))
        self.assertEqual(status, SubstitutionStatus.success)
        _, status = op.substitute_variables(constants=SparsePair(ind=['x4'], val=[2]))
        self.assertEqual(status, SubstitutionStatus.success)
        _, status = op.substitute_variables(constants=SparsePair(ind=['x4'], val=[2.01]))
        self.assertEqual(status, SubstitutionStatus.infeasible)

    def test_substitute_variables_lin_cst2(self):
        """ test substitute variables linear constraints 2 """
        op = QuadraticProgram()
        n = 3
        op.variables.add(names=['x' + str(i) for i in range(n)], types='I' * n,
                         lb=[-2] * n, ub=[4] * n)
        op.linear_constraints.add(
            lin_expr=[SparsePair(ind=["x0", "x2"], val=[1.0, -1.0]),
                      SparsePair(ind=["x0", "x1"], val=[1.0, 1.0]),
                      SparsePair(ind=["x0", "x1", "x2"], val=[-1.0] * 3),
                      SparsePair(ind=["x1", "x2"], val=[10.0, -2.0])],
            senses=["E", "L", "G", "R"],
            rhs=[0.0, 1.0, -1.0, 2.0],
            range_values=[0.0, 0.0, 0.0, -10.0],
            names=["c0", "c1", "c2", "c3"])
        op2, status = op.substitute_variables(
            SparsePair(ind=['x0'], val=[2.0]),
            SparseTriple(ind1=['x1'], ind2=['x2'], val=[3.0])
        )
        self.assertEqual(status, SubstitutionStatus.success)
        self.assertListEqual(op2.variables.get_names(), ['x2'])
        rows = op2.linear_constraints.get_rows()
        self.assertListEqual(rows[0].ind, [0])
        self.assertListEqual(rows[0].val, [-1])
        self.assertListEqual(rows[1].ind, [0])
        self.assertListEqual(rows[1].val, [3])
        self.assertListEqual(rows[2].ind, [0])
        self.assertListEqual(rows[2].val, [-4])
        self.assertListEqual(rows[3].ind, [0])
        self.assertListEqual(rows[3].val, [28])
        self.assertListEqual(op2.linear_constraints.get_rhs(), [-2, -1, 1, 2])

    def test_substitute_variables_quad_cst1(self):
        """ test substitute variables quadratic constraints 1 """
        op = QuadraticProgram()
        n = 3
        op.variables.add(names=['x' + str(i) for i in range(n)], types='I' * n,
                         lb=[-2] * n, ub=[4] * n)
        op.quadratic_constraints.add(
            lin_expr=SparsePair(ind=['x0', 'x1'], val=[1.0, -1.0]),
            quad_expr=SparseTriple(ind1=['x0', 'x1', 'x2'], ind2=['x1', 'x2', 'x2'],
                                   val=[1.0, -2.0, 3.0]),
            sense='L',
            rhs=1.0
        )
        op2, status = op.substitute_variables(
            constants=SparsePair(ind=['x0', 'x1', 'x2'], val=[1, 1, 1]))
        self.assertEqual(status, SubstitutionStatus.infeasible)

        op2, status = op.substitute_variables(SparsePair(ind=['x0', 'x1', 'x2'], val=[-1, 1, 1]))
        self.assertEqual(status, SubstitutionStatus.success)

        op2, status = op.substitute_variables(SparsePair(ind=['x0', 'x1'], val=[1, -1]))
        self.assertEqual(status, SubstitutionStatus.success)
        self.assertEqual(op2.quadratic_constraints.get_num(), 1)
        lin = op2.quadratic_constraints.get_linear_components(0)
        self.assertListEqual(lin.ind, [0])
        self.assertListEqual(lin.val, [2])
        q = op2.quadratic_constraints.get_quadratic_components(0)
        self.assertListEqual(q.ind1, [0])
        self.assertListEqual(q.ind2, [0])
        self.assertListEqual(q.val, [3])
        self.assertEqual(op2.quadratic_constraints.get_senses(0), 'L')
        self.assertEqual(op2.quadratic_constraints.get_rhs(0), 0)

        with self.assertRaises(QiskitOptimizationError):
            op.substitute_variables(
                variables=SparseTriple(ind1=['x0'], ind2=['x0'], val=[2]))
        with self.assertRaises(QiskitOptimizationError):
            op.substitute_variables(
                variables=SparseTriple(ind1=['x1', 'x0'], ind2=['x2', 'x1'], val=[1.5, 1]))
        with self.assertRaises(QiskitOptimizationError):
            op.substitute_variables(
                variables=SparseTriple(ind1=['x1', 'x1'], ind2=['x2', 'x0'], val=[1.5, 1]))

        op2, status = op.substitute_variables(
            variables=SparseTriple(ind1=['x1'], ind2=['x2'], val=[1.5]))
        self.assertEqual(status, op.substitution_status.success)
        lin = op2.quadratic_constraints.get_linear_components(0)
        self.assertListEqual(lin.ind, [0, 1])
        self.assertListEqual(lin.val, [1, -1.5])
        q = op2.quadratic_constraints.get_quadratic_components(0)
        self.assertListEqual(q.ind1, [1])
        self.assertListEqual(q.ind2, [0])
        self.assertListEqual(q.val, [1.5])
        self.assertEqual(op2.quadratic_constraints.get_senses(0), 'L')
        self.assertEqual(op2.quadratic_constraints.get_rhs(0), 1)

        op2, status = op.substitute_variables(
            constants=SparsePair(ind=['x2'], val=[2]))
        self.assertEqual(status, op.substitution_status.success)
        lin = op2.quadratic_constraints.get_linear_components(0)
        self.assertListEqual(lin.ind, [0, 1])
        self.assertListEqual(lin.val, [1, -5])
        q = op2.quadratic_constraints.get_quadratic_components(0)
        self.assertListEqual(q.ind1, [1])
        self.assertListEqual(q.ind2, [0])
        self.assertListEqual(q.val, [1])
        self.assertEqual(op2.quadratic_constraints.get_senses(0), 'L')
        self.assertEqual(op2.quadratic_constraints.get_rhs(0), -11)


if __name__ == '__main__':
    unittest.main()
