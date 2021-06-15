# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QuadraticProgram """

import tempfile
import unittest
from os import path
from test.optimization.optimization_test_case import QiskitOptimizationTestCase

from docplex.mp.model import DOcplexException, Model

from qiskit.aqua import MissingOptionalLibraryError
from qiskit.optimization import INFINITY, QiskitOptimizationError, QuadraticProgram
from qiskit.optimization.problems import Constraint, QuadraticObjective, Variable, VarType

# pylint: disable=no-member


class TestQuadraticProgram(QiskitOptimizationTestCase):
    """Test QuadraticProgram without the members that have separate test classes
    (VariablesInterface, etc)."""

    def test_constructor(self):
        """ test constructor """
        quadratic_program = QuadraticProgram()
        self.assertEqual(quadratic_program.name, '')
        self.assertEqual(quadratic_program.status, QuadraticProgram.Status.VALID)
        self.assertEqual(quadratic_program.get_num_vars(), 0)
        self.assertEqual(quadratic_program.get_num_linear_constraints(), 0)
        self.assertEqual(quadratic_program.get_num_quadratic_constraints(), 0)
        self.assertEqual(quadratic_program.objective.constant, 0)
        self.assertDictEqual(quadratic_program.objective.linear.to_dict(), {})
        self.assertDictEqual(quadratic_program.objective.quadratic.to_dict(), {})

    def test_clear(self):
        """ test clear """
        q_p = QuadraticProgram('test')
        q_p.binary_var('x')
        q_p.binary_var('y')
        q_p.minimize(constant=1, linear={'x': 1, 'y': 2}, quadratic={('x', 'x'): 1})
        q_p.linear_constraint({'x': 1}, '==', 1)
        q_p.quadratic_constraint({'x': 1}, {('y', 'y'): 2}, '<=', 1)
        q_p.clear()
        self.assertEqual(q_p.name, '')
        self.assertEqual(q_p.status, QuadraticProgram.Status.VALID)
        self.assertEqual(q_p.get_num_vars(), 0)
        self.assertEqual(q_p.get_num_linear_constraints(), 0)
        self.assertEqual(q_p.get_num_quadratic_constraints(), 0)
        self.assertEqual(q_p.objective.constant, 0)
        self.assertDictEqual(q_p.objective.linear.to_dict(), {})
        self.assertDictEqual(q_p.objective.quadratic.to_dict(), {})

    def test_name_setter(self):
        """ test name setter """
        q_p = QuadraticProgram()
        self.assertEqual(q_p.name, '')
        name = 'test name'
        q_p.name = name
        self.assertEqual(q_p.name, name)

    def assert_equal(self, x: Variable, y: Variable):
        """asserts variable equality"""
        self.assertEqual(x.name, y.name)
        self.assertEqual(x.lowerbound, y.lowerbound)
        self.assertEqual(x.upperbound, y.upperbound)
        self.assertEqual(x.vartype, y.vartype)

    def test_var_dict(self):
        """test {binary,integer,continuous}_var_dict"""
        q_p = QuadraticProgram()

        d_0 = q_p.continuous_var_dict(name='a', key_format='_{}', keys=3)
        self.assertSetEqual(set(d_0.keys()), {'a_0', 'a_1', 'a_2'})
        self.assertSetEqual({var.name for var in q_p.variables}, {'a_0', 'a_1', 'a_2'})
        for var in q_p.variables:
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, INFINITY)
            self.assertEqual(var.vartype, VarType.CONTINUOUS)
            self.assertTupleEqual(var.as_tuple(), d_0[var.name].as_tuple())

        d_1 = q_p.binary_var_dict(name='b', keys=5)
        self.assertSetEqual(set(d_1.keys()), {'b3', 'b4', 'b5', 'b6', 'b7'})
        self.assertSetEqual({var.name for var in q_p.variables},
                            {'a_0', 'a_1', 'a_2', 'b3', 'b4', 'b5', 'b6', 'b7'})
        for var in q_p.variables[-5:]:
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_1[var.name].as_tuple())

        d_2 = q_p.integer_var_dict(keys=1, key_format='-{}', lowerbound=-4, upperbound=10)
        self.assertSetEqual(set(d_2.keys()), {'x-8'})
        self.assertSetEqual({var.name for var in q_p.variables},
                            {'a_0', 'a_1', 'a_2', 'b3', 'b4', 'b5', 'b6', 'b7', 'x-8'})
        for var in q_p.variables[-1:]:
            self.assertAlmostEqual(var.lowerbound, -4)
            self.assertAlmostEqual(var.upperbound, 10)
            self.assertEqual(var.vartype, VarType.INTEGER)
            self.assertTupleEqual(var.as_tuple(), d_2[var.name].as_tuple())

        d_3 = q_p.binary_var_dict(name='c', keys=range(3))
        self.assertSetEqual(set(d_3.keys()), {'c0', 'c1', 'c2'})
        self.assertSetEqual({var.name for var in q_p.variables},
                            {'a_0', 'a_1', 'a_2', 'b3', 'b4', 'b5', 'b6', 'b7', 'x-8',
                             'c0', 'c1', 'c2'})
        for var in q_p.variables[-3:]:
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_3[var.name].as_tuple())

        with self.assertRaises(QiskitOptimizationError):
            q_p.binary_var_dict(name='c', keys=range(3))

        d_4 = q_p.binary_var_dict(1, 'x', '_')
        self.assertSetEqual(set(d_4.keys()), {'x_'})
        self.assertSetEqual({var.name for var in q_p.variables},
                            {'a_0', 'a_1', 'a_2', 'b3', 'b4', 'b5', 'b6', 'b7', 'x-8',
                             'c0', 'c1', 'c2', 'x_'})
        for var in q_p.variables[-1:]:
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_4[var.name].as_tuple())

        with self.assertRaises(QiskitOptimizationError):
            q_p.binary_var_dict(1, 'x', '_')

        with self.assertRaises(QiskitOptimizationError):
            q_p.binary_var('x_')

        d_5 = q_p.continuous_var_dict(1, -1, 2, '', '')
        self.assertSetEqual(set(d_5.keys()), {''})
        self.assertSetEqual({var.name for var in q_p.variables},
                            {'a_0', 'a_1', 'a_2', 'b3', 'b4', 'b5', 'b6', 'b7', 'x-8',
                             'c0', 'c1', 'c2', 'x_', ''})
        for var in q_p.variables[-1:]:
            self.assertAlmostEqual(var.lowerbound, -1)
            self.assertAlmostEqual(var.upperbound, 2)
            self.assertEqual(var.vartype, VarType.CONTINUOUS)
            self.assertTupleEqual(var.as_tuple(), d_5[var.name].as_tuple())

        with self.assertRaises(QiskitOptimizationError):
            q_p.binary_var_dict(1, '', '')

        with self.assertRaises(QiskitOptimizationError):
            q_p.integer_var(0, 1, '')

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.binary_var_dict(keys=1, key_format='{}{}')

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.binary_var_dict(keys=0)

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.binary_var_dict(keys=1, key_format='_{{}}')

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.binary_var_dict(keys=2, key_format='')

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.binary_var_dict(keys=range(2), key_format='')

    def test_var_list(self):
        """test {binary,integer,continuous}_var_list"""
        q_p = QuadraticProgram()

        d_0 = q_p.continuous_var_list(name='a', key_format='_{}', keys=3)
        names = ['a_0', 'a_1', 'a_2']
        self.assertSetEqual({var.name for var in q_p.variables}, {'a_0', 'a_1', 'a_2'})
        for i, var in enumerate(q_p.variables):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, INFINITY)
            self.assertEqual(var.vartype, VarType.CONTINUOUS)
            self.assertTupleEqual(var.as_tuple(), d_0[i].as_tuple())

        d_1 = q_p.binary_var_list(name='b', keys=5)
        names = ['b3', 'b4', 'b5', 'b6', 'b7']
        self.assertSetEqual({var.name for var in q_p.variables},
                            {'a_0', 'a_1', 'a_2', 'b3', 'b4', 'b5', 'b6', 'b7'})
        for i, var in enumerate(q_p.variables[-5:]):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_1[i].as_tuple())

        d_2 = q_p.integer_var_list(keys=1, key_format='-{}', lowerbound=-4, upperbound=10)
        names = ['x-8']
        self.assertSetEqual({var.name for var in q_p.variables},
                            {'a_0', 'a_1', 'a_2', 'b3', 'b4', 'b5', 'b6', 'b7', 'x-8'})
        for i, var in enumerate(q_p.variables[-1:]):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, -4)
            self.assertAlmostEqual(var.upperbound, 10)
            self.assertEqual(var.vartype, VarType.INTEGER)
            self.assertTupleEqual(var.as_tuple(), d_2[i].as_tuple())

        d_3 = q_p.binary_var_list(name='c', keys=range(3))
        names = ['c0', 'c1', 'c2']
        self.assertSetEqual({var.name for var in q_p.variables},
                            {'a_0', 'a_1', 'a_2', 'b3', 'b4', 'b5', 'b6', 'b7', 'x-8',
                             'c0', 'c1', 'c2'})
        for i, var in enumerate(q_p.variables[-3:]):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_3[i].as_tuple())

        with self.assertRaises(QiskitOptimizationError):
            q_p.binary_var_list(name='c', keys=range(3))

        d_4 = q_p.binary_var_dict(1, 'x', '_')
        names = ['x_']
        self.assertSetEqual({var.name for var in q_p.variables},
                            {'a_0', 'a_1', 'a_2', 'b3', 'b4', 'b5', 'b6', 'b7', 'x-8',
                             'c0', 'c1', 'c2', 'x_'})
        for i, var in enumerate(q_p.variables[-1:]):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, 0)
            self.assertAlmostEqual(var.upperbound, 1)
            self.assertEqual(var.vartype, VarType.BINARY)
            self.assertTupleEqual(var.as_tuple(), d_4[var.name].as_tuple())

        with self.assertRaises(QiskitOptimizationError):
            q_p.binary_var_list(1, 'x', '_')

        with self.assertRaises(QiskitOptimizationError):
            q_p.binary_var('x_')

        d_5 = q_p.integer_var_list(1, -1, 2, '', '')
        names = ['']
        self.assertSetEqual({var.name for var in q_p.variables},
                            {'a_0', 'a_1', 'a_2', 'b3', 'b4', 'b5', 'b6', 'b7', 'x-8',
                             'c0', 'c1', 'c2', 'x_', ''})
        for i, var in enumerate(q_p.variables[-1:]):
            self.assertEqual(var.name, names[i])
            self.assertAlmostEqual(var.lowerbound, -1)
            self.assertAlmostEqual(var.upperbound, 2)
            self.assertEqual(var.vartype, VarType.INTEGER)
            self.assertTupleEqual(var.as_tuple(), d_5[i].as_tuple())

        with self.assertRaises(QiskitOptimizationError):
            q_p.binary_var_list(1, '', '')

        with self.assertRaises(QiskitOptimizationError):
            q_p.integer_var(0, 1, '')

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.binary_var_list(keys=1, key_format='{}{}')

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.binary_var_list(keys=0)

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.binary_var_list(keys=1, key_format='_{{}}')

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.binary_var_list(keys=2, key_format='')

        with self.assertRaises(QiskitOptimizationError):
            q_p = QuadraticProgram()
            q_p.binary_var_list(keys=range(2), key_format='')

    def test_variables_handling(self):
        """ test add variables """
        quadratic_program = QuadraticProgram()

        self.assertEqual(quadratic_program.get_num_vars(), 0)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 0)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 0)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_0 = quadratic_program.continuous_var()
        self.assertEqual(x_0.name, 'x0')
        self.assertEqual(x_0.lowerbound, 0)
        self.assertEqual(x_0.upperbound, INFINITY)
        self.assertEqual(x_0.vartype, Variable.Type.CONTINUOUS)

        self.assertEqual(quadratic_program.get_num_vars(), 1)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 1)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 0)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_1 = quadratic_program.continuous_var(name='x1', lowerbound=5, upperbound=10)
        self.assertEqual(x_1.name, 'x1')
        self.assertEqual(x_1.lowerbound, 5)
        self.assertEqual(x_1.upperbound, 10)
        self.assertEqual(x_1.vartype, Variable.Type.CONTINUOUS)

        self.assertEqual(quadratic_program.get_num_vars(), 2)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 0)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_2 = quadratic_program.binary_var()
        self.assertEqual(x_2.name, 'x2')
        self.assertEqual(x_2.lowerbound, 0)
        self.assertEqual(x_2.upperbound, 1)
        self.assertEqual(x_2.vartype, Variable.Type.BINARY)

        self.assertEqual(quadratic_program.get_num_vars(), 3)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 1)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_3 = quadratic_program.binary_var(name='x3')
        self.assertEqual(x_3.name, 'x3')
        self.assertEqual(x_3.lowerbound, 0)
        self.assertEqual(x_3.upperbound, 1)
        self.assertEqual(x_3.vartype, Variable.Type.BINARY)

        self.assertEqual(quadratic_program.get_num_vars(), 4)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 2)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 0)

        x_4 = quadratic_program.integer_var()
        self.assertEqual(x_4.name, 'x4')
        self.assertEqual(x_4.lowerbound, 0)
        self.assertEqual(x_4.upperbound, INFINITY)
        self.assertEqual(x_4.vartype, Variable.Type.INTEGER)

        self.assertEqual(quadratic_program.get_num_vars(), 5)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 2)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 1)

        x_5 = quadratic_program.integer_var(name='x5', lowerbound=5, upperbound=10)
        self.assertEqual(x_5.name, 'x5')
        self.assertEqual(x_5.lowerbound, 5)
        self.assertEqual(x_5.upperbound, 10)
        self.assertEqual(x_5.vartype, Variable.Type.INTEGER)

        self.assertEqual(quadratic_program.get_num_vars(), 6)
        self.assertEqual(quadratic_program.get_num_continuous_vars(), 2)
        self.assertEqual(quadratic_program.get_num_binary_vars(), 2)
        self.assertEqual(quadratic_program.get_num_integer_vars(), 2)

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.continuous_var(name='x0')

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.binary_var(name='x0')

        with self.assertRaises(QiskitOptimizationError):
            quadratic_program.integer_var(name='x0')

        variables = [x_0, x_1, x_2, x_3, x_4, x_5]
        for i, x in enumerate(variables):
            y = quadratic_program.get_variable(i)
            z = quadratic_program.get_variable(x.name)
            self.assertEqual(x.name, y.name)
            self.assertEqual(x.name, z.name)
        self.assertDictEqual(quadratic_program.variables_index,
                             {'x' + str(i): i for i in range(6)})

    def test_linear_constraints_handling(self):
        """test linear constraints handling"""
        q_p = QuadraticProgram()
        q_p.binary_var('x')
        q_p.binary_var('y')
        q_p.binary_var('z')
        q_p.linear_constraint({'x': 1}, '==', 1)
        q_p.linear_constraint({'y': 1}, '<=', 1)
        q_p.linear_constraint({'z': 1}, '>=', 1)
        self.assertEqual(q_p.get_num_linear_constraints(), 3)
        lin = q_p.linear_constraints
        self.assertEqual(len(lin), 3)

        self.assertDictEqual(lin[0].linear.to_dict(), {0: 1})
        self.assertDictEqual(lin[0].linear.to_dict(use_name=True), {'x': 1})
        self.assertListEqual(lin[0].linear.to_array().tolist(), [1, 0, 0])
        self.assertEqual(lin[0].sense, Constraint.Sense.EQ)
        self.assertEqual(lin[0].rhs, 1)
        self.assertEqual(lin[0].name, 'c0')
        self.assertEqual(q_p.get_linear_constraint(0).name, 'c0')
        self.assertEqual(q_p.get_linear_constraint('c0').name, 'c0')

        self.assertDictEqual(lin[1].linear.to_dict(), {1: 1})
        self.assertDictEqual(lin[1].linear.to_dict(use_name=True), {'y': 1})
        self.assertListEqual(lin[1].linear.to_array().tolist(), [0, 1, 0])
        self.assertEqual(lin[1].sense, Constraint.Sense.LE)
        self.assertEqual(lin[1].rhs, 1)
        self.assertEqual(lin[1].name, 'c1')
        self.assertEqual(q_p.get_linear_constraint(1).name, 'c1')
        self.assertEqual(q_p.get_linear_constraint('c1').name, 'c1')

        self.assertDictEqual(lin[2].linear.to_dict(), {2: 1})
        self.assertDictEqual(lin[2].linear.to_dict(use_name=True), {'z': 1})
        self.assertListEqual(lin[2].linear.to_array().tolist(), [0, 0, 1])
        self.assertEqual(lin[2].sense, Constraint.Sense.GE)
        self.assertEqual(lin[2].rhs, 1)
        self.assertEqual(lin[2].name, 'c2')
        self.assertEqual(q_p.get_linear_constraint(2).name, 'c2')
        self.assertEqual(q_p.get_linear_constraint('c2').name, 'c2')

        with self.assertRaises(QiskitOptimizationError):
            q_p.linear_constraint(name='c0')
        with self.assertRaises(QiskitOptimizationError):
            q_p.linear_constraint(name='c1')
        with self.assertRaises(QiskitOptimizationError):
            q_p.linear_constraint(name='c2')
        with self.assertRaises(IndexError):
            q_p.get_linear_constraint(4)
        with self.assertRaises(KeyError):
            q_p.get_linear_constraint('c3')

        q_p.remove_linear_constraint('c1')
        lin = q_p.linear_constraints
        self.assertEqual(len(lin), 2)
        self.assertDictEqual(lin[1].linear.to_dict(), {2: 1})
        self.assertDictEqual(lin[1].linear.to_dict(use_name=True), {'z': 1})
        self.assertListEqual(lin[1].linear.to_array().tolist(), [0, 0, 1])
        self.assertEqual(lin[1].sense, Constraint.Sense.GE)
        self.assertEqual(lin[1].rhs, 1)
        self.assertEqual(lin[1].name, 'c2')
        self.assertEqual(q_p.get_linear_constraint(1).name, 'c2')
        self.assertEqual(q_p.get_linear_constraint('c2').name, 'c2')

        with self.assertRaises(KeyError):
            q_p.remove_linear_constraint('c1')
        with self.assertRaises(IndexError):
            q_p.remove_linear_constraint(9)

        q_p.linear_constraint(sense='E')
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.EQ)
        q_p.linear_constraint(sense='G')
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.GE)
        q_p.linear_constraint(sense='L')
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.LE)
        q_p.linear_constraint(sense='EQ')
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.EQ)
        q_p.linear_constraint(sense='GE')
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.GE)
        q_p.linear_constraint(sense='LE')
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.LE)
        q_p.linear_constraint(sense='=')
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.EQ)
        q_p.linear_constraint(sense='>')
        self.assertEqual(q_p.linear_constraints[-1].sense, Constraint.Sense.GE)
        q_p.linear_constraint(sense='<')

        with self.assertRaises(QiskitOptimizationError):
            q_p.linear_constraint(sense='=>')

    def test_quadratic_constraints_handling(self):
        """test quadratic constraints handling"""
        q_p = QuadraticProgram()
        q_p.binary_var('x')
        q_p.binary_var('y')
        q_p.binary_var('z')
        q_p.quadratic_constraint({'x': 1}, {('x', 'y'): 1}, '==', 1)
        q_p.quadratic_constraint({'y': 1}, {('y', 'z'): 1}, '<=', 1)
        q_p.quadratic_constraint({'z': 1}, {('z', 'x'): 1}, '>=', 1)
        self.assertEqual(q_p.get_num_quadratic_constraints(), 3)
        quad = q_p.quadratic_constraints
        self.assertEqual(len(quad), 3)

        self.assertDictEqual(quad[0].linear.to_dict(), {0: 1})
        self.assertDictEqual(quad[0].linear.to_dict(use_name=True), {'x': 1})
        self.assertListEqual(quad[0].linear.to_array().tolist(), [1, 0, 0])
        self.assertDictEqual(quad[0].quadratic.to_dict(), {(0, 1): 1})
        self.assertDictEqual(quad[0].quadratic.to_dict(symmetric=True),
                             {(0, 1): 0.5, (1, 0): 0.5})
        self.assertDictEqual(quad[0].quadratic.to_dict(use_name=True), {('x', 'y'): 1})
        self.assertDictEqual(quad[0].quadratic.to_dict(use_name=True, symmetric=True),
                             {('x', 'y'): 0.5, ('y', 'x'): 0.5})
        self.assertListEqual(quad[0].quadratic.to_array().tolist(),
                             [[0, 1, 0], [0, 0, 0], [0, 0, 0]])
        self.assertListEqual(quad[0].quadratic.to_array(symmetric=True).tolist(),
                             [[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]])
        self.assertEqual(quad[0].sense, Constraint.Sense.EQ)
        self.assertEqual(quad[0].rhs, 1)
        self.assertEqual(quad[0].name, 'q0')
        self.assertEqual(q_p.get_quadratic_constraint(0).name, 'q0')
        self.assertEqual(q_p.get_quadratic_constraint('q0').name, 'q0')

        self.assertDictEqual(quad[1].linear.to_dict(), {1: 1})
        self.assertDictEqual(quad[1].linear.to_dict(use_name=True), {'y': 1})
        self.assertListEqual(quad[1].linear.to_array().tolist(), [0, 1, 0])
        self.assertDictEqual(quad[1].quadratic.to_dict(), {(1, 2): 1})
        self.assertDictEqual(quad[1].quadratic.to_dict(symmetric=True),
                             {(1, 2): 0.5, (2, 1): 0.5})
        self.assertDictEqual(quad[1].quadratic.to_dict(use_name=True), {('y', 'z'): 1})
        self.assertDictEqual(quad[1].quadratic.to_dict(use_name=True, symmetric=True),
                             {('y', 'z'): 0.5, ('z', 'y'): 0.5})
        self.assertListEqual(quad[1].quadratic.to_array().tolist(),
                             [[0, 0, 0], [0, 0, 1], [0, 0, 0]])
        self.assertListEqual(quad[1].quadratic.to_array(symmetric=True).tolist(),
                             [[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]])
        self.assertEqual(quad[1].sense, Constraint.Sense.LE)
        self.assertEqual(quad[1].rhs, 1)
        self.assertEqual(quad[1].name, 'q1')
        self.assertEqual(q_p.get_quadratic_constraint(1).name, 'q1')
        self.assertEqual(q_p.get_quadratic_constraint('q1').name, 'q1')

        self.assertDictEqual(quad[2].linear.to_dict(), {2: 1})
        self.assertDictEqual(quad[2].linear.to_dict(use_name=True), {'z': 1})
        self.assertListEqual(quad[2].linear.to_array().tolist(), [0, 0, 1])
        self.assertDictEqual(quad[2].quadratic.to_dict(), {(0, 2): 1})
        self.assertDictEqual(quad[2].quadratic.to_dict(symmetric=True),
                             {(0, 2): 0.5, (2, 0): 0.5})
        self.assertDictEqual(quad[2].quadratic.to_dict(use_name=True), {('x', 'z'): 1})
        self.assertDictEqual(quad[2].quadratic.to_dict(use_name=True, symmetric=True),
                             {('x', 'z'): 0.5, ('z', 'x'): 0.5})
        self.assertListEqual(quad[2].quadratic.to_array().tolist(),
                             [[0, 0, 1], [0, 0, 0], [0, 0, 0]])
        self.assertListEqual(quad[2].quadratic.to_array(symmetric=True).tolist(),
                             [[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]])
        self.assertEqual(quad[2].sense, Constraint.Sense.GE)
        self.assertEqual(quad[2].rhs, 1)
        self.assertEqual(quad[2].name, 'q2')
        self.assertEqual(q_p.get_quadratic_constraint(2).name, 'q2')
        self.assertEqual(q_p.get_quadratic_constraint('q2').name, 'q2')

        with self.assertRaises(QiskitOptimizationError):
            q_p.quadratic_constraint(name='q0')
        with self.assertRaises(QiskitOptimizationError):
            q_p.quadratic_constraint(name='q1')
        with self.assertRaises(QiskitOptimizationError):
            q_p.quadratic_constraint(name='q2')
        with self.assertRaises(IndexError):
            q_p.get_quadratic_constraint(4)
        with self.assertRaises(KeyError):
            q_p.get_quadratic_constraint('q3')

        q_p.remove_quadratic_constraint('q1')
        quad = q_p.quadratic_constraints
        self.assertEqual(len(quad), 2)
        self.assertDictEqual(quad[1].linear.to_dict(), {2: 1})
        self.assertDictEqual(quad[1].linear.to_dict(use_name=True), {'z': 1})
        self.assertListEqual(quad[1].linear.to_array().tolist(), [0, 0, 1])
        self.assertDictEqual(quad[1].quadratic.to_dict(), {(0, 2): 1})
        self.assertDictEqual(quad[1].quadratic.to_dict(symmetric=True),
                             {(0, 2): 0.5, (2, 0): 0.5})
        self.assertDictEqual(quad[1].quadratic.to_dict(use_name=True), {('x', 'z'): 1})
        self.assertDictEqual(quad[1].quadratic.to_dict(use_name=True, symmetric=True),
                             {('x', 'z'): 0.5, ('z', 'x'): 0.5})
        self.assertListEqual(quad[1].quadratic.to_array().tolist(),
                             [[0, 0, 1], [0, 0, 0], [0, 0, 0]])
        self.assertListEqual(quad[1].quadratic.to_array(symmetric=True).tolist(),
                             [[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]])
        self.assertEqual(quad[1].sense, Constraint.Sense.GE)
        self.assertEqual(quad[1].rhs, 1)
        self.assertEqual(quad[1].name, 'q2')
        self.assertEqual(q_p.get_quadratic_constraint(1).name, 'q2')
        self.assertEqual(q_p.get_quadratic_constraint('q2').name, 'q2')

        with self.assertRaises(KeyError):
            q_p.remove_quadratic_constraint('q1')
        with self.assertRaises(IndexError):
            q_p.remove_quadratic_constraint(9)

    def test_objective_handling(self):
        """test objective handling"""
        q_p = QuadraticProgram()
        q_p.binary_var('x')
        q_p.binary_var('y')
        q_p.binary_var('z')
        q_p.minimize()
        obj = q_p.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MINIMIZE)
        self.assertEqual(obj.constant, 0)
        self.assertDictEqual(obj.linear.to_dict(), {})
        self.assertDictEqual(obj.quadratic.to_dict(), {})
        q_p.maximize(1, {'y': 1}, {('z', 'x'): 1, ('y', 'y'): 1})
        obj = q_p.objective
        self.assertEqual(obj.sense, QuadraticObjective.Sense.MAXIMIZE)
        self.assertEqual(obj.constant, 1)
        self.assertDictEqual(obj.linear.to_dict(), {1: 1})
        self.assertDictEqual(obj.linear.to_dict(use_name=True), {'y': 1})
        self.assertListEqual(obj.linear.to_array().tolist(), [0, 1, 0])
        self.assertDictEqual(obj.quadratic.to_dict(), {(0, 2): 1, (1, 1): 1})
        self.assertDictEqual(obj.quadratic.to_dict(symmetric=True),
                             {(0, 2): 0.5, (2, 0): 0.5, (1, 1): 1})
        self.assertDictEqual(obj.quadratic.to_dict(use_name=True),
                             {('x', 'z'): 1, ('y', 'y'): 1})
        self.assertDictEqual(obj.quadratic.to_dict(use_name=True, symmetric=True),
                             {('x', 'z'): 0.5, ('z', 'x'): 0.5, ('y', 'y'): 1})
        self.assertListEqual(obj.quadratic.to_array().tolist(),
                             [[0, 0, 1], [0, 1, 0], [0, 0, 0]])
        self.assertListEqual(obj.quadratic.to_array(symmetric=True).tolist(),
                             [[0, 0, 0.5], [0, 1, 0], [0.5, 0, 0]])

    def test_read_from_lp_file(self):
        """test read lp file"""
        try:
            q_p = QuadraticProgram()
            with self.assertRaises(FileNotFoundError):
                q_p.read_from_lp_file('')
            with self.assertRaises(FileNotFoundError):
                q_p.read_from_lp_file('no_file.txt')
            lp_file = self.get_resource_path(path.join('resources', 'test_quadratic_program.lp'))
            q_p.read_from_lp_file(lp_file)
            self.assertEqual(q_p.name, 'my problem')
            self.assertEqual(q_p.get_num_vars(), 3)
            self.assertEqual(q_p.get_num_binary_vars(), 1)
            self.assertEqual(q_p.get_num_integer_vars(), 1)
            self.assertEqual(q_p.get_num_continuous_vars(), 1)
            self.assertEqual(q_p.get_num_linear_constraints(), 3)
            self.assertEqual(q_p.get_num_quadratic_constraints(), 3)

            self.assertEqual(q_p.variables[0].name, 'x')
            self.assertEqual(q_p.variables[0].vartype, Variable.Type.BINARY)
            self.assertEqual(q_p.variables[0].lowerbound, 0)
            self.assertEqual(q_p.variables[0].upperbound, 1)
            self.assertEqual(q_p.variables[1].name, 'y')
            self.assertEqual(q_p.variables[1].vartype, Variable.Type.INTEGER)
            self.assertEqual(q_p.variables[1].lowerbound, -1)
            self.assertEqual(q_p.variables[1].upperbound, 5)
            self.assertEqual(q_p.variables[2].name, 'z')
            self.assertEqual(q_p.variables[2].vartype, Variable.Type.CONTINUOUS)
            self.assertEqual(q_p.variables[2].lowerbound, -1)
            self.assertEqual(q_p.variables[2].upperbound, 5)

            self.assertEqual(q_p.objective.sense, QuadraticObjective.Sense.MINIMIZE)
            self.assertEqual(q_p.objective.constant, 1)
            self.assertDictEqual(q_p.objective.linear.to_dict(use_name=True),
                                 {'x': 1, 'y': -1, 'z': 10})
            self.assertDictEqual(q_p.objective.quadratic.to_dict(use_name=True),
                                 {('x', 'x'): 0.5, ('y', 'z'): -1})

            cst = q_p.linear_constraints
            self.assertEqual(cst[0].name, 'lin_eq')
            self.assertDictEqual(cst[0].linear.to_dict(use_name=True), {'x': 1, 'y': 2})
            self.assertEqual(cst[0].sense, Constraint.Sense.EQ)
            self.assertEqual(cst[0].rhs, 1)
            self.assertEqual(cst[1].name, 'lin_leq')
            self.assertDictEqual(cst[1].linear.to_dict(use_name=True), {'x': 1, 'y': 2})
            self.assertEqual(cst[1].sense, Constraint.Sense.LE)
            self.assertEqual(cst[1].rhs, 1)
            self.assertEqual(cst[2].name, 'lin_geq')
            self.assertDictEqual(cst[2].linear.to_dict(use_name=True), {'x': 1, 'y': 2})
            self.assertEqual(cst[2].sense, Constraint.Sense.GE)
            self.assertEqual(cst[2].rhs, 1)

            cst = q_p.quadratic_constraints
            self.assertEqual(cst[0].name, 'quad_eq')
            self.assertDictEqual(cst[0].linear.to_dict(use_name=True), {'x': 1, 'y': 1})
            self.assertDictEqual(cst[0].quadratic.to_dict(use_name=True),
                                 {('x', 'x'): 1, ('y', 'z'): -1, ('z', 'z'): 2})
            self.assertEqual(cst[0].sense, Constraint.Sense.EQ)
            self.assertEqual(cst[0].rhs, 1)
            self.assertEqual(cst[1].name, 'quad_leq')
            self.assertDictEqual(cst[1].linear.to_dict(use_name=True), {'x': 1, 'y': 1})
            self.assertDictEqual(cst[1].quadratic.to_dict(use_name=True),
                                 {('x', 'x'): 1, ('y', 'z'): -1, ('z', 'z'): 2})
            self.assertEqual(cst[1].sense, Constraint.Sense.LE)
            self.assertEqual(cst[1].rhs, 1)
            self.assertEqual(cst[2].name, 'quad_geq')
            self.assertDictEqual(cst[2].linear.to_dict(use_name=True), {'x': 1, 'y': 1})
            self.assertDictEqual(cst[2].quadratic.to_dict(use_name=True),
                                 {('x', 'x'): 1, ('y', 'z'): -1, ('z', 'z'): 2})
            self.assertEqual(cst[2].sense, Constraint.Sense.GE)
            self.assertEqual(cst[2].rhs, 1)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))
        except RuntimeError as ex:
            self.fail(str(ex))

    def test_write_to_lp_file(self):
        """test write problem"""
        q_p = QuadraticProgram('my problem')
        q_p.binary_var('x')
        q_p.integer_var(-1, 5, 'y')
        q_p.continuous_var(-1, 5, 'z')
        q_p.minimize(1, {'x': 1, 'y': -1, 'z': 10}, {('x', 'x'): 0.5, ('y', 'z'): -1})
        q_p.linear_constraint({'x': 1, 'y': 2}, '==', 1, 'lin_eq')
        q_p.linear_constraint({'x': 1, 'y': 2}, '<=', 1, 'lin_leq')
        q_p.linear_constraint({'x': 1, 'y': 2}, '>=', 1, 'lin_geq')
        q_p.quadratic_constraint({'x': 1, 'y': 1}, {('x', 'x'): 1, ('y', 'z'): -1, ('z', 'z'): 2},
                                 '==', 1, 'quad_eq')
        q_p.quadratic_constraint({'x': 1, 'y': 1}, {('x', 'x'): 1, ('y', 'z'): -1, ('z', 'z'): 2},
                                 '<=', 1, 'quad_leq')
        q_p.quadratic_constraint({'x': 1, 'y': 1}, {('x', 'x'): 1, ('y', 'z'): -1, ('z', 'z'): 2},
                                 '>=', 1, 'quad_geq')

        reference_file_name = self.get_resource_path(path.join('resources',
                                                               'test_quadratic_program.lp'))
        with tempfile.NamedTemporaryFile(mode='w+t', suffix='.lp') as temp_output_file:
            q_p.write_to_lp_file(temp_output_file.name)
            with open(reference_file_name) as reference:
                lines1 = temp_output_file.readlines()
                lines2 = reference.readlines()
                self.assertListEqual(lines1, lines2)

        with tempfile.TemporaryDirectory() as temp_problem_dir:
            q_p.write_to_lp_file(temp_problem_dir)
            with open(path.join(temp_problem_dir, 'my_problem.lp')) as file1, open(
                    reference_file_name) as file2:
                lines1 = file1.readlines()
                lines2 = file2.readlines()
                self.assertListEqual(lines1, lines2)

        with self.assertRaises(OSError):
            q_p.write_to_lp_file('/cannot/write/this/file.lp')

        with self.assertRaises(DOcplexException):
            q_p.write_to_lp_file('')

    def test_docplex(self):
        """test from_docplex and to_docplex"""
        q_p = QuadraticProgram('test')
        q_p.binary_var(name='x')
        q_p.integer_var(name='y', lowerbound=-2, upperbound=4)
        q_p.continuous_var(name='z', lowerbound=-1.5, upperbound=3.2)
        q_p.minimize(constant=1, linear={'x': 1, 'y': 2},
                     quadratic={('x', 'y'): -1, ('z', 'z'): 2})
        q_p.linear_constraint({'x': 2, 'z': -1}, '==', 1)
        q_p.quadratic_constraint({'x': 2, 'z': -1}, {('y', 'z'): 3}, '==', 1)
        q_p2 = QuadraticProgram()
        q_p2.from_docplex(q_p.to_docplex())
        self.assertEqual(q_p.export_as_lp_string(), q_p2.export_as_lp_string())

        mod = Model('test')
        x = mod.binary_var('x')
        y = mod.integer_var(-2, 4, 'y')
        z = mod.continuous_var(-1.5, 3.2, 'z')
        mod.minimize(1 + x + 2 * y - x * y + 2 * z * z)
        mod.add(2 * x - z == 1, 'c0')
        mod.add(2 * x - z + 3 * y * z == 1, 'q0')
        self.assertEqual(q_p.export_as_lp_string(), mod.export_as_lp_string())

        with self.assertRaises(QiskitOptimizationError):
            mod = Model()
            mod.semiinteger_var(lb=1, name='x')
            q_p.from_docplex(mod)

        with self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var('x')
            mod.add_range(0, 2 * x, 1)
            q_p.from_docplex(mod)

        with self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var('x')
            y = mod.binary_var('y')
            mod.add_indicator(x, x + y <= 1, 1)
            q_p.from_docplex(mod)

        with self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var('x')
            y = mod.binary_var('y')
            mod.add_equivalence(x, x + y <= 1, 1)
            q_p.from_docplex(mod)

        with self.assertRaises(QiskitOptimizationError):
            mod = Model()
            x = mod.binary_var('x')
            y = mod.binary_var('y')
            mod.add(mod.not_equal_constraint(x, y + 1))
            q_p.from_docplex(mod)

        # test from_docplex without explicit variable names
        mod = Model()
        x = mod.binary_var()
        y = mod.continuous_var()
        z = mod.integer_var()
        mod.minimize(x + y + z + x * y + y * z + x * z)
        mod.add_constraint(x + y == z)  # linear EQ
        mod.add_constraint(x + y >= z)  # linear GE
        mod.add_constraint(x + y <= z)  # linear LE
        mod.add_constraint(x * y == z)  # quadratic EQ
        mod.add_constraint(x * y >= z)  # quadratic GE
        mod.add_constraint(x * y <= z)  # quadratic LE
        q_p = QuadraticProgram()
        q_p.from_docplex(mod)
        var_names = [v.name for v in q_p.variables]
        self.assertListEqual(var_names, ['x0', 'x1', 'x2'])
        senses = [Constraint.Sense.EQ, Constraint.Sense.GE, Constraint.Sense.LE]
        for i, c in enumerate(q_p.linear_constraints):
            self.assertDictEqual(c.linear.to_dict(use_name=True), {'x0': 1, 'x1': 1, 'x2': -1})
            self.assertEqual(c.rhs, 0)
            self.assertEqual(c.sense, senses[i])
        for i, c in enumerate(q_p.quadratic_constraints):
            self.assertEqual(c.rhs, 0)
            self.assertDictEqual(c.linear.to_dict(use_name=True), {'x2': -1})
            self.assertDictEqual(c.quadratic.to_dict(use_name=True), {('x0', 'x1'): 1})
            self.assertEqual(c.sense, senses[i])

    def test_substitute_variables(self):
        """test substitute variables"""
        q_p = QuadraticProgram('test')
        q_p.binary_var(name='x')
        q_p.integer_var(name='y', lowerbound=-2, upperbound=4)
        q_p.continuous_var(name='z', lowerbound=-1.5, upperbound=3.2)
        q_p.minimize(constant=1, linear={'x': 1, 'y': 2},
                     quadratic={('x', 'y'): -1, ('z', 'z'): 2})
        q_p.linear_constraint({'x': 2, 'z': -1}, '==', 1)
        q_p.quadratic_constraint({'x': 2, 'z': -1}, {('y', 'z'): 3}, '<=', -1)

        q_p2 = q_p.substitute_variables(constants={'x': -1})
        self.assertEqual(q_p2.status, QuadraticProgram.Status.INFEASIBLE)
        q_p2 = q_p.substitute_variables(constants={'y': -3})
        self.assertEqual(q_p2.status, QuadraticProgram.Status.INFEASIBLE)
        q_p2 = q_p.substitute_variables(constants={'x': 1, 'z': 2})
        self.assertEqual(q_p2.status, QuadraticProgram.Status.INFEASIBLE)
        q_p2.clear()
        self.assertEqual(q_p2.status, QuadraticProgram.Status.VALID)

        q_p2 = q_p.substitute_variables(constants={'x': 0})
        self.assertEqual(q_p2.status, QuadraticProgram.Status.VALID)
        self.assertDictEqual(q_p2.objective.linear.to_dict(use_name=True), {'y': 2})
        self.assertDictEqual(q_p2.objective.quadratic.to_dict(use_name=True), {('z', 'z'): 2})
        self.assertEqual(q_p2.objective.constant, 1)
        self.assertEqual(len(q_p2.linear_constraints), 1)
        self.assertEqual(len(q_p2.quadratic_constraints), 1)

        cst = q_p2.linear_constraints[0]
        self.assertDictEqual(cst.linear.to_dict(use_name=True), {'z': -1})
        self.assertEqual(cst.sense.name, 'EQ')
        self.assertEqual(cst.rhs, 1)

        cst = q_p2.quadratic_constraints[0]
        self.assertDictEqual(cst.linear.to_dict(use_name=True), {'z': -1})
        self.assertDictEqual(cst.quadratic.to_dict(use_name=True), {('y', 'z'): 3})
        self.assertEqual(cst.sense.name, 'LE')
        self.assertEqual(cst.rhs, -1)

        q_p2 = q_p.substitute_variables(constants={'z': -1})
        self.assertEqual(q_p2.status, QuadraticProgram.Status.VALID)
        self.assertDictEqual(q_p2.objective.linear.to_dict(use_name=True), {'x': 1, 'y': 2})
        self.assertDictEqual(q_p2.objective.quadratic.to_dict(use_name=True), {('x', 'y'): -1})
        self.assertEqual(q_p2.objective.constant, 3)
        self.assertEqual(len(q_p2.linear_constraints), 2)
        self.assertEqual(len(q_p2.quadratic_constraints), 0)

        cst = q_p2.linear_constraints[0]
        self.assertDictEqual(cst.linear.to_dict(use_name=True), {'x': 2})
        self.assertEqual(cst.sense.name, 'EQ')
        self.assertEqual(cst.rhs, 0)

        cst = q_p2.linear_constraints[1]
        self.assertDictEqual(cst.linear.to_dict(use_name=True), {'x': 2, 'y': -3})
        self.assertEqual(cst.sense.name, 'LE')
        self.assertEqual(cst.rhs, -2)

        q_p2 = q_p.substitute_variables(variables={'y': ('x', -0.5)})
        self.assertEqual(q_p2.status, QuadraticProgram.Status.VALID)
        self.assertDictEqual(q_p2.objective.linear.to_dict(use_name=True), {})
        self.assertDictEqual(q_p2.objective.quadratic.to_dict(use_name=True),
                             {('x', 'x'): 0.5, ('z', 'z'): 2})
        self.assertEqual(q_p2.objective.constant, 1)
        self.assertEqual(len(q_p2.linear_constraints), 1)
        self.assertEqual(len(q_p2.quadratic_constraints), 1)

        cst = q_p2.linear_constraints[0]
        self.assertDictEqual(cst.linear.to_dict(use_name=True), {'x': 2, 'z': -1})
        self.assertEqual(cst.sense.name, 'EQ')
        self.assertEqual(cst.rhs, 1)

        cst = q_p2.quadratic_constraints[0]
        self.assertDictEqual(cst.linear.to_dict(use_name=True), {'x': 2, 'z': -1})
        self.assertDictEqual(cst.quadratic.to_dict(use_name=True), {('x', 'z'): -1.5})
        self.assertEqual(cst.sense.name, 'LE')
        self.assertEqual(cst.rhs, -1)

    def test_feasibility(self):
        """Tests feasibility methods."""
        mod = Model('test')
        # 0, 5
        x = mod.continuous_var(-1, 1, 'x', )
        y = mod.continuous_var(-10, 10, 'y')
        mod.minimize(x + y)
        mod.add(x + y <= 10, 'c0')
        mod.add(x + y >= -10, 'c1')
        mod.add(x + y == 5, 'c2')
        mod.add(x * x + y <= 10, 'c3')
        mod.add(x * x + y >= 5, 'c4')
        mod.add(x * x + y * y == 25, 'c5')
        q_p = QuadraticProgram()
        q_p.from_docplex(mod)

        self.assertTrue(q_p.is_feasible([0, 5]))
        self.assertFalse(q_p.is_feasible([1, 10]))
        self.assertFalse(q_p.is_feasible([1, -12]))
        self.assertFalse(q_p.is_feasible([1, 5]))
        self.assertFalse(q_p.is_feasible([5, 0]))
        self.assertFalse(q_p.is_feasible([1, 1]))
        self.assertFalse(q_p.is_feasible([0, 0]))

        feasible, variables, constraints = q_p.get_feasibility_info([10, 0])
        self.assertFalse(feasible)
        self.assertIsNotNone(variables)
        self.assertEqual(1, len(variables))
        self.assertEqual('x', variables[0].name)

        self.assertIsNotNone(constraints)
        self.assertEqual(3, len(constraints))
        self.assertEqual('c2', constraints[0].name)
        self.assertEqual('c3', constraints[1].name)
        self.assertEqual('c5', constraints[2].name)


if __name__ == '__main__':
    unittest.main()
