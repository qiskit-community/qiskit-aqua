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

""" Test OptimizationProblem """

import unittest
import os.path
import tempfile
from test.optimization.optimization_test_case import QiskitOptimizationTestCase
import qiskit.optimization.problems.optimization_problem
from qiskit.optimization.utils.qiskit_optimization_error import QiskitOptimizationError


class TestOptimizationProblem(QiskitOptimizationTestCase):
    """Test OptimizationProblem without the members that have separate test classes
    (VariablesInterface, etc)."""

    def setUp(self):
        super().setUp()
        self.resource_file = './test/optimization/resources/op_ip2.lp'

    def test_constructor1(self):
        """ test constructor """
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(names=['x1', 'x2', 'x3'])
        self.assertEqual(op.variables.get_num(), 3)

    def test_constructor2(self):
        """ test constructor 2 """
        with self.assertRaises(QiskitOptimizationError):
            op = qiskit.optimization.OptimizationProblem("unknown")
        # If filename does not exist, an exception is raised.

    def test_constructor3(self):
        """ test constructor 3 """
        # we can pass at most one argument
        with self.assertRaises(QiskitOptimizationError):
            op = qiskit.optimization.OptimizationProblem("test", "west")

    def test_constructor_context(self):
        """ test constructor context """
        with qiskit.optimization.OptimizationProblem() as op:
            op.variables.add(names=['x1', 'x2', 'x3'])
            self.assertEqual(op.variables.get_num(), 3)

    # def test_end(self):
    #     op = qiskit.optimization.OptimizationProblem()
    #     op.end()
        # TODO: we do not need to release the object
        # with self.assertRaises(QiskitOptimizationError):
        #     op.variables.add(names=['x1', 'x2', 'x3'])

    def test_read1(self):
        """ test read 2"""
        op = qiskit.optimization.OptimizationProblem()
        op.read(self.resource_file)
        self.assertEqual(op.variables.get_num(), 3)

    def test_write1(self):
        """ test write 1 """
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(names=['x1', 'x2', 'x3'])
        f, fn = tempfile.mkstemp(suffix='.lp')
        os.close(f)
        op.write(fn)
        assert os.path.exists(fn) == 1

    def test_write2(self):
        """ test write 2 """
        op1 = qiskit.optimization.OptimizationProblem()
        op1.variables.add(names=['x1', 'x2', 'x3'])
        f, fn = tempfile.mkstemp(suffix='.lp')
        os.close(f)
        op1.write(fn)
        op2 = qiskit.optimization.OptimizationProblem()
        op2.read(fn)
        self.assertEqual(op2.variables.get_num(), 3)

    def test_write3(self):
        """ test write 3 """
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(names=['x1', 'x2', 'x3'])

        class NoOpStream(object):
            def __init__(self):
                self.was_called = False

            def write(self, bytes):
                """ write """
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
        op = qiskit.optimization.OptimizationProblem()
        op.variables.add(names=['x1', 'x2', 'x3'])
        lp_str = op.write_as_string("lp")
        self.assertGreater(len(lp_str), 0)

    def test_problem_type1(self):
        """ test problem type 1 """
        op = qiskit.optimization.OptimizationProblem()
        op.read(self.resource_file)
        self.assertEqual(op.get_problem_type(),
                         qiskit.optimization.problems.problem_type.CPXPROB_QP)
        self.assertEqual(op.problem_type[op.get_problem_type()], 'QP')

    def test_problem_type2(self):
        """ test problemm type 2"""
        op = qiskit.optimization.OptimizationProblem()
        op.set_problem_type(op.problem_type.LP)
        self.assertEqual(op.get_problem_type(),
                         qiskit.optimization.problems.problem_type.CPXPROB_LP)
        self.assertEqual(op.problem_type[op.get_problem_type()], 'LP')

    def test_problem_name(self):
        """ test problem name """
        op = qiskit.optimization.OptimizationProblem()
        op.set_problem_name("test")
        # test
        self.assertEqual(op.get_problem_name(), "test")


if __name__ == '__main__':
    unittest.main()
