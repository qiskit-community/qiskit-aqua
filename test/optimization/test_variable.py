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

"""Test Variable."""

import unittest
from test.optimization.optimization_test_case import QiskitOptimizationTestCase
import logging

from qiskit.optimization.problems import QuadraticProgram, Variable, VarType
from qiskit.optimization import infinity, QiskitOptimizationError

logger = logging.getLogger(__name__)


class TestVariable(QiskitOptimizationTestCase):
    """Test Variable."""

    def test_init(self):
        """ test init """

        quadratic_program = QuadraticProgram()
        name = 'variable'
        lowerbound = 0
        upperbound = 10
        vartype = VarType.integer

        variable = Variable(quadratic_program, name, lowerbound, upperbound, vartype)

        self.assertEqual(variable.name, name)
        self.assertEqual(variable.lowerbound, lowerbound)
        self.assertEqual(variable.upperbound, upperbound)
        self.assertEqual(variable.vartype, VarType.integer)

    def test_init_default(self):
        """ test init with default values."""

        quadratic_program = QuadraticProgram()
        name = 'variable'

        variable = Variable(quadratic_program, name)

        self.assertEqual(variable.name, name)
        self.assertEqual(variable.lowerbound, 0)
        self.assertEqual(variable.upperbound, infinity)
        self.assertEqual(variable.vartype, VarType.continuous)

    def test_setters(self):
        """ test setters. """

        quadratic_program = QuadraticProgram()
        name = 'variable'
        lowerbound = 0
        upperbound = 10
        vartype = VarType.continuous

        variable = Variable(quadratic_program, name, lowerbound, upperbound, vartype)

        variable.name = 'test'
        self.assertEqual(variable.name, 'test')

        self.assertEqual(variable.lowerbound, lowerbound)
        variable.lowerbound = 1
        self.assertEqual(variable.lowerbound, 1)
        with self.assertRaises(QiskitOptimizationError):
            variable.lowerbound = 20

        self.assertEqual(variable.upperbound, upperbound)
        variable.upperbound = 5
        self.assertEqual(variable.upperbound, 5)
        with self.assertRaises(QiskitOptimizationError):
            variable.upperbound = 0

        self.assertEqual(variable.vartype, vartype)
        variable.vartype = VarType.integer
        self.assertEqual(variable.vartype, VarType.integer)
        variable.vartype = VarType.binary
        self.assertEqual(variable.vartype, VarType.binary)
        variable.vartype = VarType.continuous
        self.assertEqual(variable.vartype, VarType.continuous)


if __name__ == '__main__':
    unittest.main()
