# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import math

import numpy as np
from qiskit import BasicAer
from qiskit.quantum_info import Pauli

from qiskit.aqua import run_algorithm
from qiskit.aqua.input import EnergyInput
from qiskit.aqua.translators.ising.portfolio_diversification import *
from test.common import QiskitAquaTestCase


class ClassicalOptimizer:
    def __init__(self, rho, n, q):

        self.rho = rho
        self.n = n  # number of inner variables
        self.q = q  # number of required selection

    def compute_allowed_combinations(self):
        f = math.factorial
        return int(f(self.n) / f(self.q) / f(self.n - self.q))

    def cplex_solution(self):

        # refactoring
        rho = self.rho
        n = self.n
        q = self.q

        my_obj = list(rho.reshape(1, n ** 2)[0]) + [0. for x in range(0, n)]
        my_ub = [1 for x in range(0, n ** 2 + n)]
        my_lb = [0 for x in range(0, n ** 2 + n)]
        my_ctype = "".join(['I' for x in range(0, n ** 2 + n)])

        my_rhs = [q] + [1 for x in range (0, n)] +[0 for x in range (0, n)] + [0.1 for x in range(0, n ** 2)]
        my_sense = "".join(['E' for x in range(0, 1+n)]) + "".join(['E' for x in range(0, n)]) + "".join(
            ['L' for x in range(0, n ** 2)])

        try:
            my_prob = cplex.Cplex()
            self.populate_by_row(my_prob, my_obj, my_ub, my_lb, my_ctype, my_sense, my_rhs)

            my_prob.solve()

        except CplexError as exc:
            print(exc)
            return

        x = my_prob.solution.get_values()
        x = np.array(x)
        cost = my_prob.solution.get_objective_value()

        return x, cost

    def populate_by_row(self, prob, my_obj, my_ub, my_lb, my_ctype, my_sense, my_rhs):

        n = self.n

        prob.objective.set_sense(prob.objective.sense.minimize)
        prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype)

        prob.set_log_stream(None)
        prob.set_error_stream(None)
        prob.set_warning_stream(None)
        prob.set_results_stream(None)

        rows = []
        col = [x for x in range(n**2, n**2+n)]
        coef = [1 for x in range(0, n)]
        rows.append([col, coef])

        for ii in range(0, n):
            col = [x for x in range(0+n*ii, n+n*ii)]
            coef = [1 for x in range(0, n)]

            rows.append([col, coef])

        for ii in range(0, n):
            col = [ii * n + ii, n ** 2 + ii]
            coef = [1, -1]
            rows.append([col, coef])

        for ii in range(0, n):
            for jj in range(0, n):
                col = [ii*n + jj, n ** 2 + jj]
                coef = [1, -1]

                rows.append([col, coef])
        
        prob.linear_constraints.add(lin_expr=rows, senses=my_sense, rhs=my_rhs)


# To run only this test, issue:
# python -m unittest test.test_portfoliodiversification.TestPortfolioDiversification

class TestPortfolioDiversification(QiskitAquaTestCase):
    """Tests vehicle routing Ising translator."""

    def setUp(self):
        super().setUp()
        np.random.seed(100)        
        self.n = 2
        self.q = 1
        self.instance = np.ones((self.n,self.n))
        self.instance[0,1] = 0.8
        self.instance[1,0] = 0.8
        # self.instance = -1 * self.instance
        self.qubit_op = get_portfoliodiversification_qubitops(self.instance, self.n, self.q)
        self.algo_input = EnergyInput(self.qubit_op)

    def test_simple1(self):
        # Compares the output in terms of Paulis.
        paulis = [
            (-249.5, Pauli(
                z=[True, False, False, False, False, False],
                x=[False, False, False, False, False, False]
            )),
            (-249.60000000000002, Pauli(
                z=[False, True, False, False, False, False],
                x=[False, False, False, False, False, False]
            )),
            (-249.60000000000002, Pauli(
                z=[False, False, True, False, False, False],
                x=[False, False, False, False, False, False]
            )),
            (-249.5, Pauli(
                z=[False, False, False, True, False, False],
                x=[False, False, False, False, False, False]
            )),
            (500.0, Pauli(
                z=[False, False, False, False, True, False],
                x=[False, False, False, False, False, False]
            )),
            (500.0, Pauli(
                z=[False, False, False, False, False, True],
                x=[False, False, False, False, False, False]
            )),
            (500.0, Pauli(
                z=[True, True, False, False, False, False],
                x=[False, False, False, False, False, False]
            )),
            (500.0, Pauli(
                z=[False, False, True, True, False, False],
                x=[False, False, False, False, False, False]
            )),
            (-750.0, Pauli(
                z=[True, False, False, False, True, False],
                x=[False, False, False, False, False, False]
            )),
            (-250.0, Pauli(
                z=[False, False, True, False, True, False],
                x=[False, False, False, False, False, False]
            )),
            (-250.0, Pauli(
                z=[False, True, False, False, False, True],
                x=[False, False, False, False, False, False]
            )),
            (-750.0, Pauli(
                z=[False, False, False, True, False, True],
                x=[False, False, False, False, False, False]
            )),
            (500.0, Pauli(
                z=[False, False, False, False, True, True],
                x=[False, False, False, False, False, False]
            )),
            (3498.2, Pauli(
                z=[False, False, False, False, False, False],
                x=[False, False, False, False, False, False]
            ))
        ]
        for pauliA, pauliB in zip(self.qubit_op._paulis, paulis):
            costA, binaryA = pauliA
            costB, binaryB = pauliB
            # Note that the construction is a bit iffy, e.g., I can get:
            # Items are not equal to 7 significant digits:
            # ACTUAL: -250.5
            # DESIRED: -249.5
            # even when the ordering is the same. Obviously, when the ordering changes, the test will become invalid.
            np.testing.assert_approx_equal(costA, costB, 2)
            self.assertEqual(binaryA, binaryB)

    def test_simple2(self):
        # Computes the cost using the exact eigensolver and compares it against pre-determined value.
        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)        
        quantum_solution = get_portfoliodiversification_solution(self.instance, self.n, self.q, result)
        ground_level = get_portfoliodiversification_value(self.instance, self.n, self.q, quantum_solution)
        np.testing.assert_approx_equal(ground_level, 1.8)
    
    def test_portfolio_diversification(self):
        # Something of an integration test
        # Solve the problem in a classical fashion via CPLEX and compare the solution
        # Note that CPLEX uses a completely different integer linear programming formulation.
        x = None
        try:
            classical_optimizer = ClassicalOptimizer(self.instance, self.n, self.q)
            x, classical_cost = classical_optimizer.cplex_solution()
        except: 
            # This test should not focus on the availability of CPLEX, so we just eat the exception.
            self.skipTest("CPLEX may be missing.")
        # Solve the problem using the exact eigensolver
        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, self.algo_input)        
        quantum_solution = get_portfoliodiversification_solution(self.instance, self.n, self.q, result)
        ground_level = get_portfoliodiversification_value(self.instance, self.n, self.q, quantum_solution)
        if x:
            np.testing.assert_approx_equal(ground_level, classical_cost)
            np.testing.assert_array_almost_equal(quantum_solution, x, 5)
