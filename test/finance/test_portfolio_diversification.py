# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Portfolio Diversification Optimization """

import unittest
import math
from test.finance import QiskitFinanceTestCase
import warnings
import logging
import numpy as np

from qiskit.quantum_info import Pauli

from qiskit.aqua import aqua_globals
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.finance.applications.ising.portfolio_diversification import \
    (get_portfoliodiversification_solution,
     get_operator,
     get_portfoliodiversification_value)

logger = logging.getLogger(__name__)

try:
    import cplex
    _HAS_CPLEX = True
except ImportError:
    _HAS_CPLEX = False


class ClassicalOptimizer:
    """ Classical Optimizer """
    def __init__(self, rho, n, q):

        self.rho = rho
        self.n = n  # number of inner variables
        self.q = q  # number of required selection

    def _compute_allowed_combinations(self):
        fac = math.factorial
        return int(fac(self.n) / fac(self.q) / fac(self.n - self.q))

    def cplex_solution(self):
        """ cplex solution """

        # refactoring
        rho = self.rho
        n = self.n
        q = self.q

        my_obj = list(rho.reshape(1, n ** 2)[0]) + [0. for x in range(0, n)]
        my_ub = [1 for x in range(0, n ** 2 + n)]
        my_lb = [0 for x in range(0, n ** 2 + n)]
        my_ctype = "".join(['I' for x in range(0, n ** 2 + n)])

        my_rhs = [q] + [1 for x in range(0, n)] + \
                 [0 for x in range(0, n)] + [0.1 for x in range(0, n ** 2)]
        my_sense = "".join(['E' for x in range(0, 1 + n)]) + \
                   "".join(['E' for x in range(0, n)]) + \
                   "".join(['L' for x in range(0, n ** 2)])

        try:
            my_prob = cplex.Cplex()
            self._populate_by_row(my_prob, my_obj, my_ub, my_lb, my_ctype, my_sense, my_rhs)

            my_prob.solve()

        except Exception as ex:  # pylint: disable=broad-except
            print(str(ex))
            return None, None

        x = my_prob.solution.get_values()
        x = np.array(x)
        cost = my_prob.solution.get_objective_value()

        return x, cost

    def _populate_by_row(self, prob, my_obj, my_ub, my_lb, my_ctype, my_sense, my_rhs):

        n = self.n

        prob.objective.set_sense(prob.objective.sense.minimize)
        prob.variables.add(obj=my_obj, lb=my_lb, ub=my_ub, types=my_ctype)

        prob.set_log_stream(None)
        prob.set_error_stream(None)
        prob.set_warning_stream(None)
        prob.set_results_stream(None)

        rows = []
        col = list(range(n ** 2, n ** 2 + n))
        coef = [1 for x in range(0, n)]
        rows.append([col, coef])

        for i_i in range(0, n):
            col = list(range(0 + n * i_i, n + n * i_i))
            coef = [1 for x in range(0, n)]

            rows.append([col, coef])

        for i_i in range(0, n):
            col = [i_i * n + i_i, n ** 2 + i_i]
            coef = [1, -1]
            rows.append([col, coef])

        for i_i in range(0, n):
            for j_j in range(0, n):
                col = [i_i * n + j_j, n ** 2 + j_j]
                coef = [1, -1]

                rows.append([col, coef])

        prob.linear_constraints.add(lin_expr=rows, senses=my_sense, rhs=my_rhs)


# To run only this test, issue:
# python -m unittest test.finance_portfoliodiversification.TestPortfolioDiversification

class TestPortfolioDiversification(QiskitFinanceTestCase):
    """Tests Portfolio Diversification Ising translator."""

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 100
        self.n = 2
        self.q = 1
        self.instance = np.ones((self.n, self.n))
        self.instance[0, 1] = 0.8
        self.instance[1, 0] = 0.8
        # self.instance = -1 * self.instance
        self.qubit_op = get_operator(self.instance, self.n, self.q)
        warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

    def tearDown(self):
        super().tearDown()
        warnings.filterwarnings(action="always", message="unclosed", category=ResourceWarning)

    def test_simple1(self):
        """ simple1 test """
        # Compares the output in terms of Paulis.
        paulis = [
            (-249.5, Pauli(
                ([True, False, False, False, False, False],
                 [False, False, False, False, False, False])
            )),
            (-249.60000000000002, Pauli(
                ([False, True, False, False, False, False],
                 [False, False, False, False, False, False])
            )),
            (-249.60000000000002, Pauli(
                ([False, False, True, False, False, False],
                 [False, False, False, False, False, False])
            )),
            (-249.5, Pauli(
                ([False, False, False, True, False, False],
                 [False, False, False, False, False, False])
            )),
            (500.0, Pauli(
                ([False, False, False, False, True, False],
                 [False, False, False, False, False, False])
            )),
            (500.0, Pauli(
                ([False, False, False, False, False, True],
                 [False, False, False, False, False, False])
            )),
            (500.0, Pauli(
                ([True, True, False, False, False, False],
                 [False, False, False, False, False, False])
            )),
            (500.0, Pauli(
                ([False, False, True, True, False, False],
                 [False, False, False, False, False, False])
            )),
            (-750.0, Pauli(
                ([True, False, False, False, True, False],
                 [False, False, False, False, False, False])
            )),
            (-250.0, Pauli(
                ([False, False, True, False, True, False],
                 [False, False, False, False, False, False])
            )),
            (-250.0, Pauli(
                ([False, True, False, False, False, True],
                 [False, False, False, False, False, False])
            )),
            (-750.0, Pauli(
                ([False, False, False, True, False, True],
                 [False, False, False, False, False, False])
            )),
            (500.0, Pauli(
                ([False, False, False, False, True, True],
                 [False, False, False, False, False, False])
            )),
            (3498.2, Pauli(
                ([False, False, False, False, False, False],
                 [False, False, False, False, False, False])
            ))
        ]
        for pauli_a, pauli_b in zip(self.qubit_op._paulis, paulis):
            cost_a, binary_a = pauli_a
            cost_b, binary_b = pauli_b
            # Note that the construction is a bit iffy, e.g., I can get:
            # Items are not equal to 7 significant digits:
            # ACTUAL: -250.5
            # DESIRED: -249.5
            # even when the ordering is the same. Obviously, when the ordering changes,
            # the test will become invalid.
            np.testing.assert_approx_equal(np.real(cost_a), cost_b, 2)
            self.assertEqual(binary_a, binary_b)

    def test_simple2(self):
        """ simple2 test """
        # Computes the cost using the exact eigensolver
        # and compares it against pre-determined value.
        result = NumPyMinimumEigensolver(self.qubit_op).run()
        quantum_solution = get_portfoliodiversification_solution(self.instance,
                                                                 self.n,
                                                                 self.q, result)
        ground_level = get_portfoliodiversification_value(self.instance,
                                                          self.n, self.q,
                                                          quantum_solution)
        np.testing.assert_approx_equal(ground_level, 1.8)

    def test_portfolio_diversification(self):
        """ portfolio diversification test """
        # Something of an integration test
        # Solve the problem in a classical fashion via CPLEX and compare the solution
        # Note that CPLEX uses a completely different integer linear programming formulation.
        if not _HAS_CPLEX:
            self.skipTest('CPLEX is not installed.')
        x = None
        classical_optimizer = ClassicalOptimizer(self.instance, self.n, self.q)
        x, classical_cost = classical_optimizer.cplex_solution()

        # Solve the problem using the exact eigensolver
        result = NumPyMinimumEigensolver(self.qubit_op).run()
        quantum_solution = get_portfoliodiversification_solution(self.instance,
                                                                 self.n,
                                                                 self.q, result)
        ground_level = get_portfoliodiversification_value(self.instance,
                                                          self.n,
                                                          self.q, quantum_solution)
        if x is not None:
            np.testing.assert_approx_equal(ground_level, classical_cost)
            np.testing.assert_array_almost_equal(quantum_solution, x, 5)


if __name__ == '__main__':
    unittest.main()
