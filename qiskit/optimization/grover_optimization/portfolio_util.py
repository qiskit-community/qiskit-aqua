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

from qiskit.finance.data_providers import RandomDataProvider
import datetime


def get_mu_sigma(num_assets):
    # Generate expected return and covariance matrix from (random) time-series
    stocks = [("TICKER%s" % i) for i in range(num_assets)]
    data = RandomDataProvider(tickers=stocks, start=datetime.datetime(2020, 1, 1), end=datetime.datetime(2020, 1, 30))
    data.run()
    mu = data.get_period_return_mean_vector()
    sigma = data.get_period_return_covariance_matrix()

    return mu, sigma


def get_qubo_solutions(f, n_key, print_solutions=False):
    # Determine constant.
    constant = 0
    if -1 in f:
        constant = f[-1]
    format_string = '{0:0'+str(n_key)+'b}'

    # Iterate through every key combination.
    if print_solutions:
        print("QUBO Solutions:")
        print("==========================")
    solutions = {}
    for i in range(2**n_key):
        solution = constant

        # Convert int to a list of binary variables.
        bin_key = format_string.format(i)
        bin_list = [int(bin_key[j]) for j in range(len(bin_key))]

        # Handle the linear terms.
        for k in range(len(bin_key)):
            if bin_list[k] == 1 and k in f:
                solution += f[k]

        # Handle the quadratic terms.
        for p in range(len(bin_key)):
            for q in range(len(bin_key)):
                if (p, q) in f and p != q and bin_list[p] == 1 and bin_list[q] == 1:
                    solution += f[(p, q)]

        # Print row.
        if print_solutions:
            spacer = "" if i >= 10 else " "
            value_spacer = " " if solution < 0 else "  "
            print(spacer + str(i), "=", bin_key, "->" + value_spacer + str(round(solution, 4)))

        # Record solution.
        solutions[i] = str(round(solution, 4))

    if print_solutions:
        print()

    return solutions
