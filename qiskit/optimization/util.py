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

""" Optimization Utilities module """

import datetime


def get_qubo_solutions(function_dict, n_key, print_solutions=False):
    """
    Calculates all of the outputs of a QUBO function representable by n key qubits.
    Args:
        function_dict (dict): A dictionary representation of the function, where the keys correspond
            to a variable, and the values are the corresponding coefficients.
        n_key (int): The number of key qubits.
        print_solutions (bool, optional): If true, the solutions will be formatted and printed.
    Returns:
        dict: A dictionary of the inputs (keys) and outputs (values) of the QUBO function.
    """
    # Determine constant.
    constant = 0
    if -1 in function_dict:
        constant = function_dict[-1]
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
            if bin_list[k] == 1 and k in function_dict:
                solution += function_dict[k]

        # Handle the quadratic terms.
        for j in range(len(bin_key)):
            for q in range(len(bin_key)):
                if (j, q) in function_dict and j != q and bin_list[j] == 1 and bin_list[q] == 1:
                    solution += function_dict[(j, q)]

        # Print row.
        if print_solutions:
            spacer = "" if i >= 10 else " "
            value_spacer = " " if solution < 0 else "  "
            print(spacer + str(i), "=", bin_key, "->" + value_spacer + str(round(solution, 4)))

        # Record solution.
        solutions[i] = solution

    if print_solutions:
        print()

    return solutions
