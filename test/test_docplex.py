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

from math import fsum, isclose

import networkx as nx
import numpy as np
from docplex.mp.model import Model
from qiskit.quantum_info import Pauli

from qiskit.aqua import Operator, AquaError
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.translators.ising import tsp, docplex
from test.common import QiskitAquaTestCase

# Reference operators and offsets for maxcut and tsp.
qubitOp_maxcut = Operator(paulis=[[0.5, Pauli(z=[True, True, False, False], x=[False, False, False, False])],
                                  [0.5, Pauli(z=[True, False, True, False], x=[False, False, False, False])],
                                  [0.5, Pauli(z=[False, True, True, False], x=[False, False, False, False])],
                                  [0.5, Pauli(z=[True, False, False, True], x=[False, False, False, False])],
                                  [0.5, Pauli(z=[False, False, True, True], x=[False, False, False, False])]])
offset_maxcut = -2.5
qubitOp_tsp = Operator(paulis=[[-100057.0, Pauli(z=[True, False, False, False, False, False, False, False, False],
                                                 x=[False, False, False, False, False, False, False, False, False])],
                               [-100071.0, Pauli(z=[False, False, False, False, True, False, False, False, False],
                                                 x=[False, False, False, False, False, False, False, False, False])],
                               [14.5, Pauli(z=[True, False, False, False, True, False, False, False, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [-100057.0, Pauli(z=[False, True, False, False, False, False, False, False, False],
                                                 x=[False, False, False, False, False, False, False, False, False])],
                               [-100071.0, Pauli(z=[False, False, False, False, False, True, False, False, False],
                                                 x=[False, False, False, False, False, False, False, False, False])],
                               [14.5, Pauli(z=[False, True, False, False, False, True, False, False, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [-100057.0, Pauli(z=[False, False, True, False, False, False, False, False, False],
                                                 x=[False, False, False, False, False, False, False, False, False])],
                               [-100071.0, Pauli(z=[False, False, False, True, False, False, False, False, False],
                                                 x=[False, False, False, False, False, False, False, False, False])],
                               [14.5, Pauli(z=[False, False, True, True, False, False, False, False, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [-100070.0, Pauli(z=[False, False, False, False, False, False, False, True, False],
                                                 x=[False, False, False, False, False, False, False, False, False])],
                               [14.0, Pauli(z=[True, False, False, False, False, False, False, True, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [-100070.0, Pauli(z=[False, False, False, False, False, False, False, False, True],
                                                 x=[False, False, False, False, False, False, False, False, False])],
                               [14.0, Pauli(z=[False, True, False, False, False, False, False, False, True],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [-100070.0, Pauli(z=[False, False, False, False, False, False, True, False, False],
                                                 x=[False, False, False, False, False, False, False, False, False])],
                               [14.0, Pauli(z=[False, False, True, False, False, False, True, False, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [14.5, Pauli(z=[False, True, False, True, False, False, False, False, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [14.5, Pauli(z=[False, False, True, False, True, False, False, False, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [14.5, Pauli(z=[True, False, False, False, False, True, False, False, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [21.0, Pauli(z=[False, False, False, True, False, False, False, True, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [21.0, Pauli(z=[False, False, False, False, True, False, False, False, True],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [21.0, Pauli(z=[False, False, False, False, False, True, True, False, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [14.0, Pauli(z=[False, True, False, False, False, False, True, False, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [14.0, Pauli(z=[False, False, True, False, False, False, False, True, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [14.0, Pauli(z=[True, False, False, False, False, False, False, False, True],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [21.0, Pauli(z=[False, False, False, False, True, False, True, False, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [21.0, Pauli(z=[False, False, False, False, False, True, False, True, False],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [21.0, Pauli(z=[False, False, False, True, False, False, False, False, True],
                                            x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[True, False, False, True, False, False, False, False, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[True, False, False, False, False, False, True, False, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, False, False, True, False, False, True, False, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, True, False, False, True, False, False, False, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, True, False, False, False, False, False, True, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, False, False, False, True, False, False, True, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, False, True, False, False, True, False, False, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, False, True, False, False, False, False, False, True],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, False, False, False, False, True, False, False, True],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[True, True, False, False, False, False, False, False, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[True, False, True, False, False, False, False, False, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, True, True, False, False, False, False, False, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, False, False, True, True, False, False, False, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, False, False, True, False, True, False, False, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, False, False, False, True, True, False, False, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, False, False, False, False, False, True, True, False],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, False, False, False, False, False, True, False, True],
                                               x=[False, False, False, False, False, False, False, False, False])],
                               [50000.0, Pauli(z=[False, False, False, False, False, False, False, True, True],
                                               x=[False, False, False, False, False, False, False, False, False])]])
offset_tsp = 600297.0


class TestDocplex(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        np.random.seed(100)

    def test_validation(self):
        num_var = 3
        # validate an object type of the input.
        with self.assertRaises(AquaError):
            docplex._validate_input_model("Model")

        # validate the types of the variables are binary or not
        with self.assertRaises(AquaError):
            mdl = Model(name='Error_integer_variables')
            x = {i: mdl.integer_var(name='x_{0}'.format(i)) for i in range(num_var)}
            obj_func = mdl.sum(x[i] for i in range(num_var))
            mdl.maximize(obj_func)
            docplex.get_qubitops(mdl)

        # validate types of constraints are equality constraints or not.
        with self.assertRaises(AquaError):
            mdl = Model(name='Error_inequality_constraints')
            x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(num_var)}
            obj_func = mdl.sum(x[i] for i in range(num_var))
            mdl.maximize(obj_func)
            mdl.add_constraint(mdl.sum(x[i] for i in range(num_var)) <= 1)
            docplex.get_qubitops(mdl)

    def test_auto_define_penalty(self):
        # check _auto_define_penalty() for positive coefficients.
        positive_coefficients = np.random.rand(10, 10)
        for i in range(10):
            mdl = Model(name='Positive_auto_define_penalty')
            x = {j: mdl.binary_var(name='x_{0}'.format(j)) for j in range(10)}
            obj_func = mdl.sum(positive_coefficients[i][j] * x[j] for j in range(10))
            mdl.maximize(obj_func)
            actual = docplex._auto_define_penalty(mdl)
            expected = fsum(abs(j) for j in positive_coefficients[i]) + 1
            self.assertEqual(isclose(actual, expected), True)

        # check _auto_define_penalty() for negative coefficients
        negative_coefficients = -1 * np.random.rand(10, 10)
        for i in range(10):
            mdl = Model(name='Negative_auto_define_penalty')
            x = {j: mdl.binary_var(name='x_{0}'.format(j)) for j in range(10)}
            obj_func = mdl.sum(negative_coefficients[i][j] * x[j] for j in range(10))
            mdl.maximize(obj_func)
            actual = docplex._auto_define_penalty(mdl)
            expected = fsum(abs(j) for j in negative_coefficients[i]) + 1
            self.assertEqual(isclose(actual, expected), True)

        # check _auto_define_penalty() for mixed coefficients
        mixed_coefficients = np.random.randint(-100, 100, (10, 10))
        for i in range(10):
            mdl = Model(name='Mixed_auto_define_penalty')
            x = {j: mdl.binary_var(name='x_{0}'.format(j)) for j in range(10)}
            obj_func = mdl.sum(mixed_coefficients[i][j] * x[j] for j in range(10))
            mdl.maximize(obj_func)
            actual = docplex._auto_define_penalty(mdl)
            expected = fsum(abs(j) for j in mixed_coefficients[i]) + 1
            self.assertEqual(isclose(actual, expected), True)

        # check that 1e5 is being used when coefficients have float numbers.
        float_coefficients = [0.1 * i for i in range(3)]
        mdl = Model(name='Float_auto_define_penalty')
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(3)}
        obj_func = mdl.sum(x[i] for i in range(3))
        mdl.maximize(obj_func)
        mdl.add_constraint(mdl.sum(float_coefficients[i] * x[i] for i in range(3)) == 1)
        actual = docplex._auto_define_penalty(mdl)
        expected = 1e5
        self.assertEqual(actual, expected)

    def test_docplex_maxcut(self):
        # Generating a graph of 4 nodes
        n = 4
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, n, 1))
        elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        G.add_weighted_edges_from(elist)
        # Computing the weight matrix from the random graph
        w = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                temp = G.get_edge_data(i, j, default=0)
                if temp != 0:
                    w[i, j] = temp['weight']

        # Create an Ising Hamiltonian with docplex.
        mdl = Model(name='max_cut')
        mdl.node_vars = mdl.binary_var_list(list(range(4)), name='node')
        maxcut_func = mdl.sum(w[i, j] * mdl.node_vars[i] * (1 - mdl.node_vars[j]) for i in range(n) for j in range(n))
        mdl.maximize(maxcut_func)
        qubitOp, offset = docplex.get_qubitops(mdl)

        ee = ExactEigensolver(qubitOp, k=1)
        result = ee.run()

        ee_expected = ExactEigensolver(qubitOp_maxcut, k=1)
        expected_result = ee_expected.run()

        # Compare objective
        self.assertEqual(result['energy'] + offset, expected_result['energy'] + offset_maxcut)

    def test_docplex_tsp(self):
        # Generating a graph of 3 nodes
        n = 3
        ins = tsp.random_tsp(n)
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, n, 1))
        num_node = ins.dim

        # Create an Ising Hamiltonian with docplex.
        mdl = Model(name='tsp')
        x = {(i, p): mdl.binary_var(name='x_{0}_{1}'.format(i, p)) for i in range(num_node) for p in range(num_node)}
        tsp_func = mdl.sum(
            ins.w[i, j] * x[(i, p)] * x[(j, (p + 1) % num_node)] for i in range(num_node) for j in range(num_node) for p
            in range(num_node))
        mdl.minimize(tsp_func)
        for i in range(num_node):
            mdl.add_constraint(mdl.sum(x[(i, p)] for p in range(num_node)) == 1)
        for p in range(num_node):
            mdl.add_constraint(mdl.sum(x[(i, p)] for i in range(num_node)) == 1)
        qubitOp, offset = docplex.get_qubitops(mdl)

        ee = ExactEigensolver(qubitOp, k=1)
        result = ee.run()

        ee_expected = ExactEigensolver(qubitOp_tsp, k=1)
        expected_result = ee_expected.run()

        # Compare objective
        self.assertEqual(result['energy'] + offset, expected_result['energy'] + offset_tsp)

    def test_docplex_integer_constraints(self):
        # Create an Ising Homiltonian with docplex
        mdl = Model(name='integer_constraints')
        x = {i: mdl.binary_var(name='x_{0}'.format(i)) for i in range(1, 5)}
        max_vars_func = mdl.sum(x[i] for i in range(1, 5))
        mdl.maximize(max_vars_func)
        mdl.add_constraint(mdl.sum(i * x[i] for i in range(1, 5)) == 3)
        qubitOp, offset = docplex.get_qubitops(mdl)

        ee = ExactEigensolver(qubitOp, k=1)
        result = ee.run()

        expected_result = -2

        # Compare objective
        self.assertEqual(result['energy'] + offset, expected_result)
