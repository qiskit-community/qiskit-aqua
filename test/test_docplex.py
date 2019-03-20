# -*- coding: utf-8 -*-

# Copyright 2019 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

from docplex.mp.advmodel import AdvModel as Model
import numpy as np
import networkx as nx

from test.common import QiskitAquaTestCase

from qiskit.aqua.translators.ising import maxcut, tsp, docplex


class TestDocplex(QiskitAquaTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        np.random.seed(100)

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

        # Create an Ising Hamiltonian with an exsiting translator of Qiskit Aqua
        expected_qubitOp, expected_offset = maxcut.get_maxcut_qubitops(w)

        # Create an Ising Hamiltonian with docplex.
        mdl = Model(name='max_cut')
        mdl.node_vars = mdl.binary_var_list(list(range(4)), name='node')
        maxcut_func = mdl.sum(w[i, j] * mdl.node_vars[i] * (1 - mdl.node_vars[j]) for i in range(n) for j in range(n))
        mdl.maximize(maxcut_func)
        qubitOp, offset = docplex.get_docplex_qubitops(mdl)

        self.assertEqual(qubitOp, expected_qubitOp)
        self.assertEqual(offset, expected_offset)

    def test_docplex_tsp(self):
        # Generating a graph of 3 nodes
        n = 3
        ins = tsp.random_tsp(n)
        G = nx.Graph()
        G.add_nodes_from(np.arange(0, n, 1))
        num_node = ins.dim

        # Create an Ising Hamiltonian with an exsiting translator of Qiskit Aqua
        expected_qubitOp, expected_offset = tsp.get_tsp_qubitops(ins)

        # Create an Ising Hamiltonian with docplex.
        mdl = Model(name='tsp')
        x = {(i, p): mdl.binary_var(name='x_{0}_{1}'.format(i, p)) for i in range(num_node) for p in range(num_node)}
        tsp_func = mdl.sum(
            ins.w[i, j] * x[(i, p)] * x[(j, (p + 1) % num_node)] for i in range(num_node) for j in range(num_node) for p
            in
            range(num_node))
        mdl.minimize(tsp_func)
        for i in range(num_node):
            mdl.add_constraint(mdl.sum(x[(i, p)] for p in range(num_node)) == 1)
        for p in range(num_node):
            mdl.add_constraint(mdl.sum(x[(i, p)] for i in range(num_node)) == 1)
        qubitOp, offset = docplex.get_docplex_qubitops(mdl)

        self.assertEqual(qubitOp, expected_qubitOp)
        self.assertEqual(offset, expected_offset)
