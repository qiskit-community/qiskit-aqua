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

"""
Code inside the test is the optimization sample from the readme.
If this test fails and code changes are needed here to resolve
the issue then ensure changes are made to readme too.
"""

import unittest
from test.optimization import QiskitOptimizationTestCase


class TestReadmeSample(QiskitOptimizationTestCase):
    """Test sample code from readme"""

    def test_readme_sample(self):
        """ readme sample test """
        # pylint: disable=import-outside-toplevel

        # --- Exact copy of sample code ----------------------------------------

        import networkx as nx
        import numpy as np
        from docplex.mp.model import Model

        from qiskit import BasicAer
        from qiskit.aqua import aqua_globals, QuantumInstance
        from qiskit.aqua.algorithms import QAOA
        from qiskit.aqua.components.optimizers import SPSA
        from qiskit.optimization.ising import docplex, max_cut
        from qiskit.optimization.ising.common import sample_most_likely

        # Generate a graph of 4 nodes
        n = 4
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(0, n, 1))
        elist = [(0, 1, 1.0), (0, 2, 1.0), (0, 3, 1.0), (1, 2, 1.0), (2, 3, 1.0)]
        graph.add_weighted_edges_from(elist)
        # Compute the weight matrix from the graph
        w = np.zeros([n, n])
        for i in range(n):
            for j in range(n):
                temp = graph.get_edge_data(i, j, default=0)
                if temp != 0:
                    w[i, j] = temp['weight']

        # Create an Ising Hamiltonian with docplex.
        mdl = Model(name='max_cut')
        mdl.node_vars = mdl.binary_var_list(list(range(n)), name='node')
        maxcut_func = mdl.sum(w[i, j] * mdl.node_vars[i] * (1 - mdl.node_vars[j])
                              for i in range(n) for j in range(n))
        mdl.maximize(maxcut_func)
        qubit_op, offset = docplex.get_operator(mdl)

        # Run quantum algorithm QAOA on qasm simulator
        seed = 40598
        aqua_globals.random_seed = seed

        spsa = SPSA(max_trials=250)
        qaoa = QAOA(qubit_op, spsa, p=5)
        backend = BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=1024, seed_simulator=seed,
                                           seed_transpiler=seed)
        result = qaoa.run(quantum_instance)

        x = sample_most_likely(result.eigenstate)
        self.log.debug('energy: %s', result.eigenvalue.real)
        self.log.debug('time: %s', result.optimizer_time)
        self.log.debug('max-cut objective: %s', result.eigenvalue.real + offset)
        self.log.debug('solution: %s', max_cut.get_graph_solution(x))
        self.log.debug('solution objective: %s', max_cut.max_cut_value(x, w))

        # ----------------------------------------------------------------------

        self.assertListEqual(max_cut.get_graph_solution(x).tolist(), [1, 0, 1, 0])
        self.assertAlmostEqual(max_cut.max_cut_value(x, w), 4.0)


if __name__ == '__main__':
    unittest.main()
