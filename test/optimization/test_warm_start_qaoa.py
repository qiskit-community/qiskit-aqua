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

""" Test warm start QAOA optimizer. """

import numpy as np
from test.optimization import QiskitOptimizationTestCase

from qiskit import BasicAer
from qiskit.aqua.algorithms import QAOA
from qiskit.optimization.algorithms.goemans_williamson_optimizer import GoemansWilliamsonOptimizer
from qiskit.optimization.algorithms.warm_start_qaoa_optimizer import WarmStartQAOAOptimizer, \
    MeanAggregator
from qiskit.optimization.applications.ising.max_cut import max_cut_qp


class TestWarmStartQAOAOptimizer(QiskitOptimizationTestCase):
    def test(self):
        graph = np.array([[0., 1., 2., 0.],
                          [1., 0., 1., 0.],
                          [2., 1., 0., 1.],
                          [0., 0., 1., 0.]])

        # G = nx.from_numpy_matrix(graph)
        # nx.draw_networkx(G)

        presolver = GoemansWilliamsonOptimizer(num_cuts=10)
        problem = max_cut_qp(graph)

        backend = BasicAer.get_backend('statevector_simulator')
        qaoa = QAOA(quantum_instance=backend, p=1)
        aggregator = MeanAggregator()
        optimizer = WarmStartQAOAOptimizer(pre_solver=presolver, qaoa=qaoa, epsilon=0.25,
                                           aggregator=aggregator)
        result_warm = optimizer.solve(problem)

        print(result_warm)
        print(result_warm.samples)

        # qaoa = WarmQAOA(quantum_instance=BasicAer.get_backend('statevector_simulator'), p=1,
        #                     epsilon=0.25)
        # aggregator = MeanAggregator()
        # qaoa = WarmStartMinimumEigenOptimizer(qaoa, presolver, aggregator=aggregator)
        # rqaoa = RecursiveMinimumEigenOptimizer(
        #     qaoa)  # , min_num_vars=3, min_num_vars_optimizer=qaoa)
        # result_warm = rqaoa.solve(problem)
        #
        # print(result_warm)
