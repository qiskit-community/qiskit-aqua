# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Partition """

from test.optimization.common import QiskitOptimizationTestCase
import warnings
import numpy as np
from qiskit import BasicAer
from qiskit.aqua import run_algorithm, aqua_globals, QuantumInstance
from qiskit.aqua.input import EnergyInput
from qiskit.optimization.ising import partition
from qiskit.optimization.ising.common import read_numbers_from_file, sample_most_likely
from qiskit.aqua.algorithms import ExactEigensolver, VQE
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.components.variational_forms import RY


class TestSetPacking(QiskitOptimizationTestCase):
    """Cplex Ising tests."""

    def setUp(self):
        super().setUp()
        warnings.filterwarnings("ignore", message=aqua_globals.CONFIG_DEPRECATION_MSG,
                                category=DeprecationWarning)
        input_file = self._get_resource_path('sample.partition')
        number_list = read_numbers_from_file(input_file)
        self.qubit_op, _ = partition.get_operator(number_list)

    def test_partition(self):
        """ Partition test """
        params = {
            'problem': {'name': 'ising'},
            'algorithm': {'name': 'ExactEigensolver'}
        }
        result = run_algorithm(params, EnergyInput(self.qubit_op))

        x = sample_most_likely(result['eigvecs'][0])
        np.testing.assert_array_equal(x, [0, 1, 0])

    def test_partition_direct(self):
        """ Partition Direct test """
        algo = ExactEigensolver(self.qubit_op, k=1, aux_operators=[])
        result = algo.run()
        x = sample_most_likely(result['eigvecs'][0])
        np.testing.assert_array_equal(x, [0, 1, 0])

    def test_partition_vqe(self):
        """ Partition VQE test """
        aqua_globals.random_seed = 100
        result = VQE(self.qubit_op,
                     RY(self.qubit_op.num_qubits, depth=5, entanglement='linear'),
                     SPSA(max_trials=200),
                     max_evals_grouped=2).run(
                         QuantumInstance(BasicAer.get_backend('qasm_simulator'),
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        x = sample_most_likely(result['eigvecs'][0])
        self.assertNotEqual(x[0], x[1])
        self.assertNotEqual(x[2], x[1])  # hardcoded oracle
