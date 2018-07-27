# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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

import unittest

import numpy as np
from qiskit import QuantumRegister
from qiskit.tools.qi.qi import state_fidelity
from qiskit.wrapper import execute as q_execute

from test.common import QiskitAquaTestCase
from qiskit_aqua.operator import Operator
from qiskit_aqua import get_algorithm_instance, get_initial_state_instance


class TestEvolution(QiskitAquaTestCase):
    """Evolution tests."""

    def test_evolution(self):
        SIZE = 2
        #SPARSITY = 0
        #X = [[0, 1], [1, 0]]
        #Y = [[0, -1j], [1j, 0]]
        Z = [[1, 0], [0, -1]]
        I = [[1, 0], [0, 1]]
        h1 = np.kron(I, Z)  # + 0.5 * np.kron(Y, X)# + 0.3 * np.kron(Z, X) + 0.4 * np.kron(Z, Y)

        # np.random.seed(2)
        temp = np.random.random((2 ** SIZE, 2 ** SIZE))
        h1 = temp + temp.T
        qubitOp = Operator(matrix=h1)
        # qubitOp_jw.chop_by_threshold(10 ** -10)

        # self.log.debug('matrix:\n{}\n'.format(qubitOp.matrix))
        # self.log.debug('paulis:')
        # self.log.debug(qubitOp.print_operators('paulis'))

        if qubitOp.grouped_paulis is None:
            qubitOp._matrix_to_paulis()
            qubitOp._paulis_to_grouped_paulis()

        for ps in qubitOp.grouped_paulis:
            for p1 in ps:
                for p2 in ps:
                    if p1 != p2:
                        np.testing.assert_almost_equal(
                            p1[1].to_matrix() @ p2[1].to_matrix(),
                            p2[1].to_matrix() @ p1[1].to_matrix()
                        )

        flattened_grouped_paulis = [pauli for group in qubitOp.grouped_paulis for pauli in group[1:]]
        self.assertEqual(sorted([str(p) for p in flattened_grouped_paulis]), sorted([str(p) for p in qubitOp.paulis]))

        state_in = get_initial_state_instance('CUSTOM')
        state_in.init_args(SIZE, state='random')

        evo_time = 1
        num_time_slices = 1

        # announces params
        self.log.debug('evo time:        {}'.format(evo_time))
        self.log.debug('num time slices: {}'.format(num_time_slices))
        self.log.debug('state_in:        {}'.format(state_in._state_vector))

        # get the exact state_out from raw matrix multiplication
        state_out_exact = qubitOp.evolve(state_in.construct_circuit('vector'), evo_time, 'matrix', 0)
        # self.log.debug('exact:\n{}'.format(state_out_exact))

        for grouping in ['default', 'random']:
            self.log.debug('Under {} paulis grouping:'.format(grouping))
            for expansion_mode in ['trotter', 'suzuki']:
                self.log.debug('Under {} expansion mode:'.format(expansion_mode))
                for expansion_order in [1, 2, 3, 4] if expansion_mode == 'suzuki' else [1]:
                    if expansion_mode == 'suzuki':
                        self.log.debug('With expansion order {}:'.format(expansion_order))
                    state_out_matrix = qubitOp.evolve(
                        state_in.construct_circuit('vector'), evo_time, 'matrix', num_time_slices,
                        paulis_grouping=grouping,
                        expansion_mode=expansion_mode,
                        expansion_order=expansion_order
                    )

                    quantum_registers = QuantumRegister(qubitOp.num_qubits)
                    qc = state_in.construct_circuit('circuit', quantum_registers)
                    qc += qubitOp.evolve(
                        None, evo_time, 'circuit', num_time_slices,
                        quantum_registers=quantum_registers,
                        paulis_grouping=grouping,
                        expansion_mode=expansion_mode,
                        expansion_order=expansion_order,
                    )
                    job = q_execute(qc, 'local_statevector_simulator', skip_transpiler=True)
                    state_out_circuit = np.asarray(job.result().get_statevector(qc))

                    self.log.debug('The fidelity between exact and matrix:   {}'.format(
                        state_fidelity(state_out_exact, state_out_matrix)
                    ))
                    self.log.debug('The fidelity between exact and circuit:  {}'.format(
                        state_fidelity(state_out_exact, state_out_circuit)
                    ))
                    f_mc = state_fidelity(state_out_matrix, state_out_circuit)
                    self.log.debug('The fidelity between matrix and circuit: {}'.format(f_mc))
                    self.assertAlmostEqual(f_mc, 1)


if __name__ == '__main__':
    unittest.main()
