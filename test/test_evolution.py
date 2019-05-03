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

import unittest
import copy
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit
from qiskit import BasicAer
from qiskit import execute as q_execute
from qiskit.quantum_info import state_fidelity

from test.common import QiskitAquaTestCase
from qiskit.aqua import Operator
from qiskit.aqua.components.initial_states import Custom


class TestEvolution(QiskitAquaTestCase):
    """Evolution tests."""

    def test_evolution(self):
        SIZE = 2
        #SPARSITY = 0
        #X = [[0, 1], [1, 0]]
        #Y = [[0, -1j], [1j, 0]]
        Z = [[1, 0], [0, -1]]
        I = [[1, 0], [0, 1]]
        # + 0.5 * np.kron(Y, X)# + 0.3 * np.kron(Z, X) + 0.4 * np.kron(Z, Y)
        h1 = np.kron(I, Z)

        # np.random.seed(2)
        temp = np.random.random((2 ** SIZE, 2 ** SIZE))
        h1 = temp + temp.T
        qubit_op = Operator(matrix=h1)
        # qubit_op_jw.chop_by_threshold(10 ** -10)

        if qubit_op.grouped_paulis is None:
            qubit_op._matrix_to_paulis()
            qubit_op._paulis_to_grouped_paulis()

        for ps in qubit_op.grouped_paulis:
            for p1 in ps:
                for p2 in ps:
                    if p1 != p2:
                        np.testing.assert_almost_equal(
                            p1[1].to_matrix() @ p2[1].to_matrix(),
                            p2[1].to_matrix() @ p1[1].to_matrix()
                        )

        state_in = Custom(SIZE, state='random')

        evo_time = 1
        num_time_slices = 3

        # announces params
        self.log.debug('evo time:        {}'.format(evo_time))
        self.log.debug('num time slices: {}'.format(num_time_slices))
        self.log.debug('state_in:        {}'.format(state_in._state_vector))

        # get the exact state_out from raw matrix multiplication
        state_out_exact = qubit_op.evolve(
            state_in=state_in.construct_circuit('vector'),
            evo_time=evo_time,
            evo_mode='matrix',
            num_time_slices=0
        )
        # self.log.debug('exact:\n{}'.format(state_out_exact))
        qubit_op_temp = copy.deepcopy(qubit_op)
        for expansion_mode in ['trotter', 'suzuki']:
            self.log.debug(
                'Under {} expansion mode:'.format(expansion_mode))
            for expansion_order in [1, 2, 3, 4] if expansion_mode == 'suzuki' else [1]:
                # assure every time the operator from the original one
                qubit_op = copy.deepcopy(qubit_op_temp)
                if expansion_mode == 'suzuki':
                    self.log.debug(
                        'With expansion order {}:'.format(expansion_order))
                state_out_matrix = qubit_op.evolve(
                    state_in=state_in.construct_circuit('vector'),
                    evo_time=evo_time,
                    evo_mode='matrix',
                    num_time_slices=num_time_slices,
                    expansion_mode=expansion_mode,
                    expansion_order=expansion_order
                )

                quantum_registers = QuantumRegister(qubit_op.num_qubits, name='q')
                qc = QuantumCircuit(quantum_registers)
                qc += state_in.construct_circuit(
                    'circuit', quantum_registers)
                qc += qubit_op.evolve(
                    evo_time=evo_time,
                    evo_mode='circuit',
                    num_time_slices=num_time_slices,
                    quantum_registers=quantum_registers,
                    expansion_mode=expansion_mode,
                    expansion_order=expansion_order,
                )
                job = q_execute(qc, BasicAer.get_backend('statevector_simulator'))
                state_out_circuit = np.asarray(
                    job.result().get_statevector(qc, decimals=16))

                self.log.debug('The fidelity between exact and matrix:   {}'.format(
                    state_fidelity(state_out_exact, state_out_matrix)
                ))
                self.log.debug('The fidelity between exact and circuit:  {}'.format(
                    state_fidelity(state_out_exact, state_out_circuit)
                ))
                f_mc = state_fidelity(state_out_matrix, state_out_circuit)
                self.log.debug(
                    'The fidelity between matrix and circuit: {}'.format(f_mc))
                self.assertAlmostEqual(f_mc, 1)


if __name__ == '__main__':
    unittest.main()
