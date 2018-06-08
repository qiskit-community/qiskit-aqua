# -*- coding: utf-8 -*-

# Copyright 2018 IBM RESEARCH. All Rights Reserved.
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
from parameterized import parameterized
from qiskit_acqua import get_algorithm_instance, get_initial_state_instance, get_iqft_instance, Operator
from qiskit_acqua.utils.subsystem import get_subsystem_fidelity
import numpy as np
from test.common import QISKitAcquaTestCase
from qiskit.wrapper import execute as q_execute
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.tools.qi.qi import qft


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])
h1 = X + Y + Z + I
qubitOp_simple = Operator(matrix=h1)


pauli_dict = {
    'paulis': [
        {"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
        {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "ZI"},
        {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "IZ"},
        {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
        {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
    ]
}
qubitOp_h2_with_2_qubit_reduction = Operator.load_from_dict(pauli_dict)


class TestQPE(QISKitAcquaTestCase):
    """QPE tests."""

    @parameterized.expand([
        [qubitOp_simple],
        [qubitOp_h2_with_2_qubit_reduction],
    ])
    def test_qpe(self, qubitOp):
        self.algorithm = 'QPE'
        self.log.debug('Testing QPE')

        self.qubitOp = qubitOp

        exact_eigensolver = get_algorithm_instance('ExactEigensolver')
        exact_eigensolver.init_args(self.qubitOp, k=1)
        results = exact_eigensolver.run()

        w = results['eigvals']
        v = results['eigvecs']

        # self.qubitOp._check_representation('matrix')
        # np.testing.assert_almost_equal(
        #     self.qubitOp.matrix @ v[0],
        #     w[0] * v[0]
        # )
        # np.testing.assert_almost_equal(
        #     expm(-1.j * self.qubitOp.matrix) @ v[0],
        #     np.exp(-1.j * w[0]) * v[0]
        # )

        self.ref_eigenval = w[0]
        self.ref_eigenvec = v[0]
        self.log.debug('The exact eigenvalue is:       {}'.format(self.ref_eigenval))
        self.log.debug('The corresponding eigenvector: {}'.format(self.ref_eigenvec))

        num_time_slices = 20
        n_ancillae = 8

        qpe = get_algorithm_instance('QPE')
        qpe.setup_quantum_backend(backend='local_qasm_simulator', shots=1000, skip_translation=False)

        state_in = get_initial_state_instance('CUSTOM')
        state_in.init_args(self.qubitOp.num_qubits, state_vector=self.ref_eigenvec)

        iqft = get_iqft_instance('STANDARD')
        iqft.init_args(n_ancillae)

        qpe.init_args(
            self.qubitOp, state_in, iqft, num_time_slices, n_ancillae,
            paulis_grouping='random',
            expansion_mode='suzuki',
            expansion_order=2,
            use_basis_gates=False
        )

        # check that controlled-U's don't alter the quantum state
        qpe._setup_qpe()
        qc = QuantumCircuit(
            qpe._ret['circuit_components']['registers']['a'],
            qpe._ret['circuit_components']['registers']['q']
        )
        qc += qpe._ret['circuit_components']['state_init']
        qc += qpe._ret['circuit_components']['ancilla_superposition']
        qc += qpe._ret['circuit_components']['phase_kickback']
        # self.log.debug('Phase kickback circuit:\n\n{}'.format(qc.qasm()))
        vec_qpe = np.asarray(q_execute(qc, 'local_statevector_simulator').result().get_statevector(qc))

        qc = qpe._ret['circuit_components']['state_init']
        # self.log.debug('Initial quantum state circuit:\n\n{}'.format(qc.qasm()))
        vec_init = np.asarray(q_execute(qc, 'local_statevector_simulator').result().get_statevector(qc))
        # self.log.debug('Full quantum state vector after phase kickback: {}'.format(vec_qpe))
        # self.log.debug('Concise quantum state vector after initialization: {}'.format(vec_init))
        fidelity = get_subsystem_fidelity(vec_qpe, range(n_ancillae), vec_init)
        self.log.debug('Quantum state fidelity after phase kickback: {}'.format(fidelity))
        # np.testing.assert_approx_equal(fidelity, 1, 4)

        # run the complete qpe
        result = qpe.run()
        self.log.debug('operator paulis:\n{}'.format(self.qubitOp.print_operators('paulis')))
        # self.log.debug('qpe circuit:\n\n{}'.format(result['circuit']['complete'].qasm()))
        qft_q = QuantumRegister(n_ancillae, 'a')
        qft_qc = QuantumCircuit(qft_q)
        qft(qft_qc, qft_q, n_ancillae)
        # self.log.debug('size {} qft circuit:\n\n{}'.format(n_ancillae, qft_qc.qasm()))

        self.log.debug('measurement results:   {}'.format(result['measurements']))
        self.log.debug('top result str label:  {}'.format(result['top_measurement_label']))
        self.log.debug('top result in decimal: {}'.format(result['top_measurement_decimal']))
        self.log.debug('stretch:               {}'.format(result['stretch']))
        self.log.debug('translation:           {}'.format(result['translation']))
        self.log.debug('final energy from QPE: {}'.format(result['energy']))
        self.log.debug('reference energy:      {}'.format(self.ref_eigenval))
        self.log.debug('reference energy d:    {}'.format((self.ref_eigenval + result['translation']) * result['stretch']))

        np.testing.assert_approx_equal(self.ref_eigenval, result['energy'], significant=2)


if __name__ == '__main__':
    unittest.main()
