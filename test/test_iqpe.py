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

import numpy as np
from parameterized import parameterized
from scipy.linalg import expm
from scipy import sparse
from qiskit.transpiler import PassManager
from qiskit.aqua import get_aer_backend
from qiskit.qobj import RunConfig
from test.common import QiskitAquaTestCase
from qiskit.aqua import Operator, QuantumInstance
from qiskit.aqua.utils import decimal_to_binary
from qiskit.aqua.algorithms import IQPE
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.components.initial_states import Custom


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])
h1 = X + Y + Z + I
qubitOp_simple = Operator(matrix=h1)


pauli_dict = {
    'paulis': [
        {"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
        {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
        {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
        {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
        {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
    ]
}
qubitOp_h2_with_2_qubit_reduction = Operator.load_from_dict(pauli_dict)


class TestIQPE(QiskitAquaTestCase):
    """IQPE tests."""

    @parameterized.expand([
        [qubitOp_simple, 'qasm_simulator'],
        [qubitOp_h2_with_2_qubit_reduction, 'statevector_simulator'],
    ])
    def test_iqpe(self, qubitOp, simulator):
        self.algorithm = 'IQPE'
        self.log.debug('Testing IQPE')

        self.qubitOp = qubitOp

        exact_eigensolver = ExactEigensolver(self.qubitOp, k=1)
        results = exact_eigensolver.run()

        w = results['eigvals']
        v = results['eigvecs']

        self.qubitOp.to_matrix()
        np.testing.assert_almost_equal(
            self.qubitOp.matrix @ v[0],
            w[0] * v[0]
        )
        np.testing.assert_almost_equal(
            expm(-1.j * sparse.csc_matrix(self.qubitOp.matrix)) @ v[0],
            np.exp(-1.j * w[0]) * v[0]
        )

        self.ref_eigenval = w[0]
        self.ref_eigenvec = v[0]
        self.log.debug('The exact eigenvalue is:       {}'.format(self.ref_eigenval))
        self.log.debug('The corresponding eigenvector: {}'.format(self.ref_eigenvec))

        num_time_slices = 50
        num_iterations = 6
        state_in = Custom(self.qubitOp.num_qubits, state_vector=self.ref_eigenvec)
        iqpe = IQPE(self.qubitOp, state_in, num_time_slices, num_iterations,
                    expansion_mode='suzuki', expansion_order=2, shallow_circuit_concat=True)

        backend = get_aer_backend(simulator)
        run_config = RunConfig(shots=100, max_credits=10, memory=False)
        quantum_instance = QuantumInstance(backend, run_config, pass_manager=PassManager())

        result = iqpe.run(quantum_instance)

        self.log.debug('top result str label:         {}'.format(result['top_measurement_label']))
        self.log.debug('top result in decimal:        {}'.format(result['top_measurement_decimal']))
        self.log.debug('stretch:                      {}'.format(result['stretch']))
        self.log.debug('translation:                  {}'.format(result['translation']))
        self.log.debug('final eigenvalue from IQPE:   {}'.format(result['energy']))
        self.log.debug('reference eigenvalue:         {}'.format(self.ref_eigenval))
        self.log.debug('ref eigenvalue (transformed): {}'.format(
            (self.ref_eigenval + result['translation']) * result['stretch'])
        )
        self.log.debug('reference binary str label:   {}'.format(decimal_to_binary(
            (self.ref_eigenval.real + result['translation']) * result['stretch'],
            max_num_digits=num_iterations + 3,
            fractional_part_only=True
        )))

        np.testing.assert_approx_equal(result['energy'], self.ref_eigenval.real, significant=2)


if __name__ == '__main__':
    unittest.main()
