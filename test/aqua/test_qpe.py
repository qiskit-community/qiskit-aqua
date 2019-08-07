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

import numpy as np
from parameterized import parameterized
from qiskit import BasicAer

from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import MatrixOperator, WeightedPauliOperator, op_converter
from qiskit.aqua.utils import decimal_to_binary
from qiskit.aqua.algorithms import ExactEigensolver
from qiskit.aqua.algorithms import QPE
from qiskit.aqua.components.iqfts import Standard
from qiskit.aqua.components.initial_states import Custom
from test.aqua.common import QiskitAquaTestCase


X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
_I = np.array([[1, 0], [0, 1]])
h1 = X + Y + Z + _I
qubit_op_simple = MatrixOperator(matrix=h1)
qubit_op_simple = op_converter.to_weighted_pauli_operator(qubit_op_simple)

pauli_dict = {
    'paulis': [
        {"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
        {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
        {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
        {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
        {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
    ]
}
qubit_op_h2_with_2_qubit_reduction = WeightedPauliOperator.from_dict(pauli_dict)


pauli_dict_zz = {
    'paulis': [
        {"coeff": {"imag": 0.0, "real": 1.0}, "label": "ZZ"}
    ]
}
qubit_op_zz = WeightedPauliOperator.from_dict(pauli_dict_zz)


class TestQPE(QiskitAquaTestCase):
    """QPE tests."""

    @parameterized.expand([
        [qubit_op_simple, 'qasm_simulator', 1, 5],
        [qubit_op_zz, 'statevector_simulator', 1, 1],
        [qubit_op_h2_with_2_qubit_reduction, 'statevector_simulator', 1, 6],
    ])
    def test_qpe(self, qubit_op, simulator, num_time_slices, n_ancillae):
        self.algorithm = 'QPE'
        self.log.debug('Testing QPE')

        self.qubit_op = qubit_op

        exact_eigensolver = ExactEigensolver(self.qubit_op, k=1)
        results = exact_eigensolver.run()

        self.ref_eigenval = results['eigvals'][0]
        self.ref_eigenvec = results['eigvecs'][0]
        self.log.debug('The exact eigenvalue is:       {}'.format(self.ref_eigenval))
        self.log.debug('The corresponding eigenvector: {}'.format(self.ref_eigenvec))

        state_in = Custom(self.qubit_op.num_qubits, state_vector=self.ref_eigenvec)
        iqft = Standard(n_ancillae)

        qpe = QPE(self.qubit_op, state_in, iqft, num_time_slices, n_ancillae,
                  expansion_mode='suzuki', expansion_order=2,
                  shallow_circuit_concat=True)

        backend = BasicAer.get_backend(simulator)
        quantum_instance = QuantumInstance(backend, shots=100)

        # run qpe
        result = qpe.run(quantum_instance)

        # report result
        self.log.debug('top result str label:         {}'.format(result['top_measurement_label']))
        self.log.debug('top result in decimal:        {}'.format(result['top_measurement_decimal']))
        self.log.debug('stretch:                      {}'.format(result['stretch']))
        self.log.debug('translation:                  {}'.format(result['translation']))
        self.log.debug('final eigenvalue from QPE:    {}'.format(result['energy']))
        self.log.debug('reference eigenvalue:         {}'.format(self.ref_eigenval))
        self.log.debug('ref eigenvalue (transformed): {}'.format(
            (self.ref_eigenval + result['translation']) * result['stretch'])
        )
        self.log.debug('reference binary str label:   {}'.format(decimal_to_binary(
            (self.ref_eigenval.real + result['translation']) * result['stretch'],
            max_num_digits=n_ancillae + 3,
            fractional_part_only=True
        )))

        np.testing.assert_approx_equal(result['energy'], self.ref_eigenval.real, significant=2)


if __name__ == '__main__':
    unittest.main()
