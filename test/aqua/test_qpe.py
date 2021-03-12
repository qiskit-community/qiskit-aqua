# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test QPE """

import unittest
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, data, unpack
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import MatrixOperator, WeightedPauliOperator
from qiskit.aqua.operators.legacy import op_converter
from qiskit.aqua.utils import decimal_to_binary
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.algorithms import QPE
from qiskit.circuit.library import QFT
from qiskit.aqua.components.initial_states import Custom

# pylint: disable=invalid-name


@ddt
class TestQPE(QiskitAquaTestCase):
    """QPE tests."""

    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    _I = np.array([[1, 0], [0, 1]])
    H1 = X + Y + Z + _I

    PAULI_DICT = {
        'paulis': [
            {"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
            {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
            {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
            {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
            {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
        ]
    }

    PAULI_DICT_ZZ = {
        'paulis': [
            {"coeff": {"imag": 0.0, "real": 1.0}, "label": "ZZ"}
        ]
    }

    def setUp(self):
        super().setUp()
        qubit_op_simple = MatrixOperator(matrix=TestQPE.H1)
        qubit_op_simple = op_converter.to_weighted_pauli_operator(qubit_op_simple)
        qubit_op_h2_with_2_qubit_reduction = \
            WeightedPauliOperator.from_dict(TestQPE.PAULI_DICT)
        qubit_op_zz = WeightedPauliOperator.from_dict(TestQPE.PAULI_DICT_ZZ)
        self._dict = {
            'QUBIT_OP_SIMPLE': qubit_op_simple.to_opflow(),
            'QUBIT_OP_ZZ': qubit_op_zz.to_opflow(),
            'QUBIT_OP_H2_WITH_2_QUBIT_REDUCTION': qubit_op_h2_with_2_qubit_reduction.to_opflow()
        }

    @data(
        ('QUBIT_OP_SIMPLE', 'qasm_simulator', 1, 5),
        ('QUBIT_OP_ZZ', 'statevector_simulator', 1, 1),
        ('QUBIT_OP_H2_WITH_2_QUBIT_REDUCTION', 'statevector_simulator', 1, 6),
    )
    @unpack
    def test_qpe(self, qubit_op, simulator, num_time_slices, n_ancillae):
        """Test the QPE algorithm."""
        self.log.debug('Testing QPE')
        qubit_op = self._dict[qubit_op]
        exact_eigensolver = NumPyMinimumEigensolver(qubit_op)
        results = exact_eigensolver.run()

        ref_eigenval = results.eigenvalue
        ref_eigenvec = results.eigenstate
        self.log.debug('The exact eigenvalue is:       %s', ref_eigenval)
        self.log.debug('The corresponding eigenvector: %s', ref_eigenvec)

        state_in = Custom(qubit_op.num_qubits, state_vector=ref_eigenvec)
        iqft = QFT(n_ancillae, do_swaps=False).inverse().reverse_bits()

        qpe = QPE(qubit_op, state_in, iqft, num_time_slices, n_ancillae,
                  expansion_mode='suzuki', expansion_order=2,
                  shallow_circuit_concat=True)

        backend = BasicAer.get_backend(simulator)
        quantum_instance = QuantumInstance(backend, shots=100, seed_transpiler=1, seed_simulator=1)

        # run qpe
        result = qpe.run(quantum_instance)

        # report result
        self.log.debug('top result str label:         %s', result.top_measurement_label)
        self.log.debug('top result in decimal:        %s', result.top_measurement_decimal)
        self.log.debug('stretch:                      %s', result.stretch)
        self.log.debug('translation:                  %s', result.translation)
        self.log.debug('final eigenvalue from QPE:    %s', result.eigenvalue)
        self.log.debug('reference eigenvalue:         %s', ref_eigenval)
        self.log.debug('ref eigenvalue (transformed): %s',
                       (ref_eigenval + result.translation) * result.stretch)
        self.log.debug('reference binary str label:   %s', decimal_to_binary(
            (ref_eigenval.real + result.translation) * result.stretch,
            max_num_digits=n_ancillae + 3,
            fractional_part_only=True
        ))

        self.assertAlmostEqual(result.eigenvalue.real, ref_eigenval.real, delta=2e-2)


if __name__ == '__main__':
    unittest.main()
