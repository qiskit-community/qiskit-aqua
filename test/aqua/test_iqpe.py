# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test IQPE """

import unittest
import warnings
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, idata, unpack
from qiskit import BasicAer, QuantumCircuit
from qiskit.aqua import QuantumInstance
from qiskit.aqua.utils import decimal_to_binary
from qiskit.aqua.algorithms import IQPE
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.aqua.operators.legacy import op_converter
from qiskit.aqua.components.initial_states import Custom

# pylint: disable=invalid-name


@ddt
class TestIQPE(QiskitAquaTestCase):
    """IQPE tests."""

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
        qubit_op_simple = MatrixOperator(matrix=TestIQPE.H1)
        qubit_op_simple = op_converter.to_weighted_pauli_operator(qubit_op_simple)
        qubit_op_h2_with_2_qubit_reduction = WeightedPauliOperator.from_dict(TestIQPE.PAULI_DICT)
        qubit_op_zz = WeightedPauliOperator.from_dict(TestIQPE.PAULI_DICT_ZZ)
        self._dict = {
            'QUBIT_OP_SIMPLE': qubit_op_simple.to_opflow(),
            'QUBIT_OP_ZZ': qubit_op_zz.to_opflow(),
            'QUBIT_OP_H2_WITH_2_QUBIT_REDUCTION': qubit_op_h2_with_2_qubit_reduction.to_opflow()
        }

    @idata([
        ['QUBIT_OP_SIMPLE', 'qasm_simulator', 1, 5, True],
        ['QUBIT_OP_ZZ', 'statevector_simulator', 1, 1, False],
        ['QUBIT_OP_H2_WITH_2_QUBIT_REDUCTION', 'statevector_simulator', 1, 6, True],
    ])
    @unpack
    def test_iqpe(self, qubit_op, simulator, num_time_slices, num_iterations, use_circuits):
        """ iqpe test """
        self.log.debug('Testing IQPE')
        qubit_op = self._dict[qubit_op]
        exact_eigensolver = NumPyMinimumEigensolver(qubit_op)
        results = exact_eigensolver.run()

        ref_eigenval = results.eigenvalue
        ref_eigenvec = results.eigenstate
        self.log.debug('The exact eigenvalue is:       %s', ref_eigenval)
        self.log.debug('The corresponding eigenvector: %s', ref_eigenvec)

        if use_circuits:
            state_in = QuantumCircuit(qubit_op.num_qubits)
            state_in.initialize(ref_eigenvec.primitive.data, state_in.qubits)
        else:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                state_in = Custom(qubit_op.num_qubits, state_vector=ref_eigenvec)
        iqpe = IQPE(qubit_op, state_in, num_time_slices, num_iterations,
                    expansion_mode='suzuki', expansion_order=2,
                    shallow_circuit_concat=True)

        backend = BasicAer.get_backend(simulator)
        quantum_instance = QuantumInstance(backend, shots=100)

        result = iqpe.run(quantum_instance)
        self.log.debug('top result str label:         %s', result.top_measurement_label)
        self.log.debug('top result in decimal:        %s', result.top_measurement_decimal)
        self.log.debug('stretch:                      %s', result.stretch)
        self.log.debug('translation:                  %s', result.translation)
        self.log.debug('final eigenvalue from IQPE:   %s', result.eigenvalue)
        self.log.debug('reference eigenvalue:         %s', ref_eigenval)
        self.log.debug('ref eigenvalue (transformed): %s',
                       (ref_eigenval.real + result.translation) * result.stretch)
        self.log.debug('reference binary str label:   %s', decimal_to_binary(
            (ref_eigenval.real + result.translation) * result.stretch,
            max_num_digits=num_iterations + 3,
            fractional_part_only=True
        ))

        np.testing.assert_approx_equal(result.eigenvalue.real, ref_eigenval.real, significant=2)


if __name__ == '__main__':
    unittest.main()
