# -*- coding: utf-8 -*-

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
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, idata, unpack
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.utils import decimal_to_binary
from qiskit.aqua.algorithms import IQPEMinimumEigensolver
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.aqua.operators.legacy import op_converter
from qiskit.aqua.components.initial_states import Custom


def _params_generator():
    # pylint: disable=invalid-name
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    _I = np.array([[1, 0], [0, 1]])
    H1 = X + Y + Z + _I
    qubit_op_simple = MatrixOperator(matrix=H1)
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

    for x in [[qubit_op_simple, 'qasm_simulator', 1, 5],
              [qubit_op_zz, 'statevector_simulator', 1, 1],
              [qubit_op_h2_with_2_qubit_reduction, 'statevector_simulator', 1, 6]]:
        yield x


@ddt
class TestIQPE(QiskitAquaTestCase):
    """IQPE tests."""

    @idata(_params_generator())
    @unpack
    def test_iqpe(self, qubit_op, simulator, num_time_slices, num_iterations):
        """ iqpe test """
        self.log.debug('Testing IQPE')
        tmp_qubit_op = qubit_op.copy()
        exact_eigensolver = NumPyMinimumEigensolver(qubit_op)
        results = exact_eigensolver.run()

        ref_eigenval = results.eigenvalue
        ref_eigenvec = results.eigenstate
        self.log.debug('The exact eigenvalue is:       %s', ref_eigenval)
        self.log.debug('The corresponding eigenvector: %s', ref_eigenvec)

        state_in = Custom(qubit_op.num_qubits, state_vector=ref_eigenvec)
        iqpe = IQPEMinimumEigensolver(qubit_op, state_in, num_time_slices, num_iterations,
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
        self.assertEqual(tmp_qubit_op, qubit_op, "Operator is modified after IQPE.")


if __name__ == '__main__':
    unittest.main()
