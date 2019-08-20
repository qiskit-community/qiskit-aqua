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

""" Test QPE """

import unittest
from test.aqua.common import QiskitAquaTestCase
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
from qiskit.aqua.utils import CircuitFactory

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
_I = np.array([[1, 0], [0, 1]])
H1 = X + Y + Z + _I
QUBIT_OP_SIMPLE = MatrixOperator(matrix=H1)
QUBIT_OP_SIMPLE = op_converter.to_weighted_pauli_operator(QUBIT_OP_SIMPLE)

PAULI_DICT = {
    'paulis': [
        {"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
        {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
        {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
        {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
        {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
    ]
}
QUBIT_OP_H2_WITH_2_QUBIT_REDUCTION = WeightedPauliOperator.from_dict(PAULI_DICT)


PAULI_DICT_ZZ = {
    'paulis': [
        {"coeff": {"imag": 0.0, "real": 1.0}, "label": "ZZ"}
    ]
}
QUBIT_OP_ZZ = WeightedPauliOperator.from_dict(PAULI_DICT_ZZ)


class FlipSuperposition(CircuitFactory):
    def __init__(self, state_in):
        self._state_in = state_in
        super().__init__(state_in._num_qubits)
            
    def build(self, qc, q, aux):

        if self._state_in is not None:
                qc.data += self._state_in.construct_circuit('circuit', q).data
        else:
            raise(AquaError("State-In not specified."))

        # Place Auxillary Register into Uniform Superposition
        qc.u2(0, np.pi, aux)
        # Conditionally flip state qubit register
        qc.cx(aux, q)

    def required_ancillas(self):
        return 1

class TestQPE(QiskitAquaTestCase):
    """QPE tests."""

    @parameterized.expand([
        [QUBIT_OP_SIMPLE, 'qasm_simulator', 1, 5],
        [QUBIT_OP_ZZ, 'statevector_simulator', 1, 1],
        [QUBIT_OP_H2_WITH_2_QUBIT_REDUCTION, 'statevector_simulator', 1, 5],
    ])
    def test_qpe(self, qubit_op, simulator, num_time_slices, n_ancillae):
        """ QPE test """
        self.log.debug('Testing QPE')

        exact_eigensolver = ExactEigensolver(qubit_op, k=1)
        results = exact_eigensolver.run()

        ref_eigenval = results['eigvals'][0]
        ref_eigenvec = results['eigvecs'][0]
        self.log.debug('The exact eigenvalue is:       %s', ref_eigenval)
        self.log.debug('The corresponding eigenvector: %s', ref_eigenvec)

        state_in = Custom(qubit_op.num_qubits, state_vector=ref_eigenvec)
        iqft = Standard(n_ancillae)

        qpe = QPE(qubit_op, iqft, state_in=state_in, num_time_slices=num_time_slices, num_ancillae=n_ancillae,
                  expansion_mode='suzuki', expansion_order=2,
                  shallow_circuit_concat=True)

        backend = BasicAer.get_backend(simulator)
        quantum_instance = QuantumInstance(backend, shots=100)

        # run qpe
        result = qpe.run(quantum_instance)

        # report result
        self.log.debug('top result str label:         %s', result['top_measurement_label'])
        self.log.debug('top result in decimal:        %s', result['top_measurement_decimal'])
        self.log.debug('stretch:                      %s', result['stretch'])
        self.log.debug('translation:                  %s', result['translation'])
        self.log.debug('final eigenvalue from QPE:    %s', result['energy'])
        self.log.debug('reference eigenvalue:         %s', ref_eigenval)
        self.log.debug('ref eigenvalue (transformed): %s',
                       (ref_eigenval + result['translation']) * result['stretch'])
        self.log.debug('reference binary str label:   %s', decimal_to_binary(
            (ref_eigenval.real + result['translation']) * result['stretch'],
            max_num_digits=n_ancillae + 3,
            fractional_part_only=True
        ))

        np.testing.assert_approx_equal(result['energy'], ref_eigenval.real, significant=2)

        #Re-run, now with state_in_circuit_factory
        superpose_state_and_flip = FlipSuperposition(state_in)
    
        qpe = QPE(qubit_op, iqft, state_in_circuit_factory=superpose_state_and_flip, num_time_slices=num_time_slices, num_ancillae=n_ancillae, expansion_mode='suzuki', expansion_order=2, shallow_circuit_concat=True)

        backend = BasicAer.get_backend(simulator)
        quantum_instance = QuantumInstance(backend, shots=100)

        # run qpe
        result = qpe.run(quantum_instance)

        ancilla_counts = result["ancilla_counts"]
        if simulator=="qasm_simulator":
            self.assertEqual(result['top_measurement_label'], sorted([(ancilla_counts[k], k) for k in ancilla_counts])[::-1][0][-1][::-1])
        else:
            self.assertEqual(len(ancilla_counts), 1<<n_ancillae)
            
        self.assertEqual(len(result["aux_counts"]), 1<<superpose_state_and_flip.required_ancillas()) 


if __name__ == '__main__':
    unittest.main()
