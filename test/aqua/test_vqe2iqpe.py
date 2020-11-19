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

""" Test VQE to IQPE """

import unittest
import warnings
from test.aqua import QiskitAquaTestCase
from ddt import ddt, data

from qiskit import BasicAer
from qiskit.circuit.library import TwoLocal

from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.operators import I, X, Z
from qiskit.aqua.utils import decimal_to_binary
from qiskit.aqua.components.initial_states import VarFormBased
from qiskit.aqua.components.optimizers import SPSA
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.algorithms import IQPE


@ddt
class TestVQE2IQPE(QiskitAquaTestCase):
    """ Test VQE to IQPE """

    def setUp(self):
        super().setUp()
        self.seed = 20798
        aqua_globals.random_seed = self.seed
        self.qubit_op = -1.052373245772859 * (I ^ I) \
            + 0.39793742484318045 * (I ^ Z) \
            - 0.39793742484318045 * (Z ^ I) \
            - 0.01128010425623538 * (Z ^ Z) \
            + 0.18093119978423156 * (X ^ X)

    @data('initial_state', 'circuit')
    def test_vqe_2_iqpe(self, mode):
        """ vqe to iqpe test """
        backend = BasicAer.get_backend('qasm_simulator')
        num_qbits = self.qubit_op.num_qubits
        wavefunction = TwoLocal(num_qbits, ['ry', 'rz'], 'cz', reps=3)

        optimizer = SPSA(maxiter=10)
        algo = VQE(self.qubit_op, wavefunction, optimizer)

        quantum_instance = QuantumInstance(backend, seed_simulator=self.seed,
                                           seed_transpiler=self.seed)
        result = algo.run(quantum_instance)

        self.log.debug('VQE result: %s.', result)

        ref_eigenval = -1.85727503  # Known reference value

        num_time_slices = 1
        num_iterations = 6

        param_dict = result.optimal_parameters
        if mode == 'initial_state':
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=DeprecationWarning)
                state_in = VarFormBased(wavefunction, param_dict)
        else:
            state_in = wavefunction.assign_parameters(param_dict)

        iqpe = IQPE(self.qubit_op, state_in, num_time_slices, num_iterations,
                    expansion_mode='suzuki', expansion_order=2,
                    shallow_circuit_concat=True)
        quantum_instance = QuantumInstance(
            backend, shots=100, seed_transpiler=self.seed, seed_simulator=self.seed
        )
        result = iqpe.run(quantum_instance)

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
            max_num_digits=num_iterations + 3,
            fractional_part_only=True
        ))

        self.assertAlmostEqual(result.eigenvalue.real, ref_eigenval.real, delta=1e-2)


if __name__ == '__main__':
    unittest.main()
