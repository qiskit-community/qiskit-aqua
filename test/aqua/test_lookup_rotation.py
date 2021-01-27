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

""" Test Lookup Rotation """

import unittest
from test.aqua import QiskitAquaTestCase
import numpy as np
from ddt import ddt, idata, unpack
from qiskit import (QuantumRegister, QuantumCircuit, execute, BasicAer)
from qiskit.aqua.components.reciprocals.lookup_rotation import LookupRotation
from qiskit.quantum_info import (state_fidelity, Statevector)


@ddt
class TestLookupRotation(QiskitAquaTestCase):
    """Lookup Rotation tests."""

    @idata([[3, 1 / 2], [5, 1 / 4], [7, 1 / 8], [9, 1 / 16], [11, 1 / 32]])
    @unpack
    def test_lookup_rotation(self, reg_size, ref_rot):
        """ lookup rotation test """
        self.log.debug('Testing Lookup Rotation with positive eigenvalues')

        ref_sv_ampl = ref_rot**2
        ref_size = reg_size + 3  # add work, msq and anc qubits
        ref_dim = 2**ref_size
        ref_sv = np.zeros(ref_dim, dtype=complex)
        ref_sv[int(ref_dim / 2) + 1] = ref_sv_ampl + 0j
        ref_sv[1] = np.sqrt(1 - ref_sv_ampl ** 2) + 0j
        state = Statevector.from_label('0' * (reg_size - 1) + '1').data
        q_a = QuantumRegister(reg_size, name='a')
        init_circuit = QuantumCircuit(q_a)
        init_circuit.initialize(state, q_a)
        lrot = LookupRotation(negative_evals=False)
        lrot_circuit = init_circuit + lrot.construct_circuit('', q_a)
        lrot_sv = _sim_statevec(lrot_circuit)
        fidelity = state_fidelity(lrot_sv, ref_sv)
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('Lookup rotation register size: %s', reg_size)
        self.log.debug('Lookup rotation fidelity:      %s', fidelity)

    @idata([[3, 0], [5, 1 / 4], [7, 1 / 8], [9, 1 / 16], [11, 1 / 32]])
    @unpack
    def test_lookup_rotation_neg(self, reg_size, ref_rot):
        """ lookup rotation neg test """
        self.log.debug('Testing Lookup Rotation with support for negative '
                       'eigenvalues')

        ref_sv_ampl = ref_rot**2
        ref_size = reg_size + 3  # add work, msq and anc qubits
        ref_dim = 2**ref_size
        ref_sv = np.zeros(ref_dim, dtype=complex)
        ref_sv[int(ref_dim / 2) + 1] = -ref_sv_ampl + 0j
        ref_sv[1] = -np.sqrt(1 - ref_sv_ampl ** 2) + 0j
        state = Statevector.from_label('0' * (reg_size - 1) + '1').data
        q_a = QuantumRegister(reg_size, name='a')
        init_circuit = QuantumCircuit(q_a)
        init_circuit.initialize(state, q_a)
        lrot = LookupRotation(negative_evals=True)
        lrot_circuit = init_circuit + lrot.construct_circuit('', q_a)
        lrot_sv = _sim_statevec(lrot_circuit)
        fidelity = state_fidelity(lrot_sv, ref_sv)
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('Lookup rotation register size: %s', reg_size)
        self.log.debug('Lookup rotation fidelity:      %s', fidelity)


def _sim_statevec(qc):
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    result = job.result()
    state_vec = result.get_statevector(qc)
    return state_vec


if __name__ == '__main__':
    unittest.main()
