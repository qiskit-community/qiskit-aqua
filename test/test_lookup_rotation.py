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

from parameterized import parameterized
from qiskit import QuantumRegister, QuantumCircuit
from test.common import QiskitAquaTestCase
from qiskit.aqua.components.reciprocals.lookup_rotation import LookupRotation
from qiskit import execute
from qiskit import BasicAer
import numpy as np
from qiskit.quantum_info import state_fidelity, basis_state

class TestLookupRotation(QiskitAquaTestCase):
    """Lookup Rotation tests."""

    #def setUp(self):

    @parameterized.expand([[3, 1/2], [5, 1/4], [7, 1/8], [9, 1/16], [11, 1/32]])
    def test_lookup_rotation(self, reg_size, ref_rot):
        self.log.debug('Testing Lookup Rotation with positive eigenvalues')

        ref_sv_ampl = ref_rot**2
        ref_size = reg_size + 3 # add work, msq and anc qubits
        ref_dim = 2**ref_size
        ref_sv = np.zeros(ref_dim, dtype=complex)
        ref_sv[int(ref_dim/2)+1] = ref_sv_ampl+0j
        ref_sv[1] = np.sqrt(1-ref_sv_ampl**2)+0j
        state = basis_state('1', reg_size)
        a = QuantumRegister(reg_size, name='a')
        init_circuit = QuantumCircuit(a)
        init_circuit.initialize(state, a)
        lrot = LookupRotation(negative_evals=False)
        lrot_circuit = init_circuit + lrot.construct_circuit('', a)
        lrot_sv = sim_statevec(lrot_circuit)
        fidelity = state_fidelity(lrot_sv, ref_sv)
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('Lookup rotation register size: {}'.format(reg_size))
        self.log.debug('Lookup rotation fidelity:      {}'.format(fidelity))

    @parameterized.expand([[3, 0], [5, 1/4], [7, 1/8], [9, 1/16], [11, 1/32]])
    def test_lookup_rotation_neg(self, reg_size, ref_rot):
        self.log.debug('Testing Lookup Rotation with support for negative '
                       'eigenvalues')

        ref_sv_ampl = ref_rot**2
        ref_size = reg_size + 3 # add work, msq and anc qubits
        ref_dim = 2**ref_size
        ref_sv = np.zeros(ref_dim, dtype=complex)
        ref_sv[int(ref_dim/2)+1] = -ref_sv_ampl+0j
        ref_sv[1] = -np.sqrt(1-ref_sv_ampl**2)+0j
        state = basis_state('1', reg_size)
        a = QuantumRegister(reg_size, name='a')
        init_circuit = QuantumCircuit(a)
        init_circuit.initialize(state, a)
        lrot = LookupRotation(negative_evals=True)
        lrot_circuit = init_circuit + lrot.construct_circuit('', a)
        lrot_sv = sim_statevec(lrot_circuit)
        fidelity = state_fidelity(lrot_sv, ref_sv)
        np.testing.assert_approx_equal(fidelity, 1, significant=5)

        self.log.debug('Lookup rotation register size: {}'.format(reg_size))
        self.log.debug('Lookup rotation fidelity:      {}'.format(fidelity))


def sim_statevec(qc):
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(qc, backend)
    result = job.result()
    state_vec = result.get_statevector(qc)
    return state_vec


if __name__ == '__main__':
    unittest.main()
