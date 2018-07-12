# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
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
from collections import OrderedDict
import numpy as np
from qiskit_acqua.utils import decimal_to_binary
from qiskit_acqua import get_algorithm_instance, get_initial_state_instance, get_iqft_instance
from test.common import QISKitAcquaChemistryTestCase
from qiskit_acqua_chemistry.drivers import ConfigurationManager
from qiskit_acqua_chemistry import FermionicOperator


class TestEnd2EndWithQPE(QISKitAcquaChemistryTestCase):
    """QPE tests."""

    @parameterized.expand([
        [0.5],
        [0.735],
        [1],
    ])
    def test_qpe(self, distance):
        self.algorithm = 'QPE'
        self.log.debug('Testing End-to-End with QPE on H2 with inter-atomic distance {}.'.format(distance))
        cfg_mgr = ConfigurationManager()
        pyscf_cfg = OrderedDict([
            ('atom', 'H .0 .0 .0; H .0 .0 {}'.format(distance)),
            ('unit', 'Angstrom'),
            ('charge', 0),
            ('spin', 0),
            ('basis', 'sto3g')
        ])
        section = {}
        section['properties'] = pyscf_cfg
        driver = cfg_mgr.get_driver_instance('PYSCF')
        self.molecule = driver.run(section)

        ferOp = FermionicOperator(h1=self.molecule._one_body_integrals, h2=self.molecule._two_body_integrals)
        self.qubitOp = ferOp.mapping(map_type='PARITY', threshold=1e-10).two_qubit_reduced_operator(2)

        exact_eigensolver = get_algorithm_instance('ExactEigensolver')
        exact_eigensolver.init_args(self.qubitOp, k=1)
        results = exact_eigensolver.run()
        self.reference_energy = results['energy']
        self.log.debug('The exact ground state energy is: {}'.format(results['energy']))

        num_particles = self.molecule._num_alpha + self.molecule._num_beta
        two_qubit_reduction = True
        num_orbitals = self.qubitOp.num_qubits + (2 if two_qubit_reduction else 0)
        qubit_mapping = 'parity'

        num_time_slices = 50
        n_ancillae = 9

        qpe = get_algorithm_instance('QPE')
        qpe.setup_quantum_backend(backend='local_qasm_simulator', shots=100, skip_transpiler=True)

        state_in = get_initial_state_instance('HartreeFock')
        state_in.init_args(self.qubitOp.num_qubits, num_orbitals, qubit_mapping, two_qubit_reduction, num_particles)

        iqft = get_iqft_instance('STANDARD')
        iqft.init_args(n_ancillae)

        qpe.init_args(
            self.qubitOp, state_in, iqft, num_time_slices, n_ancillae,
            paulis_grouping='random',
            expansion_mode='suzuki',
            expansion_order=2
        )

        result = qpe.run()

        self.log.debug('measurement results:      {}'.format(result['measurements']))
        self.log.debug('top result str label:     {}'.format(result['top_measurement_label']))
        self.log.debug('top result in decimal:    {}'.format(result['top_measurement_decimal']))
        self.log.debug('stretch:                  {}'.format(result['stretch']))
        self.log.debug('translation:              {}'.format(result['translation']))
        self.log.debug('final energy from QPE:    {}'.format(result['energy']))
        self.log.debug('reference energy:         {}'.format(self.reference_energy))
        self.log.debug('ref energy (transformed): {}'.format(
            (self.reference_energy + result['translation']) * result['stretch'])
        )
        self.log.debug('ref binary str label:     {}'.format(decimal_to_binary(
            (self.reference_energy + result['translation']) * result['stretch'],
            max_num_digits=n_ancillae + 3,
            fractional_part_only=True
        )))

        np.testing.assert_approx_equal(result['energy'], self.reference_energy, significant=2)


if __name__ == '__main__':
    unittest.main()
