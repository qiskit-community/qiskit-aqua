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
from collections import OrderedDict

from qiskit_acqua import get_algorithm_instance, get_initial_state_instance
from test.common import QISKitAcquaChemistryTestCase
from qiskit_acqua_chemistry.drivers import ConfigurationManager
from qiskit_acqua_chemistry import FermionicOperator


class TestIQPE(QISKitAcquaChemistryTestCase):
    """IQPE tests."""

    def setUp(self):
        self.algorithm = 'IQPE'
        self.log.debug('Testing IQPE with H2')
        cfg_mgr = ConfigurationManager()
        pyscf_cfg = OrderedDict([
            ('atom', 'H .0 .0 .0; H .0 .0 0.735'),
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
        self.qubitOp = ferOp.mapping(map_type='JORDAN_WIGNER', threshold=1e-10)

        exact_eigensolver = get_algorithm_instance('ExactEigensolver')
        exact_eigensolver.init_args(self.qubitOp, k=1)
        results = exact_eigensolver.run()
        self.reference_energy = results['energy']
        self.log.debug('The exact ground state energy is: {}'.format(results['energy']))

    def test_qpe(self):
        num_particles = self.molecule._num_alpha + self.molecule._num_beta
        two_qubit_reduction = False
        num_orbitals = self.qubitOp.num_qubits + (2 if two_qubit_reduction else 0)
        qubit_mapping = 'jordan_wigner'

        num_time_slices = 1
        num_iterations = 5

        iqpe = get_algorithm_instance('IQPE')
        iqpe.setup_quantum_backend(backend='local_qasm_simulator', shots=20)

        state_in = get_initial_state_instance('HartreeFock')
        state_in.init_args(self.qubitOp.num_qubits, num_orbitals, qubit_mapping, two_qubit_reduction, num_particles)

        iqpe.init_args(self.qubitOp, state_in, num_time_slices, num_iterations,
                       paulis_grouping='default', expansion_mode='trotter', expansion_order=2)

        result = iqpe.run()

        self.log.debug('phase estimation:  {}'.format(result['phase']))
        self.log.debug('energy estimation: {}'.format(result['energy']))
        self.log.debug('reference energy:  {}'.format(self.reference_energy))


if __name__ == '__main__':
    unittest.main()
