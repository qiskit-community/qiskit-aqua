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
from parameterized import parameterized
from test.common import QISKitAcquaChemistryTestCase

from qiskit_acqua import run_algorithm
from qiskit_acqua.input import get_input_instance

from qiskit_acqua_chemistry.drivers import ConfigurationManager
from qiskit_acqua_chemistry.core import get_chemistry_operator_instance

@unittest.skipUnless(QISKitAcquaChemistryTestCase.SLOW_TEST, 'slow')
class TestEnd2End(QISKitAcquaChemistryTestCase):
    """End2End tests."""

    def setUp(self):
        cfg_mgr = ConfigurationManager()
        pyscf_cfg = OrderedDict([
            ('atom', 'H .0 .0 .0; H .0 .0 0.735'),
            ('unit', 'Angstrom'),
            ('charge', 0),
            ('spin', 0),
            ('basis', 'sto3g')
        ])
        section = {'properties': pyscf_cfg}
        driver = cfg_mgr.get_driver_instance('PYSCF')
        self.qmolecule = driver.run(section)

        core = get_chemistry_operator_instance('hamiltonian')
        hamiltonian_cfg = OrderedDict([
            ('name', 'hamiltonian'),
            ('transformation', 'full'),
            ('qubit_mapping', 'parity'),
            ('two_qubit_reduction', True),
            ('freeze_core', False),
            ('orbital_reduction', [])
        ])
        core.init_params(hamiltonian_cfg)
        self.algo_input = core.run(self.qmolecule)


        algo_params = {'problem': {'name': 'energy', 'random_seed': 50},
                        'algorithm': {'name': 'ExactEigensolver', 'k': 1} }

        results = run_algorithm(algo_params, self.algo_input)
        self.reference_energy = results['energy']

    @parameterized.expand([
        ['COBYLA', 'local_statevector_simulator', 'matrix', 1],
        ['COBYLA', 'local_statevector_simulator', 'paulis', 1],
        ['SPSA', 'local_qasm_simulator', 'paulis', 1024],
        ['SPSA', 'local_qasm_simulator', 'grouped_paulis', 1024]
    ])
    def test_end2end_H2(self, optimizer, backend, mode, shots):

        optimizer_params = {'name': optimizer}
        if optimizer == 'COBYLA':
            optimizer_params['maxiter'] = 1000
        elif optimizer == 'SPSA':
            optimizer_params['max_trials'] = 1000
            optimizer_params['save_steps'] = 25

        algo_params = {'problem': {'name': 'energy'},
                    'backend': {'name': backend, 'shots': shots},
                    'algorithm': {'name': 'VQE'},
                    'optimizer': optimizer_params,
                    'variational_form': {'name': 'RYRZ', 'depth': 3, 'entanglement': 'full'}
                    }

        results = run_algorithm(algo_params, self.algo_input)
        self.assertAlmostEqual(results['energy'], self.reference_energy)

if __name__ == '__main__':
    unittest.main()

