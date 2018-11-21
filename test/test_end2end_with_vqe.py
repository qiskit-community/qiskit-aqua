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
from qiskit_aqua import run_algorithm
from qiskit import Aer

from test.common import QiskitAquaChemistryTestCase
from qiskit_aqua_chemistry.drivers import ConfigurationManager
from qiskit_aqua_chemistry.core import Hamiltonian


class TestEnd2End(QiskitAquaChemistryTestCase):
    """End2End tests."""

    def setUp(self):
        cfg_mgr = ConfigurationManager()
        hdf5_cfg = OrderedDict([
            ('hdf5_input', self._get_resource_path('test_driver_hdf5.hdf5'))
        ])
        section = {'properties': hdf5_cfg}
        driver = cfg_mgr.get_driver_instance('HDF5')
        self.qmolecule = driver.run(section)

        core = Hamiltonian(transformation='full', qubit_mapping='parity',
                           two_qubit_reduction=True,
                           freeze_core=False,
                           orbital_reduction=[],
                           max_workers=4)

        self.algo_input = core.run(self.qmolecule)
        self.reference_energy = -1.857275027031588

    @parameterized.expand([
        ['COBYLA_M', 'COBYLA', Aer.get_backend('statevector_simulator'), 'matrix', 1],
        ['COBYLA_P', 'COBYLA', Aer.get_backend('statevector_simulator'), 'paulis', 1],
        # ['SPSA_P', 'SPSA', 'qasm_simulator', 'paulis', 1024],
        # ['SPSA_GP', 'SPSA', 'qasm_simulator', 'grouped_paulis', 1024]
    ])
    def test_end2end_h2(self, name, optimizer, backend, mode, shots):

        optimizer_params = {'name': optimizer}
        if optimizer == 'COBYLA':
            optimizer_params['maxiter'] = 1000
        elif optimizer == 'SPSA':
            optimizer_params['max_trials'] = 2000
            optimizer_params['save_steps'] = 25

        algo_params = {'problem': {'name': 'energy'},
                       'backend': {'name': backend, 'shots': shots},
                       'algorithm': {'name': 'VQE', 'operator_mode': mode},
                       'optimizer': optimizer_params,
                       'variational_form': {'name': 'RYRZ', 'depth': 3, 'entanglement': 'full'}
                       }

        results = run_algorithm(algo_params, self.algo_input)
        self.assertAlmostEqual(results['energy'], self.reference_energy, places=6)


if __name__ == '__main__':
    unittest.main()
