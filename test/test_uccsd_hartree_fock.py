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

"""
Test of UCCSD and HartreeFock Aqua extensions.
"""

from test.common import QiskitChemistryTestCase
from qiskit.chemistry import QiskitChemistry
# from qiskit.chemistry import set_qiskit_chemistry_logging
# import logging


class TestUCCSDHartreeFock(QiskitChemistryTestCase):
    """Test for these aqua extensions."""

    def setUp(self):
        self.config = {'driver': {'name': 'HDF5'},
                       'hdf5': {'hdf5_input': self._get_resource_path('test_driver_hdf5.hdf5')},
                       'operator': {'name': 'hamiltonian', 'qubit_mapping': 'parity', 'two_qubit_reduction': True},
                       'algorithm': {'name': 'VQE', 'operator_mode': 'matrix'},
                       'optimizer': {'name': 'SLSQP', 'maxiter': 100},
                       'variational_form': {'name': 'UCCSD'},
                       'initial_state': {'name': 'HartreeFock'},
                       'backend': {'provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'}}
        self.reference_energy = -1.1373060356951838
        pass

    def test_uccsd_hf(self):
        # set_qiskit_chemistry_logging(logging.DEBUG)
        solver = QiskitChemistry()
        result = solver.run(self.config)
        self.assertAlmostEqual(result['energy'], self.reference_energy, places=6)
