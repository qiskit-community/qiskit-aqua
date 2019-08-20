# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of UCCSD and HartreeFock Aqua extensions """

from test.chemistry.common import QiskitChemistryTestCase
from qiskit.chemistry import QiskitChemistry
# from qiskit.chemistry import set_qiskit_chemistry_logging
# import logging


class TestUCCSDHartreeFock(QiskitChemistryTestCase):
    """Test for these aqua extensions."""

    def setUp(self):
        super().setUp()
        self.config = {'driver': {'name': 'HDF5'},
                       'hdf5': {'hdf5_input': self._get_resource_path('test_driver_hdf5.hdf5')},
                       'operator': {'name': 'hamiltonian',
                                    'qubit_mapping': 'parity',
                                    'two_qubit_reduction': True},
                       'algorithm': {'name': 'VQE', 'operator_mode': 'matrix'},
                       'optimizer': {'name': 'SLSQP', 'maxiter': 100},
                       'variational_form': {'name': 'UCCSD'},
                       'initial_state': {'name': 'HartreeFock'},
                       'backend': {'provider': 'qiskit.BasicAer', 'name': 'statevector_simulator'}}
        self.reference_energy = -1.1373060356951838
        pass

    def test_uccsd_hf(self):
        """ uccsd hf test """
        # set_qiskit_chemistry_logging(logging.DEBUG)
        solver = QiskitChemistry()
        result = solver.run(self.config)
        self.assertAlmostEqual(result['energy'], self.reference_energy, places=6)
