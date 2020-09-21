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

""" Test of the Adaptive VQE ground state calculations """
import unittest
from test.chemistry import QiskitChemistryTestCase

from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.providers.basicaer import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.ground_state_calculation import AdaptVQEGroundStateCalculation


class TestAdaptVQEGroundStateCalculation(QiskitChemistryTestCase):
    """ Test Adaptive VQE Ground State Calculation """
    def setUp(self):
        super().setUp()

        try:
            self.driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                      unit=UnitsType.ANGSTROM,
                                      basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
            return

        self.expected = -1.137306

        self.qinst = QuantumInstance(BasicAer.get_backend('statevector_simulator'))

    def test_default(self):
        """ Default execution """
        calc = AdaptVQEGroundStateCalculation(self.qinst)
        res = calc.compute_ground_state(self.driver)
        self.assertAlmostEqual(res.energy, self.expected, places=6)

if __name__ == '__main__':
    unittest.main()
