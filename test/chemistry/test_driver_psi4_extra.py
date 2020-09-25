# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver PSI4 """

import unittest

from test.chemistry import QiskitChemistryTestCase
from qiskit.chemistry.drivers import PSI4Driver
from qiskit.chemistry import QiskitChemistryError


class TestDriverPSI4Extra(QiskitChemistryTestCase):
    """PSI4 Driver extra tests for driver specifics, errors etc"""

    def setUp(self):
        super().setUp()
        try:
            PSI4Driver._check_valid()
        except QiskitChemistryError:
            self.skipTest('PSI4 driver does not appear to be installed')

    def test_input_format_list(self):
        """ input as a list"""
        driver = PSI4Driver([
            'molecule h2 {',
            '  0 1',
            '  H  0.0 0.0 0.0',
            '  H  0.0 0.0 0.735',
            '  no_com',
            '  no_reorient',
            '}',
            '',
            'set {',
            '  basis sto-3g',
            '  scf_type pk',
            '}'])
        qmolecule = driver.run()
        self.assertAlmostEqual(qmolecule.hf_energy, -1.117, places=3)

    def test_input_format_string(self):
        """ input as a multi line string """
        cfg = """
molecule h2 {
0 1
H  0.0 0.0 0.0
H  0.0 0.0 0.735
no_com
no_reorient
}

set {
basis sto-3g
scf_type pk
}
"""
        driver = PSI4Driver(cfg)
        qmolecule = driver.run()
        self.assertAlmostEqual(qmolecule.hf_energy, -1.117, places=3)

    def test_input_format_fail(self):
        """ input type failure """
        with self.assertRaises(QiskitChemistryError):
            _ = PSI4Driver(1.000)

    def test_psi4_failure(self):
        """ check we catch psi4 failures (bad scf type used here) """
        bad_cfg = """
molecule h2 {
0 1
H  0.0 0.0 0.0
H  0.0 0.0 0.735
no_com
no_reorient
}

set {
basis sto-3g
scf_type unknown
}
"""
        driver = PSI4Driver(bad_cfg)
        with self.assertRaises(QiskitChemistryError) as ctxmgr:
            _ = driver.run()
        self.assertTrue(str(ctxmgr.exception).startswith("'psi4 process return code"))


if __name__ == '__main__':
    unittest.main()
