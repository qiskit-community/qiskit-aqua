# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Driver FCIDump Dumping """

import tempfile
import unittest
from abc import ABC, abstractmethod
from test.chemistry import QiskitChemistryTestCase
import numpy as np
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers import FCIDumpDriver, PySCFDriver, UnitsType


class BaseTestDriverFCIDumpDumper(ABC):
    """FCIDump Driver dumping base test class."""

    def __init__(self):
        self.log = None
        self.dumped = None
        self.core_energy = None
        self.num_orbitals = None
        self.num_electrons = None
        self.spin_number = None
        self.wf_symmetry = None
        self.orb_symmetries = None
        self.mo_onee = None
        self.mo_eri = None

    @abstractmethod
    def assertAlmostEqual(self, first, second, places=None, msg=None, delta=None):
        """ assert Almost Equal """
        raise Exception('Abstract method')

    @abstractmethod
    def assertEqual(self, first, second, msg=None):
        """ assert equal """
        raise Exception('Abstract method')

    @abstractmethod
    def assertSequenceEqual(self, seq1, seq2, msg=None, seq_type=None):
        """ assert Sequence Equal """
        raise Exception('Abstract method')

    def test_dumped_inactive_energy(self):
        """ dumped inactive energy test """
        self.log.debug('Dumped inactive energy is {:g}'.format(self.dumped['ECORE']))
        self.assertAlmostEqual(self.dumped['ECORE'], self.core_energy, places=3)

    def test_dumped_num_orbitals(self):
        """ dumped number of orbitals test """
        self.log.debug('Dumped number of orbitals is {:d}'.format(self.dumped['NORB']))
        self.assertEqual(self.dumped['NORB'], self.num_orbitals)

    def test_dumped_num_electrons(self):
        """ dumped number of electrons test """
        self.log.debug('Dumped number of electrons is {:d}'.format(self.dumped['NELEC']))
        self.assertEqual(self.dumped['NELEC'], self.num_electrons)

    def test_dumped_spin_number(self):
        """ dumped spin number test """
        self.log.debug('Dumped spin number is {:d}'.format(self.dumped['MS2']))
        self.assertEqual(self.dumped['MS2'], self.spin_number)

    def test_dumped_wave_function_sym(self):
        """ dumped wave function symmetry test """
        self.log.debug('Dumped wave function symmetry is {:d}'.format(self.dumped['ISYM']))
        self.assertEqual(self.dumped['ISYM'], self.wf_symmetry)

    def test_dumped_orbital_syms(self):
        """ dumped orbital symmetries test """
        self.log.debug('Dumped orbital symmetries is %s', self.dumped['ORBSYM'])
        self.assertEqual(self.dumped['ORBSYM'], self.orb_symmetries)

    def test_dumped_h1(self):
        """ dumped h1 integrals test """
        self.log.debug('Dumped h1 integrals are %s', self.dumped['H1'])
        np.testing.assert_array_almost_equal(np.absolute(self.dumped['H1']),
                                             np.absolute(self.mo_onee), decimal=4)

    def test_dumped_h2(self):
        """ dumped h2 integrals test """
        self.log.debug('Dumped h2 integrals are %s', self.dumped['H2'])
        np.testing.assert_array_almost_equal(np.absolute(self.dumped['H2']),
                                             np.absolute(self.mo_eri), decimal=4)


class TestDriverFCIDumpDumpH2(QiskitChemistryTestCase, BaseTestDriverFCIDumpDumper):
    """RHF FCIDump Driver tests."""

    def setUp(self):
        super().setUp()
        self.core_energy = 0.7199
        self.num_orbitals = 2
        self.num_electrons = 2
        self.spin_number = 0
        self.wf_symmetry = 1
        self.orb_symmetries = [1, 1]
        self.mo_onee = [[1.2563, 0.0], [0.0, 0.4719]]
        self.mo_eri = [0.6757, 0.0, 0.1809, 0.6646, 0.0, 0.6986]
        try:
            driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                 unit=UnitsType.ANGSTROM,
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
            qmolecule = driver.run()

            with tempfile.NamedTemporaryFile() as dump:
                FCIDumpDriver.dump(qmolecule, dump.name)
                # pylint: disable=import-outside-toplevel
                from pyscf.tools import fcidump as pyscf_fcidump
                self.dumped = pyscf_fcidump.read(dump.name)
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed.')
        except ImportError:
            self.skipTest('PYSCF driver does not appear to be installed.')


if __name__ == '__main__':
    unittest.main()
