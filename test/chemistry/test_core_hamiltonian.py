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

""" Test Core Hamiltonian """

import warnings
import unittest

from test.chemistry import QiskitChemistryTestCase
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.core import Hamiltonian, TransformationType, QubitMappingType


class TestCoreHamiltonian(QiskitChemistryTestCase):
    """core/hamiltonian Driver tests."""

    def setUp(self):
        super().setUp()
        try:
            driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                 unit=UnitsType.ANGSTROM,
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
        self.qmolecule = driver.run()

    def _validate_vars(self, core, energy_shift=0.0, ph_energy_shift=0.0):
        self.assertAlmostEqual(core._hf_energy, -1.117, places=3)
        self.assertAlmostEqual(core._energy_shift, energy_shift)
        self.assertAlmostEqual(core._ph_energy_shift, ph_energy_shift)

    def _validate_info(self, core, num_particles=None,
                       num_orbitals=4, actual_two_qubit_reduction=False):
        num_particles = num_particles if num_particles is not None else (1, 1)
        z2symmetries = core.molecule_info.pop('z2symmetries')
        self.assertEqual(z2symmetries.is_empty(), True)
        self.assertEqual(core.molecule_info, {'num_particles': num_particles,
                                              'num_orbitals': num_orbitals,
                                              'two_qubit_reduction': actual_two_qubit_reduction})

    def _validate_input_object(self, qubit_op, num_qubits=4, num_paulis=15):
        self.assertTrue(isinstance(qubit_op, WeightedPauliOperator))
        self.assertIsNotNone(qubit_op)
        self.assertEqual(qubit_op.num_qubits, num_qubits)
        self.assertEqual(len(qubit_op.to_dict()['paulis']), num_paulis)

    def test_output(self):
        """ output test """
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.PARITY,
                           two_qubit_reduction=True,
                           freeze_core=False,
                           orbital_reduction=[])
        warnings.filterwarnings('always', category=DeprecationWarning)
        qubit_op, _ = core.run(self.qmolecule)
        self._validate_vars(core)
        self._validate_info(core, actual_two_qubit_reduction=True)
        self._validate_input_object(qubit_op, num_qubits=2, num_paulis=5)

    def test_jordan_wigner(self):
        """ jordan wigner test """
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                           two_qubit_reduction=False,
                           freeze_core=False,
                           orbital_reduction=[])
        warnings.filterwarnings('always', category=DeprecationWarning)
        qubit_op, _ = core.run(self.qmolecule)
        self._validate_vars(core)
        self._validate_info(core)
        self._validate_input_object(qubit_op)

    def test_jordan_wigner_2q(self):
        """ jordan wigner 2q test """
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                           two_qubit_reduction=True,
                           freeze_core=False,
                           orbital_reduction=[])
        warnings.filterwarnings('always', category=DeprecationWarning)
        qubit_op, _ = core.run(self.qmolecule)
        self._validate_vars(core)
        # Reported effective 2 qubit reduction should be false
        self._validate_info(core, actual_two_qubit_reduction=False)
        self._validate_input_object(qubit_op)

    def test_parity(self):
        """ parity test """
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.PARITY,
                           two_qubit_reduction=False,
                           freeze_core=False,
                           orbital_reduction=[])
        warnings.filterwarnings('always', category=DeprecationWarning)
        qubit_op, _ = core.run(self.qmolecule)
        self._validate_vars(core)
        self._validate_info(core)
        self._validate_input_object(qubit_op)

    def test_bravyi_kitaev(self):
        """ bravyi kitaev test """
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.BRAVYI_KITAEV,
                           two_qubit_reduction=False,
                           freeze_core=False,
                           orbital_reduction=[])
        warnings.filterwarnings('always', category=DeprecationWarning)
        qubit_op, _ = core.run(self.qmolecule)
        self._validate_vars(core)
        self._validate_info(core)
        self._validate_input_object(qubit_op)

    def test_particle_hole(self):
        """ particle hole test """
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        core = Hamiltonian(transformation=TransformationType.PARTICLE_HOLE,
                           qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                           two_qubit_reduction=False,
                           freeze_core=False,
                           orbital_reduction=[])
        warnings.filterwarnings('always', category=DeprecationWarning)
        qubit_op, _ = core.run(self.qmolecule)
        self._validate_vars(core, ph_energy_shift=-1.83696799)
        self._validate_info(core)
        self._validate_input_object(qubit_op)

    def test_freeze_core(self):
        """ freeze core test -- Should be in effect a no-op for H2 """
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                           two_qubit_reduction=False,
                           freeze_core=True,
                           orbital_reduction=[])
        warnings.filterwarnings('always', category=DeprecationWarning)
        qubit_op, _ = core.run(self.qmolecule)
        self._validate_vars(core)
        self._validate_info(core)
        self._validate_input_object(qubit_op)

    def test_orbital_reduction(self):
        """ orbital reduction test --- Remove virtual orbital just
            for test purposes (not sensible!)
        """
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        core = Hamiltonian(transformation=TransformationType.FULL,
                           qubit_mapping=QubitMappingType.JORDAN_WIGNER,
                           two_qubit_reduction=False,
                           freeze_core=False,
                           orbital_reduction=[-1])
        warnings.filterwarnings('always', category=DeprecationWarning)
        qubit_op, _ = core.run(self.qmolecule)
        self._validate_vars(core)
        self._validate_info(core, num_orbitals=2)
        self._validate_input_object(qubit_op, num_qubits=2, num_paulis=4)


if __name__ == '__main__':
    unittest.main()
