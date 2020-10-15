# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of Symmetry UCCSD processing """

import unittest

from test.chemistry import QiskitChemistryTestCase
from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.aqua.components.optimizers import SLSQP
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.chemistry.algorithms.ground_state_solvers import GroundStateEigensolver
from qiskit.chemistry.core import TransformationType, QubitMappingType
from qiskit.chemistry.transformations import FermionicTransformation


class TestSymmetries(QiskitChemistryTestCase):
    """Test for symmetry processing."""

    def setUp(self):
        super().setUp()
        try:
            self.driver = PySCFDriver(atom='Li .0 .0 .0; H .0 .0 1.6',
                                      unit=UnitsType.ANGSTROM,
                                      charge=0,
                                      spin=0,
                                      basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')

        self.fermionic_transformation = FermionicTransformation(
            transformation=TransformationType.FULL,
            qubit_mapping=QubitMappingType.PARITY,
            two_qubit_reduction=True,
            freeze_core=True,
            orbital_reduction=[],
            z2symmetry_reduction='auto')

        self.qubit_op, _ = self.fermionic_transformation.transform(self.driver)

        self.z2_symmetries = self.fermionic_transformation.molecule_info.pop('z2_symmetries')

        self.reference_energy = -7.882096489442

    def test_symmetries(self):
        """ symmetries test """
        labels = [symm.to_label() for symm in self.z2_symmetries.symmetries]
        self.assertSequenceEqual(labels, ['ZIZIZIZI', 'ZZIIZZII'])

    def test_sq_paulis(self):
        """ sq paulis test """
        labels = [sq.to_label() for sq in self.z2_symmetries.sq_paulis]
        self.assertSequenceEqual(labels, ['IIIIIIXI', 'IIIIIXII'])

    def test_cliffords(self):
        """ clifford test """
        self.assertEqual(2, len(self.z2_symmetries.cliffords))

    def test_sq_list(self):
        """ sq list test """
        self.assertSequenceEqual(self.z2_symmetries.sq_list, [1, 2])

    def test_tapered_op(self):
        """ tapered op test """

        optimizer = SLSQP(maxiter=1000)
        init_state = HartreeFock(
            num_orbitals=self.fermionic_transformation.molecule_info['num_orbitals'],
            qubit_mapping=self.fermionic_transformation._qubit_mapping,
            two_qubit_reduction=self.fermionic_transformation._two_qubit_reduction,
            num_particles=self.fermionic_transformation.molecule_info['num_particles'],
            sq_list=self.z2_symmetries.sq_list)

        var_form = UCCSD(
            num_orbitals=self.fermionic_transformation.molecule_info['num_orbitals'],
            num_particles=self.fermionic_transformation.molecule_info['num_particles'],
            active_occupied=None,
            active_unoccupied=None,
            initial_state=init_state,
            qubit_mapping=self.fermionic_transformation._qubit_mapping,
            two_qubit_reduction=self.fermionic_transformation._two_qubit_reduction,
            num_time_slices=1,
            z2_symmetries=self.z2_symmetries)

        solver = VQE(var_form=var_form, optimizer=optimizer,
                     quantum_instance=QuantumInstance(
                         backend=BasicAer.get_backend('statevector_simulator')))

        gsc = GroundStateEigensolver(self.fermionic_transformation, solver)

        result = gsc.solve(self.driver)
        self.assertAlmostEqual(result.total_energies[0], self.reference_energy, places=6)


if __name__ == '__main__':
    unittest.main()
