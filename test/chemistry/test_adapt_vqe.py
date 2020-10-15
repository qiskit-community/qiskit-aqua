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

""" Test of the Adaptive VQE ground state calculations """
import unittest
from test.chemistry import QiskitChemistryTestCase

from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.providers.basicaer import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import VQE
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.components.variational_forms import UCCSD
from qiskit.chemistry.components.initial_states import HartreeFock
from qiskit.aqua.components.optimizers import L_BFGS_B
from qiskit.chemistry.algorithms.ground_state_solvers import AdaptVQE, VQEUCCSDFactory
from qiskit.chemistry.transformations import FermionicTransformation


class TestAdaptVQE(QiskitChemistryTestCase):
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

        self.expected = -1.85727503

        self.transformation = FermionicTransformation()

    def test_default(self):
        """ Default execution """
        solver = VQEUCCSDFactory(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        calc = AdaptVQE(self.transformation, solver)
        res = calc.solve(self.driver)
        self.assertAlmostEqual(res.electronic_energy, self.expected, places=6)

    def test_custom_minimum_eigensolver(self):
        """ Test custom MES """
        # Note: the VQEUCCSDFactory actually allows to specify an optimizer through its constructor.
        # Thus, this example is quite far fetched but for a proof-of-principle test it still works.
        class CustomFactory(VQEUCCSDFactory):
            """A custom MESFactory"""

            def get_solver(self, transformation):
                num_orbitals = transformation.molecule_info['num_orbitals']
                num_particles = transformation.molecule_info['num_particles']
                qubit_mapping = transformation.qubit_mapping
                two_qubit_reduction = transformation.molecule_info['two_qubit_reduction']
                z2_symmetries = transformation.molecule_info['z2_symmetries']
                initial_state = HartreeFock(num_orbitals, num_particles, qubit_mapping,
                                            two_qubit_reduction, z2_symmetries.sq_list)
                var_form = UCCSD(num_orbitals=num_orbitals,
                                 num_particles=num_particles,
                                 initial_state=initial_state,
                                 qubit_mapping=qubit_mapping,
                                 two_qubit_reduction=two_qubit_reduction,
                                 z2_symmetries=z2_symmetries)
                vqe = VQE(var_form=var_form, quantum_instance=self._quantum_instance,
                          optimizer=L_BFGS_B())
                return vqe

        solver = CustomFactory(QuantumInstance(BasicAer.get_backend('statevector_simulator')))

        calc = AdaptVQE(self.transformation, solver)
        res = calc.solve(self.driver)
        self.assertAlmostEqual(res.electronic_energy, self.expected, places=6)

    def test_custom_excitation_pool(self):
        """ Test custom excitation pool """

        class CustomFactory(VQEUCCSDFactory):
            """A custom MES factory."""

            def get_solver(self, transformation):
                solver = super().get_solver(transformation)
                # Here, we can create essentially any custom excitation pool.
                # For testing purposes only, we simply select some hopping operator already
                # available in the variational form object.
                # pylint: disable=no-member
                custom_excitation_pool = [solver.var_form._hopping_ops[2]]
                solver.var_form.excitation_pool = custom_excitation_pool
                return solver

        solver = CustomFactory(QuantumInstance(BasicAer.get_backend('statevector_simulator')))
        calc = AdaptVQE(self.transformation, solver)
        res = calc.solve(self.driver)
        self.assertAlmostEqual(res.electronic_energy, self.expected, places=6)

    def test_vqe_adapt_check_cyclicity(self):
        """ VQEAdapt index cycle detection """
        param_list = [
            ([1, 1], True),
            ([1, 11], False),
            ([11, 1], False),
            ([1, 12], False),
            ([12, 2], False),
            ([1, 1, 1], True),
            ([1, 2, 1], False),
            ([1, 2, 2], True),
            ([1, 2, 21], False),
            ([1, 12, 2], False),
            ([11, 1, 2], False),
            ([1, 2, 1, 1], True),
            ([1, 2, 1, 2], True),
            ([1, 2, 1, 21], False),
            ([11, 2, 1, 2], False),
            ([1, 11, 1, 111], False),
            ([11, 1, 111, 1], False),
            ([1, 2, 3, 1, 2, 3], True),
            ([1, 2, 3, 4, 1, 2, 3], False),
            ([11, 2, 3, 1, 2, 3], False),
            ([1, 2, 3, 1, 2, 31], False),
            ([1, 2, 3, 4, 1, 2, 3, 4], True),
            ([11, 2, 3, 4, 1, 2, 3, 4], False),
            ([1, 2, 3, 4, 1, 2, 3, 41], False),
            ([1, 2, 3, 4, 5, 1, 2, 3, 4], False),
        ]
        for seq, is_cycle in param_list:
            with self.subTest(msg="Checking index cyclicity in:", seq=seq):
                self.assertEqual(is_cycle, AdaptVQE._check_cyclicity(seq))


if __name__ == '__main__':
    unittest.main()
