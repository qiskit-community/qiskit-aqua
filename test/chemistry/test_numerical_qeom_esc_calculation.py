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

""" Test NumericalqEOM excited states calculation """

import unittest
from test.chemistry import QiskitChemistryTestCase

from qiskit import BasicAer
from qiskit.aqua import aqua_globals, QuantumInstance

from qiskit.aqua.algorithms import NumPyMinimumEigensolver, NumPyEigensolver
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.qubit_transformations import FermionicTransformation
from qiskit.chemistry.qubit_transformations.fermionic_transformation import QubitMappingType
from qiskit.chemistry.ground_state_calculation import MinimumEigensolverGroundStateCalculation
from qiskit.chemistry.ground_state_calculation import VQEUCCSDFactory
from qiskit.chemistry.excited_states_calculation import NumericalQEOMExcitedStatesCalculation
from qiskit.chemistry.excited_states_calculation import NumPyEigensolverFactory
from qiskit.chemistry.excited_states_calculation import EigenSolverExcitedStatesCalculation

class TestNumericalQEOMESCCalculation(QiskitChemistryTestCase):
    """ Test NumericalqEOM excited states calculation """

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 8
        try:
            self.driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.75',
                                      unit=UnitsType.ANGSTROM,
                                      charge=0,
                                      spin=0,
                                      basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')

        self.reference_energies = [-1.8427016, -1.8427016 + 0.5943372, -1.8427016 + 0.95788352,
                                   -1.8427016 + 1.5969296]
        self.transformation = FermionicTransformation(qubit_mapping=QubitMappingType.JORDAN_WIGNER)
        solver = NumPyEigensolver()
        self.ref = solver
        self.quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                     seed_transpiler=90, seed_simulator=12)


    def test_numpy_mes(self):

        solver = NumPyMinimumEigensolver()
        gsc = MinimumEigensolverGroundStateCalculation(self.transformation, solver)
        esc = NumericalQEOMExcitedStatesCalculation(gsc, 'sd')
        results = esc.compute_excitedstates(self.driver)

        for idx in range(len(self.reference_energies)):
            self.assertAlmostEqual(results.computed_energies[idx], self.reference_energies[idx],
                                   places=4)

    def test_vqe_mes(self):

        solver = VQEUCCSDFactory(self.quantum_instance)
        gsc = MinimumEigensolverGroundStateCalculation(self.transformation, solver)
        esc = NumericalQEOMExcitedStatesCalculation(gsc, 'sd')
        results = esc.compute_excitedstates(self.driver)

        for idx in range(len(self.reference_energies)):
            self.assertAlmostEqual(results.computed_energies[idx], self.reference_energies[idx],
                                   places=4)

    # def test_numpy_factory(self):
    #
    #     solver = NumPyEigensolverFactory()
    #     esc = EigenSolverExcitedStatesCalculation(self.transformation, solver)
    #     results = esc.compute_excitedstates(self.driver)
    #
    #     print(results)
    #     for idx in range(len(self.reference_energies)):
    #         self.assertAlmostEqual(results.computed_energies[idx], self.reference_energies[idx],
    #                                places=4)





if __name__ == '__main__':
    unittest.main()