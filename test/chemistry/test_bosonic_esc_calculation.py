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
from qiskit.chemistry.drivers import BaseDriver
from qiskit.aqua import aqua_globals, QuantumInstance

from qiskit.aqua.algorithms import NumPyMinimumEigensolver, NumPyEigensolver
from qiskit.chemistry import WatsonHamiltonian
from qiskit.chemistry.drivers import GaussianForcesDriver
from qiskit.chemistry.ground_state_calculation import (MinimumEigensolverGroundStateCalculation,
                                                       VQEUCCSDFactory)
from qiskit.chemistry.excited_states_calculation import NumericalQEOMExcitedStatesCalculation
from qiskit.chemistry.qubit_transformations import (BosonicTransformation,
                                                    BosonicTransformationType,
                                                    BosonicQubitMappingType)
from qiskit.chemistry.ground_state_calculation import NumPyMinimumEigensolverFactory
from qiskit.chemistry.excited_states_calculation import EigenSolverExcitedStatesCalculation
from qiskit.chemistry.excited_states_calculation import NumPyEigensolverFactory

class DumBosonicDriver(BaseDriver):

    def __init__(self):

        modes = [[605.3643675, 1, 1], [-605.3643675, -1, -1], [340.5950575, 2, 2],
                 [-340.5950575, -2, -2], [-89.09086530649508, 2, 1, 1],
                 [-15.590557244410897, 2, 2, 2], [1.6512647916666667, 1, 1, 1, 1],
                 [5.03965375, 2, 2, 1, 1], [0.43840625000000005, 2, 2, 2, 2]]
        self._watson = WatsonHamiltonian(modes, 2)

    def run(self):
        return self._watson


class TestBosonicESCCalculation(QiskitChemistryTestCase):
    """ Test NumericalqEOM excited states calculation """

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 8
        self.reference_energies = [1889.95738428, 3294.21806197, 4287.26821341, 5819.76975784]

        self.driver = DumBosonicDriver()

        self.transformation = BosonicTransformation(
            qubit_mapping=BosonicQubitMappingType.DIRECT,
            transformation_type=BosonicTransformationType.HARMONIC,
            basis_size=2,
            truncation=2)


    def test_numpy_mes(self):
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
        gsc = MinimumEigensolverGroundStateCalculation(self.transformation, solver)
        esc = NumericalQEOMExcitedStatesCalculation(gsc, 'sd')
        results = esc.compute_excitedstates(self.driver)
        print(results.computed_vibronic_energies)
        print(results.raw_result)

        for idx in range(len(self.reference_energies)):
            self.assertAlmostEqual(results.computed_vibronic_energies[idx], self.reference_energies[idx],
                                   places=4)


    def test_numpy_factory(self):

        solver = NumPyEigensolverFactory()
        esc = EigenSolverExcitedStatesCalculation(self.transformation, solver)
        results = esc.compute_excitedstates(self.driver)

        for idx in range(len(self.reference_energies)):
            self.assertAlmostEqual(results.computed_vibronic_energies[idx], self.reference_energies[idx],
                                   places=4)





if __name__ == '__main__':
    unittest.main()