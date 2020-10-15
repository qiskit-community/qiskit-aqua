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
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.chemistry import WatsonHamiltonian
from qiskit.chemistry.drivers import BaseDriver
from qiskit.chemistry.algorithms.ground_state_solvers import (
    GroundStateEigensolver, NumPyMinimumEigensolverFactory,
    VQEUVCCSDFactory
)
from qiskit.chemistry.algorithms.excited_states_solvers import (
    QEOM, ExcitedStatesEigensolver, NumPyEigensolverFactory
)
from qiskit.chemistry.transformations import (BosonicTransformation,
                                              BosonicTransformationType,
                                              BosonicQubitMappingType)


class _DummyBosonicDriver(BaseDriver):

    def __init__(self):
        super().__init__()
        modes = [[605.3643675, 1, 1], [-605.3643675, -1, -1], [340.5950575, 2, 2],
                 [-340.5950575, -2, -2], [-89.09086530649508, 2, 1, 1],
                 [-15.590557244410897, 2, 2, 2], [1.6512647916666667, 1, 1, 1, 1],
                 [5.03965375, 2, 2, 1, 1], [0.43840625000000005, 2, 2, 2, 2]]
        self._watson = WatsonHamiltonian(modes, 2)

    def run(self):
        """ Run dummy driver to return test watson hamiltonian """
        return self._watson


class TestBosonicESCCalculation(QiskitChemistryTestCase):
    """ Test Numerical QEOM excited states calculation """

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 8
        self.reference_energies = [1889.95738428, 3294.21806197, 4287.26821341, 5819.76975784]

        self.driver = _DummyBosonicDriver()

        self.transformation = BosonicTransformation(
            qubit_mapping=BosonicQubitMappingType.DIRECT,
            transformation_type=BosonicTransformationType.HARMONIC,
            basis_size=2,
            truncation=2)

    def test_numpy_mes(self):
        """ Test with NumPyMinimumEigensolver """
        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
        gsc = GroundStateEigensolver(self.transformation, solver)
        esc = QEOM(gsc, 'sd')
        results = esc.solve(self.driver)

        for idx in range(len(self.reference_energies)):
            self.assertAlmostEqual(results.computed_vibronic_energies[idx],
                                   self.reference_energies[idx],
                                   places=4)

    def test_numpy_factory(self):
        """ Test with NumPyEigensolver """
        solver = NumPyEigensolverFactory(use_default_filter_criterion=True)
        esc = ExcitedStatesEigensolver(self.transformation, solver)
        results = esc.solve(self.driver)

        for idx in range(len(self.reference_energies)):
            self.assertAlmostEqual(results.computed_vibronic_energies[idx],
                                   self.reference_energies[idx],
                                   places=4)

    def test_vqe_uvccsd_factory(self):
        """ Test with VQE plus UVCCSD """
        optimizer = COBYLA(maxiter=5000)
        solver = VQEUVCCSDFactory(QuantumInstance(BasicAer.get_backend('statevector_simulator')),
                                  optimizer=optimizer)
        gsc = GroundStateEigensolver(self.transformation, solver)
        esc = QEOM(gsc, 'sd')
        results = esc.solve(self.driver)
        for idx in range(len(self.reference_energies)):
            self.assertAlmostEqual(results.computed_vibronic_energies[idx],
                                   self.reference_energies[idx],
                                   places=1)


if __name__ == '__main__':
    unittest.main()
