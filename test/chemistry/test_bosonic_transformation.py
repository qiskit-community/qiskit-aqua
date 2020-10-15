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

""" Test Bosonic Transformation """

import unittest

from test.chemistry import QiskitChemistryTestCase

from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.chemistry.transformations import (BosonicTransformation,
                                              BosonicTransformationType,
                                              BosonicQubitMappingType)
from qiskit.chemistry.drivers import GaussianForcesDriver
from qiskit.chemistry.algorithms.ground_state_solvers import (
    GroundStateEigensolver, NumPyMinimumEigensolverFactory, VQEUVCCSDFactory
)


class TestBosonicTransformation(QiskitChemistryTestCase):
    """Bosonic Transformation tests."""

    def setUp(self):
        super().setUp()
        logfile = self.get_resource_path('CO2_freq_B3LYP_ccpVDZ.log')
        self.driver = GaussianForcesDriver(logfile=logfile)
        self.reference_energy = 2536.4879763624226

    def _validate_input_object(self, qubit_op, num_qubits, num_paulis):
        self.assertTrue(isinstance(qubit_op, WeightedPauliOperator))
        self.assertIsNotNone(qubit_op)
        self.assertEqual(qubit_op.num_qubits, num_qubits)
        self.assertEqual(len(qubit_op.to_dict()['paulis']), num_paulis)

    def test_output(self):
        """ Test output of transformation """
        bosonic_transformation = BosonicTransformation(
            qubit_mapping=BosonicQubitMappingType.DIRECT,
            transformation_type=BosonicTransformationType.HARMONIC,
            basis_size=2,
            truncation=2)

        qubit_op, aux_ops = bosonic_transformation.transform(self.driver)
        self.assertEqual(bosonic_transformation.num_modes, 4)
        self._validate_input_object(qubit_op, num_qubits=8, num_paulis=59)
        self.assertEqual(len(aux_ops), 4)

    def test_with_numpy_minimum_eigensolver(self):
        """ Test with NumPyMinimumEigensolver """
        bosonic_transformation = BosonicTransformation(
            qubit_mapping=BosonicQubitMappingType.DIRECT,
            transformation_type=BosonicTransformationType.HARMONIC,
            basis_size=2,
            truncation=2)

        solver = NumPyMinimumEigensolverFactory(use_default_filter_criterion=True)
        gsc = GroundStateEigensolver(bosonic_transformation, solver)
        result = gsc.solve(self.driver)
        self.assertAlmostEqual(result.computed_vibronic_energies[0],
                               self.reference_energy, places=4)

    def test_vqe_uvccsd_factory(self):
        """ Test with VQE  plus UCCSD """
        bosonic_transformation = BosonicTransformation(
            qubit_mapping=BosonicQubitMappingType.DIRECT,
            transformation_type=BosonicTransformationType.HARMONIC,
            basis_size=2,
            truncation=2)

        optimizer = COBYLA(maxiter=5000)
        solver = VQEUVCCSDFactory(QuantumInstance(BasicAer.get_backend('statevector_simulator')),
                                  optimizer=optimizer)
        gsc = GroundStateEigensolver(bosonic_transformation, solver)
        result = gsc.solve(self.driver)
        self.assertAlmostEqual(result.computed_vibronic_energies[0], self.reference_energy,
                               places=1)


if __name__ == '__main__':
    unittest.main()
