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

""" Test VQE UCCSD MinimumEigensovler Factory """

import unittest

from test.chemistry import QiskitChemistryTestCase

from qiskit import BasicAer
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import AerPauliExpectation
from qiskit.aqua.components.optimizers import COBYLA
from qiskit.chemistry import QiskitChemistryError
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry.transformations import FermionicTransformation
from qiskit.chemistry.transformations.fermionic_transformation import FermionicQubitMappingType
from qiskit.chemistry.algorithms.ground_state_solvers.minimum_eigensolver_factories import \
    VQEUCCSDFactory


class TestVQEUCCSDMESFactory(QiskitChemistryTestCase):
    """ Test VQE UCCSD MinimumEigensovler Factory """

    def setUp(self):
        super().setUp()
        try:
            self.driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                                      unit=UnitsType.ANGSTROM,
                                      charge=0,
                                      spin=0,
                                      basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')

        self.reference_energy = -1.137306

        self.transformation = \
            FermionicTransformation(qubit_mapping=FermionicQubitMappingType.JORDAN_WIGNER)

        self.seed = 50
        self.quantum_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                shots=1,
                                                seed_simulator=self.seed,
                                                seed_transpiler=self.seed)

        self._vqe_uccsd_factory = VQEUCCSDFactory(self.quantum_instance)

    def test_setters_getters(self):
        """ Test Getter/Setter """

        # quantum instance
        self.assertEqual(self._vqe_uccsd_factory.quantum_instance, self.quantum_instance)
        self._vqe_uccsd_factory.quantum_instance = None
        self.assertEqual(self._vqe_uccsd_factory.quantum_instance, None)

        # optimizer
        self.assertEqual(self._vqe_uccsd_factory.optimizer, None)
        optimizer = COBYLA()
        self._vqe_uccsd_factory.optimizer = optimizer
        self.assertEqual(self._vqe_uccsd_factory.optimizer, optimizer)

        # initial point
        self.assertEqual(self._vqe_uccsd_factory.initial_point, None)
        initial_point = [1, 2, 3]
        self._vqe_uccsd_factory.initial_point = initial_point
        self.assertEqual(self._vqe_uccsd_factory.initial_point, initial_point)

        # expectation
        self.assertEqual(self._vqe_uccsd_factory.expectation, None)
        expectation = AerPauliExpectation()
        self._vqe_uccsd_factory.expectation = expectation
        self.assertEqual(self._vqe_uccsd_factory.expectation, expectation)

        # include_custom
        self.assertEqual(self._vqe_uccsd_factory.include_custom, False)
        self._vqe_uccsd_factory.include_custom = True
        self.assertEqual(self._vqe_uccsd_factory.include_custom, True)

        # method_singles
        self.assertEqual(self._vqe_uccsd_factory.method_singles, 'both')
        self._vqe_uccsd_factory.method_singles = 'alpha'
        self.assertEqual(self._vqe_uccsd_factory.method_singles, 'alpha')

        # method_doubles
        self.assertEqual(self._vqe_uccsd_factory.method_doubles, 'ucc')
        self._vqe_uccsd_factory.method_doubles = 'succ'
        self.assertEqual(self._vqe_uccsd_factory.method_doubles, 'succ')

        # excitation_type
        self.assertEqual(self._vqe_uccsd_factory.excitation_type, 'sd')
        self._vqe_uccsd_factory.excitation_type = 's'
        self.assertEqual(self._vqe_uccsd_factory.excitation_type, 's')

        # same_spin_doubles
        self.assertEqual(self._vqe_uccsd_factory.same_spin_doubles, True)
        self._vqe_uccsd_factory.same_spin_doubles = False
        self.assertEqual(self._vqe_uccsd_factory.same_spin_doubles, False)


if __name__ == '__main__':
    unittest.main()
