# -*- coding: utf-8 -*-

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

""" Test IQPE """

import unittest
from test.chemistry import QiskitChemistryTestCase
import numpy as np
from ddt import ddt, idata, unpack
import qiskit
from qiskit.aqua.utils import decimal_to_binary
from qiskit.aqua import QuantumInstance
from qiskit.aqua.algorithms import IQPE, ClassicalMinimumEigensolver
from qiskit.aqua.operators import Z2Symmetries
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator, QiskitChemistryError
from qiskit.chemistry.components.initial_states import HartreeFock


@ddt
class TestIQPE(QiskitChemistryTestCase):
    """IQPE tests."""

    @idata([
        [0.5],
        [0.735],
        [1],
    ])
    @unpack
    def test_iqpe(self, distance):
        """ iqpe test """
        self.log.debug('Testing End-to-End with IQPE on H2 with '
                       'inter-atomic distance %s.', distance)
        try:
            driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 {}'.format(distance),
                                 unit=UnitsType.ANGSTROM,
                                 charge=0,
                                 spin=0,
                                 basis='sto3g')
        except QiskitChemistryError:
            self.skipTest('PYSCF driver does not appear to be installed')
        molecule = driver.run()
        qubit_mapping = 'parity'
        fer_op = FermionicOperator(h1=molecule.one_body_integrals,
                                   h2=molecule.two_body_integrals)
        qubit_op = fer_op.mapping(map_type=qubit_mapping, threshold=1e-10)
        qubit_op = Z2Symmetries.two_qubit_reduction(qubit_op, 2)

        exact_eigensolver = ClassicalMinimumEigensolver(qubit_op)
        results = exact_eigensolver.run()
        reference_energy = results.eigenvalue.real
        self.log.debug('The exact ground state energy is: %s', results.eigenvalue.real)

        num_particles = molecule.num_alpha + molecule.num_beta
        two_qubit_reduction = True
        num_orbitals = qubit_op.num_qubits + (2 if two_qubit_reduction else 0)

        num_time_slices = 1
        num_iterations = 6
        state_in = HartreeFock(qubit_op.num_qubits, num_orbitals,
                               num_particles, qubit_mapping, two_qubit_reduction)
        iqpe = IQPE(qubit_op, state_in, num_time_slices, num_iterations,
                    expansion_mode='suzuki', expansion_order=2,
                    shallow_circuit_concat=True)
        backend = qiskit.BasicAer.get_backend('qasm_simulator')
        quantum_instance = QuantumInstance(backend, shots=100)

        result = iqpe.run(quantum_instance)

        self.log.debug('top result str label:     %s', result['top_measurement_label'])
        self.log.debug('top result in decimal:    %s', result['top_measurement_decimal'])
        self.log.debug('stretch:                  %s', result['stretch'])
        self.log.debug('translation:              %s', result['translation'])
        self.log.debug('final energy from QPE:    %s', result['energy'])
        self.log.debug('reference energy:         %s', reference_energy)
        self.log.debug('ref energy (transformed): %s',
                       (reference_energy + result['translation']) * result['stretch'])
        self.log.debug('ref binary str label:     %s', decimal_to_binary(
            (reference_energy + result['translation']) * result['stretch'],
            max_num_digits=num_iterations + 3,
            fractional_part_only=True
        ))

        np.testing.assert_approx_equal(result['energy'], reference_energy, significant=2)


if __name__ == '__main__':
    unittest.main()
