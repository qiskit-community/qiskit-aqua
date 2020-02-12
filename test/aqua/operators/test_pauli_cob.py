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

""" Test OpSum """

from test.aqua import QiskitAquaTestCase

import numpy as np
import itertools

from qiskit.aqua.operators import X, Y, Z, I, CX, T, H, S, OpPrimitive, OpSum
from qiskit.aqua.operators.converters import PauliChangeOfBasis
from qiskit import QuantumCircuit


class TestPauliCoB(QiskitAquaTestCase):
    """Pauli Change of Basis Converter tests."""

    def test_pauli_cob_singles(self):
        """ from to file test """
        singles = [X, Y, Z]
        dests = [None, Y]
        for pauli, dest in itertools.product(singles, dests):
            # print(pauli)
            converter = PauliChangeOfBasis(destination_basis=dest)
            inst, dest = converter.get_cob_circuit(pauli.primitive)
            cob = converter.convert(pauli)
            np.testing.assert_array_almost_equal(pauli.to_matrix(),
                                                 inst.adjoint().to_matrix() @ dest.to_matrix() @ inst.to_matrix())
            np.testing.assert_array_almost_equal(pauli.to_matrix(), inst.adjoint().to_matrix() @ cob.to_matrix())
            np.testing.assert_array_almost_equal(inst.compose(pauli).compose(inst.adjoint()).to_matrix(),
                                                 dest.to_matrix())

    def test_pauli_cob_two_qubit(self):
        multis = [Y^X, Z^Y, I^Z, Z^I, X^X, I^X]
        for pauli, dest in itertools.product(multis, reversed(multis)):
            converter = PauliChangeOfBasis(destination_basis=dest)
            inst, dest = converter.get_cob_circuit(pauli.primitive)
            cob = converter.convert(pauli)
            np.testing.assert_array_almost_equal(pauli.to_matrix(),
                                                 inst.adjoint().to_matrix() @ dest.to_matrix() @ inst.to_matrix())
            np.testing.assert_array_almost_equal(pauli.to_matrix(), inst.adjoint().to_matrix() @ cob.to_matrix())
            np.testing.assert_array_almost_equal(inst.compose(pauli).compose(inst.adjoint()).to_matrix(),
                                                 dest.to_matrix())

    def test_pauli_cob_multiqubit(self):
        # Helpful prints for debugging commented out below.
        multis = [Y^X^I^I, I^Z^Y^X, X^Y^I^Z, I^I^I^X, X^X^X^X]
        for pauli, dest in itertools.product(multis, reversed(multis)):
            # print(pauli)
            # print(dest)
            converter = PauliChangeOfBasis(destination_basis=dest)
            inst, dest = converter.get_cob_circuit(pauli.primitive)
            cob = converter.convert(pauli)
            # qc = QuantumCircuit(pauli.num_qubits)
            # qc.append(inst.primitive, range(pauli.num_qubits))
            # qc = qc.decompose()
            # print(qc.draw())
            # print(pauli.to_matrix())
            # print(np.round(inst.adjoint().to_matrix() @ cob.to_matrix()))
            np.testing.assert_array_almost_equal(pauli.to_matrix(),
                                                 inst.adjoint().to_matrix() @ dest.to_matrix() @ inst.to_matrix())
            np.testing.assert_array_almost_equal(pauli.to_matrix(), inst.adjoint().to_matrix() @ cob.to_matrix())
            np.testing.assert_array_almost_equal(inst.compose(pauli).compose(inst.adjoint()).to_matrix(),
                                                 dest.to_matrix())
