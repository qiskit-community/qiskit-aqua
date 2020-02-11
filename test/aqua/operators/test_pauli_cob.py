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
        singles = [I, X, Y, Z]
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

    def test_pauli_cob_multiqubit(self):
        multis = [Y^X, I^Z^Y^X, Y^Y^I^X^Z^Y^I]
        for pauli in multis:
            print(pauli)
            converter = PauliChangeOfBasis()
            inst, dest = converter.get_cob_circuit(pauli.primitive)
            cob = converter.convert(pauli)
            qc = QuantumCircuit(pauli.num_qubits)
            qc.append(inst.primitive, range(pauli.num_qubits))
            qc = qc.decompose()
            print(qc.draw())
            print(pauli.to_matrix())
            print(np.round(inst.adjoint().to_matrix() @ cob.to_matrix()))
            np.testing.assert_array_almost_equal(pauli.to_matrix(),
                                                 inst.adjoint().to_matrix() @ dest.to_matrix() @ inst.to_matrix())
            np.testing.assert_array_almost_equal(pauli.to_matrix(), inst.adjoint().to_matrix() @ cob.to_matrix())
            np.testing.assert_array_almost_equal(inst.compose(pauli).compose(inst.adjoint()).to_matrix(),
                                                 dest.to_matrix())

        multis = [Y^X, I^Z^Y^X, Y^Y^I^X^Z^Y^I]
        for pauli in multis:
            print(pauli)
            converter = PauliChangeOfBasis()
            inst, dest = converter.get_cob_circuit(pauli.primitive)
            cob = converter.convert(pauli)
            qc = QuantumCircuit(pauli.num_qubits)
            qc.append(inst.primitive, range(pauli.num_qubits))
            qc = qc.decompose()
            print(qc.draw())
            print(pauli.to_matrix())
            print(np.round(inst.adjoint().to_matrix() @ cob.to_matrix()))
            np.testing.assert_array_almost_equal(pauli.to_matrix(),
                                                 inst.adjoint().to_matrix() @ dest.to_matrix() @ inst.to_matrix())
            np.testing.assert_array_almost_equal(pauli.to_matrix(), inst.adjoint().to_matrix() @ cob.to_matrix())
            np.testing.assert_array_almost_equal(inst.compose(pauli).compose(inst.adjoint()).to_matrix(),
                                                 dest.to_matrix())

        # np.testing.assert_array_almost_equal(pauli.to_matrix() @ cob.compose(inst.adjoint()).adjoint().to_matrix())
        # print(np.round(pauli.to_matrix() @ cob.compose(inst.adjoint()).to_matrix()))

        # print((H).compose(Z).compose(H).to_matrix())
        # print(pauli.to_matrix())
        # print(np.round(cob.compose(inst.adjoint()).to_matrix()))
        # print(inst.to_matrix())
        # print(dest.to_matrix())
        # # print(OpPrimitive(dest).to_matrix())
        # print(np.round(cob.compose(inst).to_matrix()))
        # qc = QuantumCircuit(2)
        # qc.append(inst.primitive, range(2))
        # qc = qc.decompose()
        # print(qc.draw())
