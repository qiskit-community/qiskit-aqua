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

import unittest
import itertools
from test.aqua import QiskitAquaTestCase
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.extensions.standard import CzGate

import numpy as np

from qiskit.aqua.operators import X, Y, Z, I, CX, T, H, S, OpPrimitive


class TestOpPrimitive(QiskitAquaTestCase):
    """Singleton tests."""

    def test_pauli_primitives(self):
        """ from to file test """
        newop = X^Y^Z^I
        self.assertEqual(newop.primitive, Pauli(label='XYZI'))

        kpower_op = (Y^5)^(I^3)
        self.assertEqual(kpower_op.primitive, Pauli(label='YYYYYIII'))

        kpower_op2 = (Y^I)^4
        self.assertEqual(kpower_op2.primitive, Pauli(label='YIYIYIYI'))

        # Check immutability
        self.assertEqual(X.primitive, Pauli(label='X'))
        self.assertEqual(Y.primitive, Pauli(label='Y'))
        self.assertEqual(Z.primitive, Pauli(label='Z'))
        self.assertEqual(I.primitive, Pauli(label='I'))

    def test_pauli_evals(self):

        # Test eval
        self.assertEqual(Z.eval('0', '0'), 1)
        self.assertEqual(Z.eval('1', '0'), 0)
        self.assertEqual(Z.eval('0', '1'), 0)
        self.assertEqual(Z.eval('1', '1'), -1)
        self.assertEqual(X.eval('0', '0'), 0)
        self.assertEqual(X.eval('1', '0'), 1)
        self.assertEqual(X.eval('0', '1'), 1)
        self.assertEqual(X.eval('1', '1'), 0)
        self.assertEqual(Y.eval('0', '0'), 0)
        self.assertEqual(Y.eval('1', '0'), -1j)
        self.assertEqual(Y.eval('0', '1'), 1j)
        self.assertEqual(Y.eval('1', '1'), 0)

        # Check that Pauli logic eval returns same as matrix logic
        self.assertEqual(OpPrimitive(Z.to_matrix()).eval('0', '0'), 1)
        self.assertEqual(OpPrimitive(Z.to_matrix()).eval('1', '0'), 0)
        self.assertEqual(OpPrimitive(Z.to_matrix()).eval('0', '1'), 0)
        self.assertEqual(OpPrimitive(Z.to_matrix()).eval('1', '1'), -1)
        self.assertEqual(OpPrimitive(X.to_matrix()).eval('0', '0'), 0)
        self.assertEqual(OpPrimitive(X.to_matrix()).eval('1', '0'), 1)
        self.assertEqual(OpPrimitive(X.to_matrix()).eval('0', '1'), 1)
        self.assertEqual(OpPrimitive(X.to_matrix()).eval('1', '1'), 0)
        self.assertEqual(OpPrimitive(Y.to_matrix()).eval('0', '0'), 0)
        self.assertEqual(OpPrimitive(Y.to_matrix()).eval('1', '0'), -1j)
        self.assertEqual(OpPrimitive(Y.to_matrix()).eval('0', '1'), 1j)
        self.assertEqual(OpPrimitive(Y.to_matrix()).eval('1', '1'), 0)

        pauli_op = Z^I^X^Y
        mat_op = OpPrimitive(pauli_op.to_matrix())
        full_basis = list(map(''.join, itertools.product('01', repeat=pauli_op.num_qubits)))
        for bstr1, bstr2 in itertools.product(full_basis, full_basis):
            # print('{} {} {} {}'.format(bstr1, bstr2, pauli_op.eval(bstr1, bstr2), mat_op.eval(bstr1, bstr2)))
            self.assertEqual(pauli_op.eval(bstr1, bstr2), mat_op.eval(bstr1, bstr2))

    def test_circuit_primitives(self):
        hadq2 = I^H
        cz = hadq2.compose(CX).compose(hadq2)

        ref_cz_mat = OpPrimitive(CzGate()).to_matrix()
        np.testing.assert_array_almost_equal(cz.to_matrix(), ref_cz_mat)

    def test_matrix_primitives(self):
        pass

    def test_io_consistency(self):
        new_op = X^Y^I
        label = 'XYI'
        # label = new_op.primitive.to_label()
        self.assertEqual(str(new_op.primitive), label)
        np.testing.assert_array_almost_equal(new_op.primitive.to_matrix(), Operator.from_label(label).data)
        self.assertEqual(new_op.primitive, Pauli(label=label))

        x_mat = X.primitive.to_matrix()
        y_mat = Y.primitive.to_matrix()
        i_mat = np.eye(2, 2)
        np.testing.assert_array_almost_equal(new_op.primitive.to_matrix(), np.kron(np.kron(x_mat, y_mat), i_mat))

        # TODO make sure this works given endianness mayhem
        # qc = QuantumCircuit(3)
        # qc.x(2)
        # qc.y(1)
        # from qiskit import BasicAer, QuantumCircuit, execute
        # unitary = execute(qc, BasicAer.get_backend('unitary_simulator')).result().get_unitary()
        # np.testing.assert_array_almost_equal(new_op.primitive.to_matrix(), unitary)

    def test_get_primitives(self):
        my_op = (.5*(Y+H)).kron(X).compose(H^I) + OpPrimitive(Operator.from_label('+r'))
        print(my_op.to_matrix())
        print(my_op)
        print(my_op.get_primitives())
