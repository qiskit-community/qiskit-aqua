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

""" Test SummedPauliOp """

import unittest
from test.aqua import QiskitAquaTestCase

import numpy as np
from scipy.sparse import csr_matrix

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.aqua.operators import (DictStateFn, I, SummedOp, SummedPauliOp, X,
                                   Y, Z, Zero)


class TestSummedPauliOp(QiskitAquaTestCase):
    """SummedPauliOp tests."""

    def test_construct(self):
        """ constructor test """
        sparse_pauli = SparsePauliOp(Pauli(label="XYZX"), coeffs=[2.0])
        coeff = 3.0
        summed_pauli = SummedPauliOp(sparse_pauli, coeff=coeff)
        self.assertIsInstance(summed_pauli, SummedPauliOp)
        self.assertEqual(summed_pauli.primitive, sparse_pauli)
        self.assertEqual(summed_pauli.coeff, coeff)
        self.assertEqual(summed_pauli.num_qubits, 4)

    def test_add(self):
        """ add test """
        summed_pauli = 3 * X + Y
        self.assertIsInstance(summed_pauli, SummedPauliOp)

        expected = SummedPauliOp(
            3.0 * SparsePauliOp(Pauli(label="X")) + SparsePauliOp(Pauli(label="Y"))
        )

        self.assertEqual(summed_pauli, expected)

    def test_adjoint(self):
        """ adjoint test """
        summed_pauli = SummedPauliOp(
            SparsePauliOp(Pauli(label="XYZX"), coeffs=[2]), coeff=3
        )
        expected = SummedPauliOp(SparsePauliOp(Pauli(label="XYZX")), coeff=-6)

        self.assertEqual(summed_pauli.adjoint(), expected)

        summed_pauli = SummedPauliOp(
            SparsePauliOp(Pauli(label="XYZY"), coeffs=[2]), coeff=3j
        )
        expected = SummedPauliOp(SparsePauliOp(Pauli(label="XYZY")), coeff=-6j)
        self.assertEqual(summed_pauli.adjoint(), expected)

    def test_equals(self):
        """ equality test """

        self.assertNotEqual((X ^ X) + (Y ^ Y), X + Y)
        self.assertEqual((X ^ X) + (Y ^ Y), (Y ^ Y) + (X ^ X))

        theta = ParameterVector("theta", 2)
        summed_pauli0 = theta[0] * (X + Z)
        summed_pauli1 = theta[1] * (X + Z)
        expected = SummedPauliOp(
            SparsePauliOp(Pauli(label="X")) + SparsePauliOp(Pauli(label="Z")),
            coeff=theta[0],
        )
        self.assertEqual(summed_pauli0, expected)
        self.assertNotEqual(summed_pauli1, expected)

    def test_tensor(self):
        """ Test for tensor operation """
        summed_pauli = ((I - Z) ^ (I - Z)) + ((X - Y) ^ (X + Y))
        expected = (
            (I ^ I)
            - (I ^ Z)
            - (Z ^ I)
            + (Z ^ Z)
            + (X ^ X)
            + (X ^ Y)
            - (Y ^ X)
            - (Y ^ Y)
        )
        self.assertEqual(summed_pauli, expected)

    def test_permute(self):
        """ permute test """
        summed_pauli = SummedPauliOp(SparsePauliOp((X ^ Y ^ Z).primitive))
        expected = SummedPauliOp(SparsePauliOp((X ^ I ^ Y ^ Z ^ I).primitive))

        self.assertEqual(summed_pauli.permute([1, 2, 4]), expected)

    def test_compose(self):
        """ compose test """
        target = (X + Z) @ (Y + Z)
        expected = 1j * Z - 1j * Y - 1j * X + I
        self.assertEqual(target, expected)

    def test_to_matrix(self):
        """ test for to_matrix method """
        target = (Z + Y).to_matrix()
        expected = np.array([[1.0, -1j], [1j, -1]])
        np.testing.assert_array_equal(target, expected)

    def test_str(self):
        """ str test """
        target = str(3.0 * (X + 2.0 * Y))
        expected = "SummedPauliOp([\n1.0 * X,\n2.0 * Y,\n]) * 3.0"
        self.assertEqual(target, expected)

    def test_eval(self):
        """ eval test """
        target0 = (2 * (X ^ Y ^ Z) + 3 * (X ^ X ^ Z)).eval("000")
        target1 = (2 * (X ^ Y ^ Z) + 3 * (X ^ X ^ Z)).eval(Zero ^ 3)
        expected = DictStateFn({"011": (2 + 3j)})
        self.assertEqual(target0, expected)
        self.assertEqual(target1, expected)

    def test_exp_i(self):
        """ exp_i test """
        # TODO: add tests when special methods are added
        pass

    def test_to_instruction(self):
        """ test for to_instruction """
        target = ((X + Z) / np.sqrt(2)).to_instruction()
        qc = QuantumCircuit(1)
        qc.u3(np.pi / 2, 0, np.pi, 0)
        self.assertEqual(target.definition, qc)

    def test_to_pauli_op(self):
        """ test to_pauli_op method """
        target = X + Y
        self.assertIsInstance(target, SummedPauliOp)
        expected = SummedOp([X, Y])
        self.assertEqual(target.to_pauli_op(), expected)

    def test_getitem(self):
        """ test getitem method """
        target = X + Z
        self.assertEqual(target[0], X.to_summed_pauli_op())
        self.assertEqual(target[1], Z.to_summed_pauli_op())

    def test_len(self):
        """ test len """
        target = X + Y + Z
        self.assertEqual(len(target), 3)

    def test_reduce(self):
        """ test reduce """
        target = X + X + Z
        self.assertEqual(len(target.reduce()), 2)

    def test_to_spmatrix(self):
        """ test to_spmatrix """
        target = X + Y
        expected = csr_matrix([[0, 1 - 1j], [1 + 1j, 0]])
        self.assertEqual((target.to_spmatrix() - expected).nnz, 0)

    def test_oplist(self):
        """ test oplist """
        target = X + Y + Z
        self.assertEqual(target.oplist, [X, Y, Z])


if __name__ == "__main__":
    unittest.main()
