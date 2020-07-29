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

""" Test Operator construction, including OpPrimitives and singletons. """


import unittest
from test.aqua import QiskitAquaTestCase
import itertools
import numpy as np
from ddt import ddt, data

from qiskit.circuit import QuantumCircuit, QuantumRegister, Instruction, Parameter
from qiskit.extensions.exceptions import ExtensionError
from qiskit.quantum_info.operators import Operator, Pauli
from qiskit.circuit.library import CZGate, ZGate

from qiskit.aqua.operators import (
    X, Y, Z, I, CX, T, H, PrimitiveOp, SummedOp, PauliOp, Minus, CircuitOp, MatrixOp,
    CircuitStateFn, VectorStateFn, DictStateFn, OperatorStateFn
)


# pylint: disable=invalid-name

@ddt
class TestOpConstruction(QiskitAquaTestCase):
    """Operator Construction tests."""

    def test_pauli_primitives(self):
        """ from to file test """
        newop = X ^ Y ^ Z ^ I
        self.assertEqual(newop.primitive, Pauli(label='XYZI'))

        kpower_op = (Y ^ 5) ^ (I ^ 3)
        self.assertEqual(kpower_op.primitive, Pauli(label='YYYYYIII'))

        kpower_op2 = (Y ^ I) ^ 4
        self.assertEqual(kpower_op2.primitive, Pauli(label='YIYIYIYI'))

        # Check immutability
        self.assertEqual(X.primitive, Pauli(label='X'))
        self.assertEqual(Y.primitive, Pauli(label='Y'))
        self.assertEqual(Z.primitive, Pauli(label='Z'))
        self.assertEqual(I.primitive, Pauli(label='I'))

    def test_composed_eval(self):
        """ Test eval of ComposedOp """
        self.assertAlmostEqual(Minus.eval('1'), -.5 ** .5)

    def test_evals(self):
        """ evals test """
        # pylint: disable=no-member
        # TODO: Think about eval names
        self.assertEqual(Z.eval('0').eval('0'), 1)
        self.assertEqual(Z.eval('1').eval('0'), 0)
        self.assertEqual(Z.eval('0').eval('1'), 0)
        self.assertEqual(Z.eval('1').eval('1'), -1)
        self.assertEqual(X.eval('0').eval('0'), 0)
        self.assertEqual(X.eval('1').eval('0'), 1)
        self.assertEqual(X.eval('0').eval('1'), 1)
        self.assertEqual(X.eval('1').eval('1'), 0)
        self.assertEqual(Y.eval('0').eval('0'), 0)
        self.assertEqual(Y.eval('1').eval('0'), -1j)
        self.assertEqual(Y.eval('0').eval('1'), 1j)
        self.assertEqual(Y.eval('1').eval('1'), 0)

        with self.assertRaises(ValueError):
            Y.eval('11')

        with self.assertRaises(ValueError):
            (X ^ Y).eval('1111')

        with self.assertRaises(ValueError):
            Y.eval((X ^ X).to_matrix_op())

        # Check that Pauli logic eval returns same as matrix logic
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval('0').eval('0'), 1)
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval('1').eval('0'), 0)
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval('0').eval('1'), 0)
        self.assertEqual(PrimitiveOp(Z.to_matrix()).eval('1').eval('1'), -1)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval('0').eval('0'), 0)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval('1').eval('0'), 1)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval('0').eval('1'), 1)
        self.assertEqual(PrimitiveOp(X.to_matrix()).eval('1').eval('1'), 0)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval('0').eval('0'), 0)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval('1').eval('0'), -1j)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval('0').eval('1'), 1j)
        self.assertEqual(PrimitiveOp(Y.to_matrix()).eval('1').eval('1'), 0)

        pauli_op = Z ^ I ^ X ^ Y
        mat_op = PrimitiveOp(pauli_op.to_matrix())
        full_basis = list(map(''.join, itertools.product('01', repeat=pauli_op.num_qubits)))
        for bstr1, bstr2 in itertools.product(full_basis, full_basis):
            # print('{} {} {} {}'.format(bstr1, bstr2, pauli_op.eval(bstr1, bstr2),
            # mat_op.eval(bstr1, bstr2)))
            np.testing.assert_array_almost_equal(pauli_op.eval(bstr1).eval(bstr2),
                                                 mat_op.eval(bstr1).eval(bstr2))

        gnarly_op = SummedOp([(H ^ I ^ Y).compose(X ^ X ^ Z).tensor(Z),
                              PrimitiveOp(Operator.from_label('+r0I')),
                              3 * (X ^ CX ^ T)], coeff=3 + .2j)
        gnarly_mat_op = PrimitiveOp(gnarly_op.to_matrix())
        full_basis = list(map(''.join, itertools.product('01', repeat=gnarly_op.num_qubits)))
        for bstr1, bstr2 in itertools.product(full_basis, full_basis):
            np.testing.assert_array_almost_equal(gnarly_op.eval(bstr1).eval(bstr2),
                                                 gnarly_mat_op.eval(bstr1).eval(bstr2))

    def test_circuit_construction(self):
        """ circuit construction test """
        hadq2 = H ^ I
        cz = hadq2.compose(CX).compose(hadq2)
        qc = QuantumCircuit(2)
        qc.append(cz.primitive, qargs=range(2))

        ref_cz_mat = PrimitiveOp(CZGate()).to_matrix()
        np.testing.assert_array_almost_equal(cz.to_matrix(), ref_cz_mat)

    def test_io_consistency(self):
        """ consistency test """
        new_op = X ^ Y ^ I
        label = 'XYI'
        # label = new_op.primitive.to_label()
        self.assertEqual(str(new_op.primitive), label)
        np.testing.assert_array_almost_equal(new_op.primitive.to_matrix(),
                                             Operator.from_label(label).data)
        self.assertEqual(new_op.primitive, Pauli(label=label))

        x_mat = X.primitive.to_matrix()
        y_mat = Y.primitive.to_matrix()
        i_mat = np.eye(2, 2)
        np.testing.assert_array_almost_equal(new_op.primitive.to_matrix(),
                                             np.kron(np.kron(x_mat, y_mat), i_mat))

        hi = np.kron(H.to_matrix(), I.to_matrix())
        hi2 = Operator.from_label('HI').data
        hi3 = (H ^ I).to_matrix()
        np.testing.assert_array_almost_equal(hi, hi2)
        np.testing.assert_array_almost_equal(hi2, hi3)

        xy = np.kron(X.to_matrix(), Y.to_matrix())
        xy2 = Operator.from_label('XY').data
        xy3 = (X ^ Y).to_matrix()
        np.testing.assert_array_almost_equal(xy, xy2)
        np.testing.assert_array_almost_equal(xy2, xy3)

        # Check if numpy array instantiation is the same as from Operator
        matrix_op = Operator.from_label('+r')
        np.testing.assert_array_almost_equal(PrimitiveOp(matrix_op).to_matrix(),
                                             PrimitiveOp(matrix_op.data).to_matrix())
        # Ditto list of lists
        np.testing.assert_array_almost_equal(PrimitiveOp(matrix_op.data.tolist()).to_matrix(),
                                             PrimitiveOp(matrix_op.data).to_matrix())

        # TODO make sure this works once we resolve endianness mayhem
        # qc = QuantumCircuit(3)
        # qc.x(2)
        # qc.y(1)
        # from qiskit import BasicAer, QuantumCircuit, execute
        # unitary = execute(qc, BasicAer.get_backend('unitary_simulator')).result().get_unitary()
        # np.testing.assert_array_almost_equal(new_op.primitive.to_matrix(), unitary)

    def test_to_matrix(self):
        """to matrix text """
        np.testing.assert_array_equal(X.to_matrix(), Operator.from_label('X').data)
        np.testing.assert_array_equal(Y.to_matrix(), Operator.from_label('Y').data)
        np.testing.assert_array_equal(Z.to_matrix(), Operator.from_label('Z').data)

        op1 = Y + H
        np.testing.assert_array_almost_equal(op1.to_matrix(), Y.to_matrix() + H.to_matrix())

        op2 = op1 * .5
        np.testing.assert_array_almost_equal(op2.to_matrix(), op1.to_matrix() * .5)

        op3 = (4 - .6j) * op2
        np.testing.assert_array_almost_equal(op3.to_matrix(), op2.to_matrix() * (4 - .6j))

        op4 = op3.tensor(X)
        np.testing.assert_array_almost_equal(op4.to_matrix(),
                                             np.kron(op3.to_matrix(), X.to_matrix()))

        op5 = op4.compose(H ^ I)
        np.testing.assert_array_almost_equal(op5.to_matrix(), np.dot(op4.to_matrix(),
                                                                     (H ^ I).to_matrix()))

        op6 = op5 + PrimitiveOp(Operator.from_label('+r').data)
        np.testing.assert_array_almost_equal(
            op6.to_matrix(), op5.to_matrix() + Operator.from_label('+r').data)

    def test_matrix_to_instruction(self):
        """Test MatrixOp.to_instruction yields an Instruction object."""
        matop = (H ^ 3).to_matrix_op()
        with self.subTest('assert to_instruction returns Instruction'):
            self.assertIsInstance(matop.to_instruction(), Instruction)

        matop = ((H ^ 3) + (Z ^ 3)).to_matrix_op()
        with self.subTest('matrix operator is not unitary'):
            with self.assertRaises(ExtensionError):
                matop.to_instruction()

    def test_adjoint(self):
        """ adjoint test """
        gnarly_op = 3 * (H ^ I ^ Y).compose(X ^ X ^ Z).tensor(T ^ Z) + \
            PrimitiveOp(Operator.from_label('+r0IX').data)
        np.testing.assert_array_almost_equal(np.conj(np.transpose(gnarly_op.to_matrix())),
                                             gnarly_op.adjoint().to_matrix())

    def test_primitive_strings(self):
        """ get primitives test """
        self.assertEqual(X.primitive_strings(), {'Pauli'})

        gnarly_op = 3 * (H ^ I ^ Y).compose(X ^ X ^ Z).tensor(T ^ Z) + \
            PrimitiveOp(Operator.from_label('+r0IX').data)
        self.assertEqual(gnarly_op.primitive_strings(), {'QuantumCircuit', 'Matrix'})

    def test_to_pauli_op(self):
        """ Test to_pauli_op method """
        gnarly_op = 3 * (H ^ I ^ Y).compose(X ^ X ^ Z).tensor(T ^ Z) + \
            PrimitiveOp(Operator.from_label('+r0IX').data)
        mat_op = gnarly_op.to_matrix_op()
        pauli_op = gnarly_op.to_pauli_op()
        self.assertIsInstance(pauli_op, SummedOp)
        for p in pauli_op:
            self.assertIsInstance(p, PauliOp)
        np.testing.assert_array_almost_equal(mat_op.to_matrix(), pauli_op.to_matrix())

    def test_circuit_permute(self):
        r""" Test the CircuitOp's .permute method """
        perm = range(7)[::-1]
        c_op = (((CX ^ 3) ^ X) @
                (H ^ 7) @
                (X ^ Y ^ Z ^ I ^ X ^ X ^ X) @
                (Y ^ (CX ^ 3)) @
                (X ^ Y ^ Z ^ I ^ X ^ X ^ X))
        c_op_perm = c_op.permute(perm)
        self.assertNotEqual(c_op, c_op_perm)
        c_op_id = c_op_perm.permute(perm)
        self.assertEqual(c_op, c_op_id)

    def test_summed_op_reduce(self):
        """Test SummedOp"""
        sum_op = (X ^ X * 2) + (Y ^ Y)  # type: SummedOp
        with self.subTest('SummedOp test 1'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [2, 1])

        sum_op = (X ^ X * 2) + (Y ^ Y)
        sum_op += Y ^ Y
        with self.subTest('SummedOp test 2-a'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [2, 1, 1])

        sum_op = sum_op.collapse_summands()
        with self.subTest('SummedOp test 2-b'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [2, 2])

        sum_op = (X ^ X * 2) + (Y ^ Y)
        sum_op += (Y ^ Y) + (X ^ X * 2)
        with self.subTest('SummedOp test 3-a'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY', 'YY', 'XX'])
            self.assertListEqual([op.coeff for op in sum_op], [2, 1, 1, 2])

        sum_op = sum_op.reduce()
        with self.subTest('SummedOp test 3-b'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2])

        sum_op = SummedOp([X ^ X * 2, Y ^ Y], 2)
        with self.subTest('SummedOp test 4-a'):
            self.assertEqual(sum_op.coeff, 2)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [2, 1])

        sum_op = sum_op.collapse_summands()
        with self.subTest('SummedOp test 4-b'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2])

        sum_op = SummedOp([X ^ X * 2, Y ^ Y], 2)
        sum_op += Y ^ Y
        with self.subTest('SummedOp test 5-a'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2, 1])

        sum_op = sum_op.collapse_summands()
        with self.subTest('SummedOp test 5-b'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [4, 3])

        sum_op = SummedOp([X ^ X * 2, Y ^ Y], 2)
        sum_op += (X ^ X) * 2 + (Y ^ Y)
        with self.subTest('SummedOp test 6-a'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY', 'XX', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2, 2, 1])

        sum_op = sum_op.collapse_summands()
        with self.subTest('SummedOp test 6-b'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [6, 3])

        sum_op = SummedOp([X ^ X * 2, Y ^ Y], 2)
        sum_op += sum_op
        with self.subTest('SummedOp test 7-a'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY', 'XX', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2, 4, 2])

        sum_op = sum_op.collapse_summands()
        with self.subTest('SummedOp test 7-b'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY'])
            self.assertListEqual([op.coeff for op in sum_op], [8, 4])

        sum_op = SummedOp([X ^ X * 2, Y ^ Y], 2) + SummedOp([X ^ X * 2, Z ^ Z], 3)
        with self.subTest('SummedOp test 8-a'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY', 'XX', 'ZZ'])
            self.assertListEqual([op.coeff for op in sum_op], [4, 2, 6, 3])

        sum_op = sum_op.collapse_summands()
        with self.subTest('SummedOp test 8-b'):
            self.assertEqual(sum_op.coeff, 1)
            self.assertListEqual([str(op.primitive) for op in sum_op], ['XX', 'YY', 'ZZ'])
            self.assertListEqual([op.coeff for op in sum_op], [10, 2, 3])

    def test_compose_op_of_different_dim(self):
        """
        Test if smaller operator expands to correct dim when composed with bigger operator.
        Test if compose methods of PrimitiveOp types are consistent.
        """
        # PauliOps of different dim
        xy_p = (X ^ Y)
        xyz_p = (X ^ Y ^ Z)

        pauli_op = xy_p @ xyz_p
        expected_result = (I ^ I ^ Z)
        self.assertEqual(pauli_op, expected_result)

        # MatrixOps of different dim
        xy_m = xy_p.to_matrix_op()
        xyz_m = xyz_p.to_matrix_op()

        matrix_op = xy_m @ xyz_m
        self.assertEqual(matrix_op, expected_result.to_matrix_op())

        # CircuitOps of different dim
        xy_c = xy_p.to_circuit_op()
        xyz_c = xyz_p.to_circuit_op()

        circuit_op = xy_c @ xyz_c

        self.assertTrue(np.array_equal(pauli_op.to_matrix(), matrix_op.to_matrix()))
        self.assertTrue(np.allclose(pauli_op.to_matrix(), circuit_op.to_matrix(), rtol=1e-14))
        self.assertTrue(np.allclose(matrix_op.to_matrix(), circuit_op.to_matrix(), rtol=1e-14))

    def test_permute_on_primitive_op(self):
        """ Test if permute methods of PrimitiveOps are consistent and work correctly. """
        indices = [1, 2, 4]

        # PauliOp
        pauli_op = (X ^ Y ^ Z)
        permuted_pauli_op = pauli_op.permute(indices)
        expected_pauli_op = (X ^ I ^ Y ^ Z ^ I)

        self.assertEqual(permuted_pauli_op, expected_pauli_op)

        # CircuitOp
        circuit_op = pauli_op.to_circuit_op()
        permuted_circuit_op = circuit_op.permute(indices)
        expected_circuit_op = expected_pauli_op.to_circuit_op()

        self.assertEqual(permuted_circuit_op.primitive.__str__(),
                         expected_circuit_op.primitive.__str__())

    def test_expand_on_state_fn(self):
        """ Tests num_qubits on the original instance and expanded instance of StateFn """
        num_qubits = 3
        add_qubits = 2

        # case CircuitStateFn, with primitive QuantumCircuit
        qc2 = QuantumCircuit(num_qubits)
        qc2.cx(0, 1)

        cfn = CircuitStateFn(qc2, is_measurement=True)

        cfn_exp = cfn.expand(add_qubits)
        self.assertEqual(cfn_exp.num_qubits, add_qubits + num_qubits)

        # case OperatorStateFn, with OperatorBase primitive, in our case CircuitStateFn
        osfn = OperatorStateFn(cfn)
        osfn_exp = osfn.expand(add_qubits)

        self.assertEqual(osfn_exp.num_qubits, add_qubits + num_qubits)

        # case DictStateFn
        dsfn = DictStateFn('1'*num_qubits, is_measurement=True)
        self.assertEqual(dsfn.num_qubits, num_qubits)

        dsfn_exp = dsfn.expand(add_qubits)
        self.assertEqual(dsfn_exp.num_qubits, num_qubits + add_qubits)

        # case VectorStateFn
        vsfn = VectorStateFn(np.ones(2**num_qubits, dtype=complex))
        self.assertEqual(vsfn.num_qubits, num_qubits)

        vsfn_exp = vsfn.expand(add_qubits)
        self.assertEqual(vsfn_exp.num_qubits, num_qubits + add_qubits)

    def test_summed_op_equals(self):
        """Test corner cases of SummedOp's equals function."""
        with self.subTest('multiplicative factor'):
            self.assertEqual(2 * X, X + X)

        with self.subTest('commutative'):
            self.assertEqual(X + Z, Z + X)

        with self.subTest('circuit and paulis'):
            z = CircuitOp(ZGate())
            self.assertEqual(Z + z, z + Z)

        with self.subTest('matrix op and paulis'):
            z = MatrixOp([[1, 0], [0, -1]])
            self.assertEqual(Z + z, z + Z)

        with self.subTest('matrix multiplicative'):
            z = MatrixOp([[1, 0], [0, -1]])
            self.assertEqual(2 * z, z + z)

        with self.subTest('parameter coefficients'):
            expr = Parameter('theta')
            z = MatrixOp([[1, 0], [0, -1]])
            self.assertEqual(expr * z, expr * z)

        with self.subTest('different coefficient types'):
            expr = Parameter('theta')
            z = MatrixOp([[1, 0], [0, -1]])
            self.assertNotEqual(expr * z, 2 * z)

        with self.subTest('additions aggregation'):
            z = MatrixOp([[1, 0], [0, -1]])
            a = z + z + Z
            b = 2 * z + Z
            c = z + Z + z
            self.assertEqual(a, b)
            self.assertEqual(b, c)
            self.assertEqual(a, c)

    def test_circuit_compose_register_independent(self):
        """Test that CircuitOp uses combines circuits independent of the register.

        I.e. that is uses ``QuantumCircuit.compose`` over ``combine`` or ``extend``.
        """
        op = Z ^ 2
        qr = QuantumRegister(2, 'my_qr')
        circuit = QuantumCircuit(qr)
        composed = op.compose(CircuitOp(circuit))

        self.assertEqual(composed.num_qubits, 2)

    @data(Z, CircuitOp(ZGate()), MatrixOp([[1, 0], [0, -1]]))
    def test_op_hashing(self, op):
        """Regression test against faulty set comparison.

        Set comparisons rely on a hash table which requires identical objects to have identical
        hashes. Thus, the PrimitiveOp.__hash__ should support this requirement.
        """
        self.assertEqual(set([2 * op]), set([2 * op]))


if __name__ == '__main__':
    unittest.main()
