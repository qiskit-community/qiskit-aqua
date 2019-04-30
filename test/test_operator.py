# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import unittest
import copy
import itertools
import os

from qiskit import BasicAer
import numpy as np
from qiskit.quantum_info import Pauli
from qiskit.transpiler import PassManager

from test.common import QiskitAquaTestCase
from qiskit.aqua import Operator
from qiskit.aqua.components.variational_forms import RYRZ


class TestOperator(QiskitAquaTestCase):
    """Operator tests."""

    def setUp(self):
        super().setUp()
        np.random.seed(0)

        self.num_qubits = 3
        m_size = np.power(2, self.num_qubits)
        matrix = np.random.rand(m_size, m_size)
        self.qubitOp = Operator(matrix=matrix)

    def test_real_eval(self):
        depth = 1
        var_form = RYRZ(self.qubitOp.num_qubits, depth)
        circuit = var_form.construct_circuit(np.array(np.random.randn(var_form.num_parameters)))
        # self.qubitOp.coloring = None
        run_config_ref = {'shots': 1}
        run_config = {'shots': 10000}
        reference = self.qubitOp.eval('matrix', circuit, BasicAer.get_backend('statevector_simulator'), run_config=run_config_ref)[0]
        reference = reference.real
        backend = BasicAer.get_backend('qasm_simulator')
        paulis_mode = self.qubitOp.eval('paulis', circuit, backend, run_config=run_config)
        grouped_paulis_mode = self.qubitOp.eval('grouped_paulis', circuit, backend, run_config=run_config)

        paulis_mode_p_3sigma = paulis_mode[0] + 3 * paulis_mode[1]
        paulis_mode_m_3sigma = paulis_mode[0] - 3 * paulis_mode[1]

        grouped_paulis_mode_p_3sigma = grouped_paulis_mode[0] + 3 * grouped_paulis_mode[1]
        grouped_paulis_mode_m_3sigma = grouped_paulis_mode[0] - 3 * grouped_paulis_mode[1]
        self.assertLessEqual(reference, paulis_mode_p_3sigma.real)
        self.assertGreaterEqual(reference, paulis_mode_m_3sigma.real)
        self.assertLessEqual(reference, grouped_paulis_mode_p_3sigma.real)
        self.assertGreaterEqual(reference, grouped_paulis_mode_m_3sigma.real)

        run_config = {'shots': 10000}
        compile_config = {'pass_manager': PassManager()}
        paulis_mode = self.qubitOp.eval('paulis', circuit, backend,
                                        run_config=run_config, compile_config=compile_config)
        grouped_paulis_mode = self.qubitOp.eval('grouped_paulis', circuit, backend,
                                                run_config=run_config, compile_config=compile_config)

        paulis_mode_p_3sigma = paulis_mode[0] + 3 * paulis_mode[1]
        paulis_mode_m_3sigma = paulis_mode[0] - 3 * paulis_mode[1]

        grouped_paulis_mode_p_3sigma = grouped_paulis_mode[0] + 3 * grouped_paulis_mode[1]
        grouped_paulis_mode_m_3sigma = grouped_paulis_mode[0] - 3 * grouped_paulis_mode[1]
        self.assertLessEqual(reference, paulis_mode_p_3sigma.real, "Without any pass manager")
        self.assertGreaterEqual(reference, paulis_mode_m_3sigma.real, "Without any pass manager")
        self.assertLessEqual(reference, grouped_paulis_mode_p_3sigma.real, "Without any pass manager")
        self.assertGreaterEqual(reference, grouped_paulis_mode_m_3sigma.real, "Without any pass manager")

    def test_exact_eval(self):
        depth = 1
        var_form = RYRZ(self.qubitOp.num_qubits, depth)
        circuit = var_form.construct_circuit(np.array(np.random.randn(var_form.num_parameters)))

        run_config = {'shots': 1}
        backend = BasicAer.get_backend('statevector_simulator')
        matrix_mode = self.qubitOp.eval('matrix', circuit, backend, run_config=run_config)[0]
        non_matrix_mode = self.qubitOp.eval('paulis', circuit, backend, run_config=run_config)[0]
        diff = abs(matrix_mode - non_matrix_mode)
        self.assertLess(diff, 0.01, "Values: ({} vs {})".format(matrix_mode, non_matrix_mode))

        run_config = {'shots': 1}
        compile_config = {'pass_manager': PassManager()}
        non_matrix_mode = self.qubitOp.eval('paulis', circuit, backend,
                                            run_config=run_config, compile_config=compile_config)[0]
        diff = abs(matrix_mode - non_matrix_mode)
        self.assertLess(diff, 0.01, "Without any pass manager, Values: ({} vs {})".format(matrix_mode, non_matrix_mode))

    def test_create_from_paulis_0(self):
        """Test with single paulis."""
        num_qubits = 3
        for pauli_label in itertools.product('IXYZ', repeat=num_qubits):
            coeff = np.random.random(1)[0]
            pauli_term = [coeff, Pauli.from_label(pauli_label)]
            op = Operator(paulis=[pauli_term])

            depth = 1
            var_form = RYRZ(op.num_qubits, depth)
            circuit = var_form.construct_circuit(np.array(np.random.randn(var_form.num_parameters)))
            run_config = {'shots': 1}
            backend = BasicAer.get_backend('statevector_simulator')
            non_matrix_mode = op.eval('paulis', circuit, backend, run_config=run_config)[0]
            matrix_mode = op.eval('matrix', circuit, backend, run_config=run_config)[0]

            self.assertAlmostEqual(matrix_mode, non_matrix_mode, 6)

    def test_create_from_matrix(self):
        """Test with matrix initialization."""
        for num_qubits in range(1, 3):
            m_size = np.power(2, num_qubits)
            matrix = np.random.rand(m_size, m_size)

            op = Operator(matrix=matrix)

            depth = 1
            var_form = RYRZ(op.num_qubits, depth)
            circuit = var_form.construct_circuit(np.array(np.random.randn(var_form.num_parameters)))
            backend = BasicAer.get_backend('statevector_simulator')
            run_config = {'shots': 1}
            non_matrix_mode = op.eval('paulis', circuit, backend, run_config=run_config)[0]
            matrix_mode = op.eval('matrix', circuit, backend, run_config=run_config)[0]

            self.assertAlmostEqual(matrix_mode, non_matrix_mode, 6)

    def test_multiplication(self):
        """Test multiplication."""
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = Operator(paulis=[pauli_term_a])
        op_b = Operator(paulis=[pauli_term_b])
        new_op = op_a * op_b
        # print(new_op.print_operators())

        self.assertEqual(1, len(new_op.paulis))
        self.assertEqual(-0.25, new_op.paulis[0][0])
        self.assertEqual('ZZYY', new_op.paulis[0][1].to_label())

    def test_addition_paulis_inplace(self):
        """Test addition."""
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = Operator(paulis=[pauli_term_a])
        op_b = Operator(paulis=[pauli_term_b])
        op_a += op_b

        self.assertEqual(2, len(op_a.paulis))

        pauli_c = 'IXYZ'
        coeff_c = 0.25
        pauli_term_c = [coeff_c, Pauli.from_label(pauli_c)]
        op_a += Operator(paulis=[pauli_term_c])

        self.assertEqual(2, len(op_a.paulis))
        self.assertEqual(0.75, op_a.paulis[0][0])

    def test_addition_matrix(self):
        """
            test addition in the matrix mode
        """
        pauli_a = 'IX'
        pauli_b = 'ZY'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = Operator(paulis=[pauli_term_a])
        op_b = Operator(paulis=[pauli_term_b])
        op_a.to_matrix()
        op_b.to_matrix()
        op_a += op_b
        op_a.to_paulis()
        self.assertEqual(2, len(op_a.paulis))
        self.assertEqual(0.5, op_a.paulis[0][0])
        self.assertEqual(0.5, op_a.paulis[1][0])

        pauli_c = 'IX'
        coeff_c = 0.25
        pauli_term_c = [coeff_c, Pauli.from_label(pauli_c)]
        op_c = Operator(paulis=[pauli_term_c])
        op_c.to_matrix()
        op_a.to_matrix()
        op_a += op_c

        op_a.to_paulis()
        self.assertEqual(2, len(op_a.paulis))
        self.assertEqual(0.75, op_a.paulis[0][0])

    def test_subtraction_matrix(self):
        """
            test subtraction in the matrix mode
        """
        pauli_a = 'IX'
        pauli_b = 'ZY'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = Operator(paulis=[pauli_term_a])
        op_b = Operator(paulis=[pauli_term_b])
        op_a.to_matrix()
        op_b.to_matrix()
        op_a -= op_b
        op_a.to_paulis()
        self.assertEqual(2, len(op_a.paulis))
        self.assertEqual(0.5, op_a.paulis[0][0])
        self.assertEqual(-0.5, op_a.paulis[1][0])

        pauli_c = 'IX'
        coeff_c = 0.25
        pauli_term_c = [coeff_c, Pauli.from_label(pauli_c)]
        op_c = Operator(paulis=[pauli_term_c])
        op_c.to_matrix()
        op_a.to_matrix()
        op_a -= op_c

        op_a.to_paulis()
        self.assertEqual(2, len(op_a.paulis))
        self.assertEqual(0.25, op_a.paulis[0][0])

    def test_addition_paulis_noninplace(self):
        """
            test addition
        """
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = Operator(paulis=[pauli_term_a])
        op_b = Operator(paulis=[pauli_term_b])
        copy_op_a = copy.deepcopy(op_a)
        new_op = op_a + op_b

        self.assertEqual(copy_op_a, op_a)
        self.assertEqual(2, len(new_op.paulis))

        pauli_c = 'IXYZ'
        coeff_c = 0.25
        pauli_term_c = [coeff_c, Pauli.from_label(pauli_c)]
        new_op = new_op + Operator(paulis=[pauli_term_c])

        self.assertEqual(2, len(new_op.paulis))
        self.assertEqual(0.75, new_op.paulis[0][0])

    def test_subtraction_noninplace(self):
        """
            test subtraction
        """
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = Operator(paulis=[pauli_term_a])
        op_b = Operator(paulis=[pauli_term_b])
        copy_op_a = copy.deepcopy(op_a)
        new_op = op_a - op_b

        self.assertEqual(copy_op_a, op_a)
        self.assertEqual(2, len(new_op.paulis))
        self.assertEqual(0.5, new_op.paulis[0][0])
        self.assertEqual(-0.5, new_op.paulis[1][0])

        pauli_c = 'IXYZ'
        coeff_c = 0.25
        pauli_term_c = [coeff_c, Pauli.from_label(pauli_c)]
        new_op = new_op - Operator(paulis=[pauli_term_c])

        self.assertEqual(2, len(new_op.paulis))
        self.assertEqual(0.25, new_op.paulis[0][0])

    def test_subtraction_inplace(self):
        """
            test addition
        """
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = Operator(paulis=[pauli_term_a])
        op_b = Operator(paulis=[pauli_term_b])
        op_a -= op_b

        self.assertEqual(2, len(op_a.paulis))

        pauli_c = 'IXYZ'
        coeff_c = 0.25
        pauli_term_c = [coeff_c, Pauli.from_label(pauli_c)]
        op_a -= Operator(paulis=[pauli_term_c])

        self.assertEqual(2, len(op_a.paulis))
        self.assertEqual(0.25, op_a.paulis[0][0])

    def test_scaling_coeff(self):
        """
            test scale
        """
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = Operator(paulis=[pauli_term_a])
        op_b = Operator(paulis=[pauli_term_b])
        op_a += op_b

        self.assertEqual(2, len(op_a.paulis))

        op_a.scaling_coeff(0.7)

        self.assertEqual(2, len(op_a.paulis))
        self.assertEqual(0.35, op_a.paulis[0][0])

    def test_str(self):
        """
            test str
        """
        pauli_a = 'IXYZ'
        pauli_b = 'ZYIX'
        coeff_a = 0.5
        coeff_b = 0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = Operator(paulis=[pauli_term_a])
        op_b = Operator(paulis=[pauli_term_b])
        op_a += op_b

        self.assertEqual("Representation: paulis, qubits: 4, size: 2", str(op_a))

    def test_zero_coeff(self):
        """
            test addition
        """
        pauli_a = 'IXYZ'
        pauli_b = 'IXYZ'
        coeff_a = 0.5
        coeff_b = -0.5
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        pauli_term_b = [coeff_b, Pauli.from_label(pauli_b)]
        op_a = Operator(paulis=[pauli_term_a])
        op_b = Operator(paulis=[pauli_term_b])
        new_op = op_a + op_b
        new_op.zeros_coeff_elimination()

        self.assertEqual(0, len(new_op.paulis), "{}".format(new_op.print_operators()))

        paulis = ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']
        coeffs = [0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op = Operator(paulis=[])
        for coeff, pauli in zip(coeffs, paulis):
            pauli_term = [coeff, Pauli.from_label(pauli)]
            op += Operator(paulis=[pauli_term])

        for i in range(6):
            op_a = Operator(paulis=[[-coeffs[i], Pauli.from_label(paulis[i])]])
            op += op_a
            op.zeros_coeff_elimination()
            self.assertEqual(6-(i+1), len(op.paulis))

    def test_zero_elimination(self):
        pauli_a = 'IXYZ'
        coeff_a = 0.0
        pauli_term_a = [coeff_a, Pauli.from_label(pauli_a)]
        op_a = Operator(paulis=[pauli_term_a])
        self.assertEqual(1, len(op_a.paulis), "{}".format(op_a.print_operators()))
        op_a.zeros_coeff_elimination()

        self.assertEqual(0, len(op_a.paulis), "{}".format(op_a.print_operators()))

    def test_dia_matrix(self):
        """
            test conversion to dia_matrix
        """
        num_qubits = 4
        pauli_term = []
        for pauli_label in itertools.product('IZ', repeat=num_qubits):
            coeff = np.random.random(1)[0]
            pauli_term.append([coeff, Pauli.from_label(pauli_label)])
        op = Operator(paulis=pauli_term)

        op.to_matrix()
        op.to_grouped_paulis()
        op._to_dia_matrix('grouped_paulis')

        self.assertEqual(op.matrix.ndim, 1)

        num_qubits = 4
        pauli_term = []
        for pauli_label in itertools.product('YZ', repeat=num_qubits):
            coeff = np.random.random(1)[0]
            pauli_term.append([coeff, Pauli.from_label(pauli_label)])
        op = Operator(paulis=pauli_term)

        op.to_matrix()
        op._to_dia_matrix('matrix')

        self.assertEqual(op.matrix.ndim, 2)

    def test_equal_operator(self):

        paulis = ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']
        coeffs = [0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op1 = Operator(paulis=[])
        for coeff, pauli in zip(coeffs, paulis):
            pauli_term = [coeff, Pauli.from_label(pauli)]
            op1 += Operator(paulis=[pauli_term])

        paulis = ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']
        coeffs = [0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op2 = Operator(paulis=[])
        for coeff, pauli in zip(coeffs, paulis):
            pauli_term = [coeff, Pauli.from_label(pauli)]
            op2 += Operator(paulis=[pauli_term])

        paulis = ['IXYY', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']
        coeffs = [0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op3 = Operator(paulis=[])
        for coeff, pauli in zip(coeffs, paulis):
            pauli_term = [coeff, Pauli.from_label(pauli)]
            op3 += Operator(paulis=[pauli_term])

        paulis = ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']
        coeffs = [-0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op4 = Operator(paulis=[])
        for coeff, pauli in zip(coeffs, paulis):
            pauli_term = [coeff, Pauli.from_label(pauli)]
            op4 += Operator(paulis=[pauli_term])

        self.assertEqual(op1, op2)
        self.assertNotEqual(op1, op3)
        self.assertNotEqual(op1, op4)
        self.assertNotEqual(op3, op4)

    def test_negation_operator(self):

        paulis = ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']
        coeffs = [0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op1 = Operator(paulis=[])
        for coeff, pauli in zip(coeffs, paulis):
            pauli_term = [coeff, Pauli.from_label(pauli)]
            op1 += Operator(paulis=[pauli_term])

        paulis = ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']
        coeffs = [-0.2, -0.6, -0.8, 0.2, 0.6, 0.8]
        op2 = Operator(paulis=[])
        for coeff, pauli in zip(coeffs, paulis):
            pauli_term = [coeff, Pauli.from_label(pauli)]
            op2 += Operator(paulis=[pauli_term])

        self.assertNotEqual(op1, op2)
        self.assertEqual(op1, -op2)
        self.assertEqual(-op1, op2)
        op1.scaling_coeff(-1.0)
        self.assertEqual(op1, op2)

    def test_chop_real_only(self):

        paulis = ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']
        coeffs = [0.2, 0.6, 0.8, -0.2, -0.6, -0.8]
        op = Operator(paulis=[])
        for coeff, pauli in zip(coeffs, paulis):
            pauli_term = [coeff, Pauli.from_label(pauli)]
            op += Operator(paulis=[pauli_term])

        op1 = copy.deepcopy(op)
        op1.chop(threshold=0.4)
        self.assertEqual(len(op1.paulis), 4, "\n{}".format(op1.print_operators()))
        gt_op1 = Operator(paulis=[])
        for i in range(1, 3):
            pauli_term = [coeffs[i], Pauli.from_label(paulis[i])]
            gt_op1 += Operator(paulis=[pauli_term])
            pauli_term = [coeffs[i+3], Pauli.from_label(paulis[i+3])]
            gt_op1 += Operator(paulis=[pauli_term])
        self.assertEqual(op1, gt_op1)

        op2 = copy.deepcopy(op)
        op2.chop(threshold=0.7)
        self.assertEqual(len(op2.paulis), 2, "\n{}".format(op2.print_operators()))
        gt_op2 = Operator(paulis=[])
        for i in range(2, 3):
            pauli_term = [coeffs[i], Pauli.from_label(paulis[i])]
            gt_op2 += Operator(paulis=[pauli_term])
            pauli_term = [coeffs[i+3], Pauli.from_label(paulis[i+3])]
            gt_op2 += Operator(paulis=[pauli_term])
        self.assertEqual(op2, gt_op2)

        op3 = copy.deepcopy(op)
        op3.chop(threshold=0.9)
        self.assertEqual(len(op3.paulis), 0, "\n{}".format(op3.print_operators()))
        gt_op3 = Operator(paulis=[])
        for i in range(3, 3):
            pauli_term = [coeffs[i], Pauli.from_label(paulis[i])]
            gt_op3 += Operator(paulis=[pauli_term])
            pauli_term = [coeffs[i+3], Pauli.from_label(paulis[i+3])]
            gt_op3 += Operator(paulis=[pauli_term])
        self.assertEqual(op3, gt_op3)

    def test_chop_complex_only_1(self):

        paulis = ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']
        coeffs = [0.2 + -1j * 0.2, 0.6 + -1j * 0.6, 0.8 + -1j * 0.8,
                  -0.2 + -1j * 0.2, -0.6 - -1j * 0.6, -0.8 - -1j * 0.8]
        op = Operator(paulis=[])
        for coeff, pauli in zip(coeffs, paulis):
            pauli_term = [coeff, Pauli.from_label(pauli)]
            op += Operator(paulis=[pauli_term])

        op1 = copy.deepcopy(op)
        op1.chop(threshold=0.4)
        self.assertEqual(len(op1.paulis), 4, "\n{}".format(op1.print_operators()))
        gt_op1 = Operator(paulis=[])
        for i in range(1, 3):
            pauli_term = [coeffs[i], Pauli.from_label(paulis[i])]
            gt_op1 += Operator(paulis=[pauli_term])
            pauli_term = [coeffs[i+3], Pauli.from_label(paulis[i+3])]
            gt_op1 += Operator(paulis=[pauli_term])
        self.assertEqual(op1, gt_op1)

        op2 = copy.deepcopy(op)
        op2.chop(threshold=0.7)
        self.assertEqual(len(op2.paulis), 2, "\n{}".format(op2.print_operators()))
        gt_op2 = Operator(paulis=[])
        for i in range(2, 3):
            pauli_term = [coeffs[i], Pauli.from_label(paulis[i])]
            gt_op2 += Operator(paulis=[pauli_term])
            pauli_term = [coeffs[i+3], Pauli.from_label(paulis[i+3])]
            gt_op2 += Operator(paulis=[pauli_term])
        self.assertEqual(op2, gt_op2)

        op3 = copy.deepcopy(op)
        op3.chop(threshold=0.9)
        self.assertEqual(len(op3.paulis), 0, "\n{}".format(op3.print_operators()))
        gt_op3 = Operator(paulis=[])
        for i in range(3, 3):
            pauli_term = [coeffs[i], Pauli.from_label(paulis[i])]
            gt_op3 += Operator(paulis=[pauli_term])
            pauli_term = [coeffs[i+3], Pauli.from_label(paulis[i+3])]
            gt_op3 += Operator(paulis=[pauli_term])
        self.assertEqual(op3, gt_op3)

    def test_chop_complex_only_2(self):

        paulis = ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']
        coeffs = [0.2 + -1j * 0.8, 0.6 + -1j * 0.6, 0.8 + -1j * 0.2,
                  -0.2 + -1j * 0.8, -0.6 - -1j * 0.6, -0.8 - -1j * 0.2]
        op = Operator(paulis=[])
        for coeff, pauli in zip(coeffs, paulis):
            pauli_term = [coeff, Pauli.from_label(pauli)]
            op += Operator(paulis=[pauli_term])

        op1 = copy.deepcopy(op)
        op1.chop(threshold=0.4)
        self.assertEqual(len(op1.paulis), 6, "\n{}".format(op1.print_operators()))

        op2 = copy.deepcopy(op)
        op2.chop(threshold=0.7)
        self.assertEqual(len(op2.paulis), 4, "\n{}".format(op2.print_operators()))

        op3 = copy.deepcopy(op)
        op3.chop(threshold=0.9)
        self.assertEqual(len(op3.paulis), 0, "\n{}".format(op3.print_operators()))

    def test_representations(self):

        self.assertEqual(len(self.qubitOp.representations), 1)
        self.assertEqual(self.qubitOp.representations, ['matrix'])
        self.qubitOp.to_paulis()
        self.assertEqual(len(self.qubitOp.representations), 1)
        self.assertEqual(self.qubitOp.representations, ['paulis'])
        self.qubitOp.to_grouped_paulis()
        self.assertEqual(len(self.qubitOp.representations), 1)
        self.assertEqual(self.qubitOp.representations, ['grouped_paulis'])

    def test_num_qubits(self):

        op = Operator(paulis=[])
        self.assertEqual(op.num_qubits, 0)
        self.assertEqual(self.qubitOp.num_qubits, self.num_qubits)

    def test_is_empty(self):
        op = Operator(paulis=[])
        self.assertTrue(op.is_empty())
        self.assertFalse(self.qubitOp.is_empty())

    def test_submit_multiple_circuits(self):
        """
            test with single paulis
        """
        num_qubits = 4
        pauli_term = []
        for pauli_label in itertools.product('IXYZ', repeat=num_qubits):
            coeff = np.random.random(1)[0]
            pauli_term.append([coeff, Pauli.from_label(pauli_label)])
        op = Operator(paulis=pauli_term)

        depth = 1
        var_form = RYRZ(op.num_qubits, depth)
        circuit = var_form.construct_circuit(np.array(np.random.randn(var_form.num_parameters)))
        run_config = {'shots': 1}
        backend = BasicAer.get_backend('statevector_simulator')
        non_matrix_mode = op.eval('paulis', circuit, backend, run_config=run_config)[0]
        matrix_mode = op.eval('matrix', circuit, backend, run_config=run_config)[0]

        self.assertAlmostEqual(matrix_mode, non_matrix_mode, 6)

    def test_load_from_file(self):
        paulis = ['IXYZ', 'XXZY', 'IIZZ', 'XXYY', 'ZZXX', 'YYYY']
        coeffs = [0.2 + -1j * 0.8, 0.6 + -1j * 0.6, 0.8 + -1j * 0.2,
                  -0.2 + -1j * 0.8, -0.6 - -1j * 0.6, -0.8 - -1j * 0.2]
        op = Operator(paulis=[])
        for coeff, pauli in zip(coeffs, paulis):
            pauli_term = [coeff, Pauli.from_label(pauli)]
            op += Operator(paulis=[pauli_term])

        op.save_to_file('temp_op.json')
        load_op = Operator.load_from_file('temp_op.json')

        self.assertTrue(os.path.exists('temp_op.json'))
        self.assertEqual(op, load_op)

        os.remove('temp_op.json')

    def test_group_paulis_1(self):
        """
            Test with color grouping approach
        """
        num_qubits = 4
        pauli_term = []
        for pauli_label in itertools.product('IXYZ', repeat=num_qubits):
            coeff = np.random.random(1)[0]
            pauli_term.append([coeff, Pauli.from_label(''.join(pauli_label))])
        op = Operator(paulis=pauli_term)
        paulis = copy.deepcopy(op.paulis)
        op.to_grouped_paulis()
        flattened_grouped_paulis = [pauli for group in op.grouped_paulis for pauli in group[1:]]

        for gp in flattened_grouped_paulis:
            passed = False
            for p in paulis:
                if p[1] == gp[1]:
                    passed = p[0] == gp[0]
                    break
            self.assertTrue(passed, "non-existed paulis in grouped_paulis: {}".format(gp[1].to_label()))

    def test_group_paulis_2(self):
        """
            Test with normal grouping approach
        """

        num_qubits = 4
        pauli_term = []
        for pauli_label in itertools.product('IXYZ', repeat=num_qubits):
            coeff = np.random.random(1)[0]
            pauli_term.append([coeff, Pauli.from_label(''.join(pauli_label))])
        op = Operator(paulis=pauli_term)
        op.coloring = None
        paulis = copy.deepcopy(op.paulis)
        op.to_grouped_paulis()
        flattened_grouped_paulis = [pauli for group in op.grouped_paulis for pauli in group[1:]]

        for gp in flattened_grouped_paulis:
            passed = False
            for p in paulis:
                if p[1] == gp[1]:
                    passed = p[0] == gp[0]
                    break
            self.assertTrue(passed, "non-existed paulis in grouped_paulis: {}".format(gp[1].to_label()))


if __name__ == '__main__':
    unittest.main()
