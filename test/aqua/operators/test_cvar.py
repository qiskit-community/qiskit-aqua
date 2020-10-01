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


from test.aqua import QiskitAquaTestCase

import numpy as np
from ddt import ddt, data

from qiskit import QuantumCircuit
from qiskit.aqua import AquaError
from qiskit.aqua.operators import (
    CVarStateFn, StateFn, Z, I, X, Plus, PauliExpectation, MatrixExpectation, CVaRExpectation,
    ListOp, CircuitOp
)


class TestCVaRMeasurement(QiskitAquaTestCase):
    """Test the CVaR measurement."""

    def expected_cvar(self, statevector, operator, alpha):
        """Compute the expected CVaR expected value."""

        probabilities = statevector * np.conj(statevector)

        # get energies
        num_bits = int(np.log2(len(statevector)))
        energies = []
        for i, _ in enumerate(probabilities):
            basis_state = np.binary_repr(i, num_bits)
            energies += [operator.eval(basis_state).eval(basis_state)]

        # sort ascending
        i_sorted = np.argsort(energies)
        energies = [energies[i] for i in i_sorted]
        probabilities = [probabilities[i] for i in i_sorted]

        # add up
        result = 0
        accumulated_probabilities = 0
        for energy, probability in zip(energies, probabilities):
            accumulated_probabilities += probability
            if accumulated_probabilities <= alpha:
                result += probability * energy
            else:  # final term
                result += (alpha - accumulated_probabilities + probability) * energy
                break

        return result / alpha

    def test_cvar_simple(self):
        """Test a simple case with a single Pauli."""
        theta = 1.2
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        statefn = StateFn(qc)

        for alpha in [0.2, 0.4, 1]:
            with self.subTest(alpha=alpha):
                cvar = (CVarStateFn(Z, alpha) @ statefn).eval()
                ref = self.expected_cvar(statefn.to_matrix(), Z, alpha)
                self.assertAlmostEqual(cvar, ref)

    def invalid_input(self):
        """Test invalid input raises an error."""
        op = Z

        with self.subTest('alpha < 0'):
            with self.assertRaises(ValueError):
                _ = CVarStateFn(op, alpha=-0.2)

        with self.subTest('alpha > 1'):
            with self.assertRaises(ValueError):
                _ = CVarStateFn(op, alpha=12.3)

        with self.subTest('operator not diagonal'):
            op = X ^ Z + Z ^ I
            with self.assertRaises(AquaError):
                _ = CVarStateFn(op)


@ddt
class TestCVaRExpectation(QiskitAquaTestCase):
    """Test the CVaR expectation object."""

    def test_construction(self):
        """Test the correct operator expression is constructed."""

        alpha = 0.5
        base_expecation = PauliExpectation()
        cvar_expecation = CVaRExpectation(alpha=alpha, expectation=base_expecation)

        with self.subTest('single operator'):
            op = ~StateFn(Z) @ Plus
            expected = CVarStateFn(Z, alpha) @ Plus
            cvar = cvar_expecation.convert(op)
            self.assertEqual(cvar, expected)

        with self.subTest('list operator'):
            op = ~StateFn(ListOp([Z ^ Z, I ^ Z])) @ (Plus ^ Plus)
            expected = ListOp(
                [CVarStateFn((Z ^ Z), alpha) @ (Plus ^ Plus),
                 CVarStateFn((I ^ Z), alpha) @ (Plus ^ Plus)]
                )
            cvar = cvar_expecation.convert(op)
            self.assertEqual(cvar, expected)

    @data(PauliExpectation(), MatrixExpectation())
    def test_underlying_expectation(self, base_expecation):
        """Test the underlying expectation works correctly."""

        cvar_expecation = CVaRExpectation(alpha=0.3, expectation=base_expecation)
        circuit = QuantumCircuit(2)
        circuit.z(0)
        circuit.cp(0.5, 0, 1)
        circuit.t(1)
        op = ~StateFn(CircuitOp(circuit)) @ (Plus ^ 2)

        cvar = cvar_expecation.convert(op)
        expected = base_expecation.convert(op)

        # test if the operators have been transformed in the same manner
        self.assertEqual(cvar.oplist[0].primitive, expected.oplist[0].primitive)
