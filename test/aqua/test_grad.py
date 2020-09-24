# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# =============================================================================

""" Test Quantum Gradient Framework """

import unittest
from itertools import product

import numpy as np
from ddt import ddt, data, idata, unpack
from qiskit import QuantumCircuit, QuantumRegister, BasicAer
from qiskit.aqua import QuantumInstance, aqua_globals
from qiskit.aqua.operators import X, Z, StateFn, CircuitStateFn, ListOp, CircuitSampler
from qiskit.aqua.operators.gradients import Gradient, NaturalGradient, Hessian
from qiskit.aqua.operators.gradients.qfi import QFI
from qiskit.circuit import Parameter, ParameterExpression
from sympy import Symbol, cos

from test.aqua import QiskitAquaTestCase


@ddt
class TestGradients(QiskitAquaTestCase):
    """ Test Qiskit Gradient Framework """

    def setUp(self):
        super().setUp()
        aqua_globals.random_seed = 50
        # Set quantum instance to run the quantum generator
        self.qi = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                                  seed_simulator=2,
                                  seed_transpiler=2)

    @data('lin_comb', 'param_shift', 'fin_diff')
    def test_gradient_p(self, method):
        """Test the state gradient for p
        |psi> = 1/sqrt(2)[[1, exp(ia)]]
        Tr(|psi><psi|Z) = 0
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a)
        """
        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        params = [a]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.p(params[0], q[0])
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: 0}, {a: np.pi / 2}]
        correct_values = [-0.5 / np.sqrt(2), 0, -0.5]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(state_grad.assign_parameters(value_dict).eval(),
                                                 correct_values[i], decimal=1)

    @data('lin_comb', 'param_shift', 'fin_diff')
    def test_gradient_u(self, method):
        """Test the state gradient for U
        Tr(|psi><psi|Z) = - 0.5 sin(a)cos(c)
        Tr(|psi><psi|X) = cos^2(a/2) cos(b+c) - sin^2(a/2) cos(b-c)
        """

        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        b = Parameter('b')
        c = Parameter('c')

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.u(a, b, c, q[0])

        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)
        params = [a, b, c]
        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: 0, c: 0}, {a: np.pi / 4, b: np.pi / 4, c: np.pi / 4}]
        correct_values = [[0.3536, 0, 0], [0.3232, -0.42678, -0.92678]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(state_grad.assign_parameters(value_dict).eval(),
                                                 correct_values[i],
                                                 decimal=1)
        """
        Tr(|psi><psi|Z) = - 0.5 sin(a)cos(c)
        Tr(|psi><psi|X) = cos^2(a/2) cos(b+c) - sin^2(a/2) cos(b-c)
        dTr(|psi><psi|H)/da = 0.5(cos(2a)) + 0.5()
        """

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.u(a, a, a, q[0])

        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)
        params = [a]
        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: np.pi / 2}]
        correct_values = [[-1.03033], [-1]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(state_grad.assign_parameters(value_dict).eval(),
                                                 correct_values[i],
                                                 decimal=1)

    @data('lin_comb', 'param_shift', 'fin_diff')
    def test_state_gradient1(self, method):
        """Test the state gradient

        Tr(|psi><psi|Z) = sin(a)sin(b)
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a) - 1 cos(a)sin(b)
        d<H>/db = - 1 sin(a)cos(b)
        """

        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: np.pi}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
                       {params[0]: np.pi / 2, params[1]: np.pi / 4}]
        correct_values = [[-0.5 / np.sqrt(2), 1 / np.sqrt(2)], [-0.5 / np.sqrt(2) - 0.5, -1 / 2.],
                          [-0.5, -1 / np.sqrt(2)]]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(state_grad.assign_parameters(value_dict).eval(),
                                                 correct_values[i], decimal=1)

    @data('lin_comb', 'param_shift', 'fin_diff')
    def test_state_gradient2(self, method):
        """Test the state gradient 2

        Tr(|psi><psi|Z) = sin(a)sin(a)
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a) - 2 cos(a)sin(a)
        """
        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        # b = Parameter('b')
        params = [a]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(a, q[0])
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: 0},
                       {a: np.pi / 2}]
        correct_values = [-1.353553, -0, -0.5]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(state_grad.assign_parameters(value_dict).eval(),
                                                 correct_values[i],
                                                 decimal=1)

    @data('lin_comb', 'param_shift', 'fin_diff')
    def test_state_gradient3(self, method):
        """Test the state gradient 3

        Tr(|psi><psi|Z) = sin(a)sin(c(a)) = sin(a)sin(cos(a)+1)
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a) - 1 cos(a)sin(cos(a)+1) + 1 sin^2(a)cos(cos(a)+1)
        """
        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        # b = Parameter('b')
        params = [a]
        x = Symbol('x')
        expr = cos(x) + 1
        c = ParameterExpression({a: x}, expr)

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(c, q[0])
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: 0}, {a: np.pi / 2}]
        correct_values = [-1.1220, -0.9093, 0.0403]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(state_grad.assign_parameters(value_dict).eval(),
                                                 correct_values[i],
                                                 decimal=1)

    @data('lin_comb', 'param_shift', 'fin_diff')
    def test_state_gradient4(self, method):
        """Test the state gradient 4
         Tr(|psi><psi|ZX) = -cos(a)
         daTr(|psi><psi|ZX) = sin(a)
        """

        H = X ^ Z
        a = Parameter('a')
        params = [a]

        q = QuantumRegister(2)
        qc = QuantumCircuit(q)
        qc.x(q[0])
        qc.h(q[1])
        qc.crz(a, q[0], q[1])
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4}, {a: 0},
                       {a: np.pi / 2}]
        correct_values = [1/np.sqrt(2), 0, 1]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(state_grad.assign_parameters(value_dict).eval(),
                                                 correct_values[i],
                                                 decimal=1)

    @data('lin_comb', 'param_shift', 'fin_diff')
    def test_state_hessian(self, method):
        """Test the state Hessian

        Tr(|psi><psi|Z) = sin(a)sin(b)
        Tr(|psi><psi|X) = cos(a)
        d^2<H>/da^2 = - 0.5 cos(a) + 1 sin(a)sin(b)
        d^2<H>/dbda = - 1 cos(a)cos(b)
        d^2<H>/dbda = - 1 cos(a)cos(b)
        d^2<H>/db^2 = + 1 sin(a)sin(b)
        """

        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        b = Parameter('b')
        params = [(a, a), (a, b), (b, b)]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(b, q[0])

        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)
        state_hess = Hessian(method=method).convert(operator=op, params=params)

        values_dict = [{a: np.pi / 4, b: np.pi}, {a: np.pi / 4, b: np.pi / 4},
                       {a: np.pi / 2, b: np.pi / 4}]
        correct_values = [[-0.5 / np.sqrt(2), 1 / np.sqrt(2), 0],
                          [-0.5 / np.sqrt(2) + 0.5, -1 / 2., 0.5],
                          [1 / np.sqrt(2), 0, 1 / np.sqrt(2)]]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(state_hess.assign_parameters(value_dict).eval(),
                                                 correct_values[i], decimal=1)

    @data('lin_comb', 'param_shift', 'fin_diff')
    def test_prob_grad(self, method):
        """Test the probability gradient

        dp0/da = cos(a)sin(b) / 2
        dp1/da = - cos(a)sin(b) / 2
        dp0/db = sin(a)cos(b) / 2
        dp1/db = - sin(a)cos(b) / 2
        """

        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        op = CircuitStateFn(primitive=qc, coeff=1.)

        prob_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: 0}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
                       {params[0]: np.pi / 2, params[1]: np.pi}]
        correct_values = [[[0, 0], [1 / (2 * np.sqrt(2)), - 1 / (2 * np.sqrt(2))]],
                          [[1 / 4, - 1 / 4], [1 / 4, - 1 / 4]],
                          [[0, 0], [- 1 / 2, 1 / 2]]]
        for i, value_dict in enumerate(values_dict):
            for j, prob_grad_result in enumerate(prob_grad.assign_parameters(value_dict).eval()):
                np.testing.assert_array_almost_equal(prob_grad_result,
                                                     correct_values[i][j], decimal=1)

    
    def test_lin_comb_qfi(self):
        """Test if the quantum fisher information calculation is correct

        QFI = [[1, 0], [0, 1]] - [[0, 0], [0, cos^2(a)]]
        """

        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        op = CircuitStateFn(primitive=qc, coeff=1.)
        qfi = QFI(method='lin_comb').convert(operator=op, params=params)
        values_dict = [{params[0]: np.pi / 4, params[1]: 0.1}, {params[0]: np.pi, params[1]: 0.1},
                       {params[0]: np.pi / 2, params[1]: 0.1}]
        correct_values = [[[1, 0], [0, 0.5]], [[1, 0], [0, 0]], [[1, 0], [0, 1]]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(qfi.assign_parameters(value_dict).eval(),
                                                 correct_values[i], decimal=1)

    @data('block_diag', 'diag')
    def test_overlap_qfi(self, approx):
        """Test if the quantum fisher information calculation is correct

        QFI = [[1, 0], [0, 1]] - [[0, 0], [0, cos^2(a)]]
        """

        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        op = CircuitStateFn(primitive=qc, coeff=1.)
        qfi = QFI(method='overlap', approx=approx).convert(operator=op, params=params)
        values_dict = [{params[0]: np.pi / 4, params[1]: 0.1}, {params[0]: np.pi, params[1]: 0.1},
                       {params[0]: np.pi / 2, params[1]: 0.1}]
        correct_values = [[[1, 0], [0, 0.5]], [[1, 0], [0, 0]], [[1, 0], [0, 1]]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(qfi.assign_parameters(value_dict).eval(),
                                                 correct_values[i], decimal=1)

    @idata(product(['lin_comb', 'param_shift'],
                   [None, 'lasso', 'perturb_diag', 'perturb_diag_elements']))
    @unpack
    def test_natural_gradient(self, method, regularization):
        """Test the natural gradient"""
        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)
        nat_grad = NaturalGradient(grad_method=method,
                                   regularization=regularization).convert(operator=op,
                                                                          params=params)
        values_dict = [{params[0]: np.pi / 4, params[1]: np.pi / 2}]
        correct_values = [[-2.36003979, 2.06503481]] if regularization == 'ridge' else [[-4.2, 0]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(nat_grad.assign_parameters(value_dict).eval(),
                                                 correct_values[i],
                                                 decimal=0)

    @idata(product(['lin_comb', 'param_shift', 'fin_diff'], [True, False]))
    @unpack
    def test_jax_chain_rule(self, method: str, autograd: bool):
        """Test the chain rule functionality using Jax

        d<H>/d<X> = 2<X>
        d<H>/d<Z> = - sin(<Z>)
        <Z> = Tr(|psi><psi|Z) = sin(a)sin(b)
        <X> = Tr(|psi><psi|X) = cos(a)
        d<H>/da = d<H>/d<X> d<X>/da + d<H>/d<Z> d<Z>/da = - 2 cos(a)sin(a)
                    - sin(sin(a)sin(b)) * cos(a)sin(b)
        d<H>/db = d<H>/d<X> d<X>/db + d<H>/d<Z> d<Z>/db = - sin(sin(a)sin(b)) * sin(a)cos(b)
        """

        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        def combo_fn(x):
            import jax.numpy as jnp
            return jnp.power(x[0], 2) + jnp.cos(x[1])

        def grad_combo_fn(x):  # should be `*x` to align with autograd
            return np.array([2 * x[0], -np.sin(x[1])])

        op = ListOp([~StateFn(X) @ CircuitStateFn(primitive=qc, coeff=1.),
                     ~StateFn(Z) @ CircuitStateFn(primitive=qc, coeff=1.)], combo_fn=combo_fn,
                    grad_combo_fn=None if autograd else grad_combo_fn)

        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: np.pi}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
                       {params[0]: np.pi / 2, params[1]: np.pi / 4}]
        correct_values = [[-1., 0.], [-1.2397, -0.2397], [0, -0.45936]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(state_grad.assign_parameters(value_dict).eval(),
                                                 correct_values[i],
                                                 decimal=1)

    @data('lin_comb', 'param_shift', 'fin_diff')
    def test_operator_coefficient_gradient(self, method):
        """Test the operator coefficient gradient

        Tr( | psi > < psi | Z) = sin(a)sin(b)
        Tr( | psi > < psi | X) = cos(a)
        """
        a = Parameter('a')
        b = Parameter('b')
        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(b, q[0])

        coeff_0 = Parameter('c_0')
        coeff_1 = Parameter('c_1')
        H = coeff_0 * X + coeff_1 * Z
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.0)
        gradient_coeffs = [coeff_0, coeff_1]
        coeff_grad = Gradient(grad_method=method).convert(op, gradient_coeffs)
        values_dict = [{coeff_0: 0.5, coeff_1: -1, a: np.pi / 4, b: np.pi},
                       {coeff_0: 0.5, coeff_1: -1, a: np.pi / 4, b: np.pi / 4}]
        correct_values = [[1 / np.sqrt(2), 0], [1 / np.sqrt(2), 1 / 2]]
        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(coeff_grad.assign_parameters(value_dict).eval(),
                                                 correct_values[i],
                                                 decimal=1)

    @data('lin_comb', 'param_shift', 'fin_diff')
    def test_operator_coefficient_hessian(self, method):
        """Test the operator coefficient hessian

        <Z> = Tr( | psi > < psi | Z) = sin(a)sin(b)
        <X> = Tr( | psi > < psi | X) = cos(a)
        d<H>/dc_0 = 2 * c_0 * <X> + c_1 * <Z>
        d<H>/dc_1 = c_0 * <Z>
        d^2<H>/dc_0^2 = 2 * <X>
        d^2<H>/dc_0dc_1 = <Z>
        d^2<H>/dc_1dc_0 = <Z>
        d^2<H>/dc_1^2 = 0
        """
        a = Parameter('a')
        b = Parameter('b')
        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(a, q[0])
        qc.rx(b, q[0])

        coeff_0 = Parameter('c_0')
        coeff_1 = Parameter('c_1')
        H = coeff_0 * coeff_0 * X + coeff_1 * coeff_0 * Z
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)
        gradient_coeffs = [(coeff_0, coeff_0), (coeff_0, coeff_1), (coeff_1, coeff_1)]
        coeff_grad = Hessian(method=method).convert(op, gradient_coeffs)
        values_dict = [{coeff_0: 0.5, coeff_1: -1, a: np.pi / 4, b: np.pi},
                       {coeff_0: 0.5, coeff_1: -1, a: np.pi / 4, b: np.pi / 4}]

        correct_values = [[2 / np.sqrt(2), 0, 0], [2 / np.sqrt(2), 1 / 2, 0]]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(coeff_grad.assign_parameters(value_dict).eval(),
                                                 correct_values[i],
                                                 decimal=1)

    @data('lin_comb', 'param_shift')
    def test_circuit_sampler(self, method):
        """Test the gradient with circuit sampler

        Tr(|psi><psi|Z) = sin(a)sin(b)
        Tr(|psi><psi|X) = cos(a)
        d<H>/da = - 0.5 sin(a) - 1 cos(a)sin(b)
        d<H>/db = - 1 sin(a)cos(b)
        """

        H = 0.5 * X - 1 * Z
        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])
        op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)
        state_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: np.pi / 4, b: np.pi}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
                       {params[0]: np.pi / 2, params[1]: np.pi / 4}]
        correct_values = [[-0.5 / np.sqrt(2), 1 / np.sqrt(2)], [-0.5 / np.sqrt(2) - 0.5, -1 / 2.],
                          [-0.5, -1 / np.sqrt(2)]]

        backend = BasicAer.get_backend('qasm_simulator')
        q_instance = QuantumInstance(backend=backend, shots=8000)

        for i, value_dict in enumerate(values_dict):
            sampler = CircuitSampler(backend=q_instance).convert(state_grad,
                                                                 params={k: [v] for k, v in
                                                                         value_dict.items()})
            np.testing.assert_array_almost_equal(sampler.eval()[0], correct_values[i], decimal=1)

    @data('lin_comb', 'param_shift')
    def test_circuit_sampler2(self, method):
        """Test the probability gradient with the circuit sampler

        dp0/da = cos(a)sin(b) / 2
        dp1/da = - cos(a)sin(b) / 2
        dp0/db = sin(a)cos(b) / 2
        dp1/db = - sin(a)cos(b) / 2
        """

        a = Parameter('a')
        b = Parameter('b')
        params = [a, b]

        q = QuantumRegister(1)
        qc = QuantumCircuit(q)
        qc.h(q)
        qc.rz(params[0], q[0])
        qc.rx(params[1], q[0])

        op = CircuitStateFn(primitive=qc, coeff=1.)

        prob_grad = Gradient(grad_method=method).convert(operator=op, params=params)
        values_dict = [{a: [np.pi / 4], b: [0]}, {params[0]: [np.pi / 4], params[1]: [np.pi / 4]},
                       {params[0]: [np.pi / 2], params[1]: [np.pi]}]
        correct_values = [[[0, 0], [1 / (2 * np.sqrt(2)), - 1 / (2 * np.sqrt(2))]],
                            [[1 / 4, -1 / 4], [1 / 4, - 1 / 4]],
                            [[ 0, 0],  [ - 1 / 2, 1 / 2]]]

        backend = BasicAer.get_backend('qasm_simulator')
        q_instance = QuantumInstance(backend=backend, shots=10000)

        for i, value_dict in enumerate(values_dict):
            sampler = CircuitSampler(backend=q_instance).convert(prob_grad,
                                                                 params=value_dict)
            result = sampler.eval()
            np.testing.assert_array_almost_equal(result[0], correct_values[i], decimal=1)


if __name__ == '__main__':
    unittest.main()

