import mini_lambda as ml
import numpy as np
import qiskit
import os
import sys
import copy
import inspect
from scipy.linalg import block_diag
import copy
from qiskit import Aer, QuantumCircuit, QuantumRegister, ClassicalRegister
import sympy as sym
import jax.numpy as jnp
from jax import grad, jit, vmap
from functools import partial, reduce
from typing import Optional, Callable, Union, List, Dict, Tuple
from collections.abc import Iterable
from qiskit.aqua import QuantumInstance
from qiskit.aqua.operators import (I,X,Y,Z, ListOp, PauliOp, Zero, DictStateFn, SummedOp, ComposedOp, TensoredOp,
                                  OperatorBase, CircuitOp, CircuitStateFn, CircuitSampler, MatrixOp,
                                  StateFn, PrimitiveOp, PauliExpectation,  One, Zero, OperatorStateFn, VectorStateFn)
from qiskit.circuit import ParameterVector, Parameter, ParameterExpression
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit.aqua.operators.gradients import DerivativeBase, Gradient, Hessian, QFI
from qiskit.quantum_info import Statevector
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
op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)
gradient_coeffs = [coeff_0, coeff_1]
coeff_grad = Gradient().convert(op, gradient_coeffs)
values_dict = [{coeff_0: 0.5, coeff_1: -1, a: np.pi / 4, b: np.pi},
               {coeff_0: 0.5, coeff_1: -1, a: np.pi / 4, b: np.pi / 4}]
correct_values = [[1 / np.sqrt(2), 0], [1 / np.sqrt(2), 1 / 2]]
print(coeff_grad)
for i, value_dict in enumerate(values_dict):
    print(i)
    np.testing.assert_array_almost_equal(coeff_grad.assign_parameters(value_dict).eval(), correct_values[i],
                                         decimal=4)