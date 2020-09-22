
from qiskit.aqua.operators.gradients.circuit_gradient_methods.lin_comb_gradient import LinCombGradient
from qiskit import BasicAer
from qiskit.aqua.operators import X, Z, CircuitStateFn
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterExpression
import numpy as np
from sympy import Symbol, cos

from qiskit.aqua import QuantumInstance, aqua_globals


aqua_globals.random_seed = 50
# Set quantum instance to run the quantum generator
qi = QuantumInstance(backend=BasicAer.get_backend('statevector_simulator'),
                     seed_simulator=2,
                     seed_transpiler=2)

H = 0.5 * X - 1 * Z
a = Parameter('a')
b = Parameter('b')
x = Symbol('x')
expr = cos(x) + 1
c = ParameterExpression({a: x}, expr)
params = [a, b]

q = QuantumRegister(1)
qc = QuantumCircuit(q)
qc.h(q)
qc.u2(b, b, q[0])
qc.rz(c, q[0])
qc.rx(c, q[0])

# op = ~StateFn(H) @ CircuitStateFn(primitive=qc, coeff=1.)
op = CircuitStateFn(primitive=qc, coeff=1.)

state_grad = LinCombGradient().convert(operator=op, params=params)
# qfi = QFI().convert(operator=op, params=params)
# state_grad = ParamShiftGradient().convert(operator=op, params=params)

values_dict = [{a: np.pi / 4, b: np.pi}, {params[0]: np.pi / 4, params[1]: np.pi / 4},
               {params[0]: np.pi / 2, params[1]: np.pi / 4}]
correct_values = [[-0.5 / np.sqrt(2), 1 / np.sqrt(2)], [-0.5 / np.sqrt(2) - 0.5, -1 / 2.],
                  [-0.5, -1 / np.sqrt(2)]]
for i, value_dict in enumerate(values_dict):
    print(state_grad.assign_parameters(value_dict).eval())

# converter = CircuitSampler(backend=qi).convert(state_grad)
# values_dict = {params[0]: np.pi / 4, params[1]: 0.1}
# qfi_value = qfi.assign_parameters(values_dict).eval()
# correct_qfi = np.allclose(qfi_value, [[1, 0], [0, 0.5]], atol=1e-6)
# values_dict = {params[0]: np.pi, params[1]: 0.1}
# qfi_value = qfi.assign_parameters(values_dict).eval()
# correct_qfi &= np.allclose(qfi_value, [[1, 0], [0, 0]], atol=1e-6)
# values_dict = {params[0]: np.pi/2, params[1]: 0.1}
# qfi_value = qfi.assign_parameters(values_dict).eval()
# correct_qfi &= np.allclose(qfi_value, [[1, 0], [0, 1]], atol=1e-6)
# print(correct_qfi)
# a = 0
