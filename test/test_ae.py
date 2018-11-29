from qiskit_aqua.algorithms.single_sample import AmplitudeEstimation
from qiskit_aqua.algorithms.components.problems import EuropeanCallExpectedValue
from qiskit_aqua.algorithms.components.problems import EuropeanCallDelta
from qiskit_aqua.algorithms.components.uncertainty_models import NormalDistribution

import matplotlib.pyplot as plt
import numpy as np
from qiskit import execute, ClassicalRegister



# number of qubits to represent the uncertainty
num_uncertainty_qubits = 2

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low  = 0
high = 2**num_uncertainty_qubits-1

mu = (low + high)/2
sigma = 1

# construct circuit factory for uncertainty model
uncertainty_model = NormalDistribution(num_uncertainty_qubits, mu=mu, sigma=sigma, low=low, high=high)


# plot probability distribution
x = uncertainty_model.values
y = uncertainty_model.probabilities
plt.bar(x, y)
plt.xticks(x, size=15)
plt.yticks(size=15)
plt.grid()
plt.xlabel('Spot Price at Maturity $S_T$ (\$)', size=15)
plt.ylabel('Probability ($\%$)', size=15)
plt.show()


# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = 1

# set the approximation scaling for the payoff function
c_approx = 0.5

# construct circuit factory for payoff function
european_call = EuropeanCallExpectedValue(uncertainty_model, strike_price=strike_price, c_approx=c_approx)


# plot payoff function
x = uncertainty_model.values
y = np.maximum(0, x - strike_price)
plt.plot(x, y, 'ro-')
plt.grid()
plt.title('Payoff Function', size=15)
plt.xlabel('Spot Price', size=15)
plt.ylabel('Payoff', size=15)
plt.xticks(x)
plt.show()


# evaluate exact expected value
exact_value = np.dot(uncertainty_model.probabilities, y)
exact_delta = sum(uncertainty_model.probabilities[x >= strike_price])
print('exact expected value:\t%.4f' % exact_value)
print('exact delta value:   \t%.4f' % exact_delta)





# set number of evaluation qubits (samples)
m = 5

# assign qubit indices (state/uncertainty quits, compare with strike price qubit, objective qubit)
i_state = range(num_uncertainty_qubits)
i_compare = len(i_state)
i_objective = i_compare + 1
params = {'i_state': i_state, 'i_compare': i_compare, 'i_objective': i_objective}

# construct amplitude estimation
ae = AmplitudeEstimation(m, european_call, additional_params=params)




# get quantum circuit for amplitude estimation
qc = ae.get_circuit()







# run circuit on statevector simulator
import qiskit
state_vector = np.asarray(execute(qc, qiskit.Aer.get_backend('statevector_simulator')).result().get_statevector(qc))

state_probabilities = np.real(state_vector.conj() * state_vector)






# run circuit on QASM simulator
cr = ClassicalRegister(m)
qc.add_register(cr)
qc.measure([q for q in qc.qregs if q.name == 'a'][0], cr)
shots=1000
results = execute(qc, shots=shots, backend=qiskit.Aer.get_backend('qasm_simulator'))
print(results.result().get_counts())
y_items = []
for state, counts in results.result().get_counts().items():
    y_items += [(int(state[:m][::-1], 2), counts/shots)]







# evaluate results
a_probabilities, y_probabilities = ae.evaluate_results(state_probabilities)

a_items = [(a, p) for (a, p) in a_probabilities.items() if p > 1e-6]
y_items = [(y, p) for (y, p) in y_probabilities.items() if p > 1e-6]
a_items = sorted(a_items)
y_items = sorted(y_items)

# map estimated values to original range and extract probabilities
mapped_values = [european_call.value_to_estimator(a_item[0]) for a_item in a_items]
values = [a_item[0] for a_item in a_items]
y_values = [y_item[0] for y_item in y_items]
probabilities = [a_item[1] for a_item in a_items]
mapped_items = [(mapped_values[i], probabilities[i]) for i in range(len(mapped_values))]

# determine most likely estimator
estimator = None
max_prob = 0
for val, prob in mapped_items:
    if prob > max_prob:
        max_prob = prob
        estimator = val
print('Exact value:    \t%.4f' % exact_value)
print('Estimated value:\t%.4f' % estimator)
print('Probability:    \t%.4f' % max_prob)



# plot estimated values for "a"
plt.bar(values, probabilities, width=0.5/len(probabilities))
plt.xticks([0, 0.25, 0.5, 0.75, 1], size=15)
plt.yticks([0, 0.25, 0.5, 0.75, 1], size=15)
plt.title('"a" Value', size=15)
plt.ylabel('Probability', size=15)
plt.xlim((0,1))
plt.ylim((0,1))
plt.grid()
plt.show()

# plot estimated values for option price
plt.bar(mapped_values, probabilities, width=1/len(probabilities))
plt.plot([exact_value, exact_value], [0,1], 'r--', linewidth=2)
plt.xticks(size=15)
plt.yticks([0, 0.25, 0.5, 0.75, 1], size=15)
plt.title('Estimated Option Price', size=15)
plt.ylabel('Probability', size=15)
plt.ylim((0,1))
plt.grid()
plt.show()






def dist(a, b):
    a = a - np.floor(a)
    b = b - np.floor(b)
    if a > b:
        c = b
        b = a
        a = c
    d1 = b - a
    d2 = 1 + a - b
    return np.minimum(d1, d2)

def prob_x(x, omega):
    M = 2**m
    if np.isclose(x, omega * M):
        return 1.0
    else:
        p = pow(np.sin(M*dist(omega, x/M)*np.pi), 2.0)
        p /= pow(M*np.sin(dist(omega, x/M)*np.pi), 2.0)
    return p

def log_likelihood(omega, y_items):
    value = 0
    for y_item in y_items:
        value += y_item[1]*np.log(prob_x(y_item[0], omega))
    return value






omegas = np.linspace(0,1,1000, endpoint=False)
yp = np.zeros(len(omegas))
for i in range(len(omegas)):
    yp[i] = log_likelihood(omegas[i], y_items)
plt.plot(omegas, yp)
plt.ylim([-20,0])
plt.show()
i_max = np.argmax(yp)
omega_max = omegas[i_max]
a_max = pow(np.sin(omega_max*np.pi), 2.0)
mapped_max = european_call.value_to_estimator(a_max)
print('omega_max', omega_max)
print('a_max    ', a_max)
print('mapped_v ', mapped_max)