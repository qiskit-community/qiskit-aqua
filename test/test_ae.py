from qiskit_aqua.algorithms.single_sample import AmplitudeEstimation
from qiskit_aqua.algorithms.components.uncertainty_problems import EuropeanCallExpectedValue
from qiskit_aqua.algorithms.components.uncertainty_models import NormalDistribution
from qiskit import Aer


# number of qubits to represent the uncertainty
num_uncertainty_qubits = 2
# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low  = 0
high = 2**num_uncertainty_qubits-1
mu = (low + high)/2
sigma = 1
# construct circuit factory for uncertainty model
uncertainty_model = NormalDistribution(num_uncertainty_qubits, mu=mu, sigma=sigma, low=low, high=high)
# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = 1
# set the approximation scaling for the payoff function
c_approx = 0.5
# construct circuit factory for payoff function
european_call = EuropeanCallExpectedValue(
    uncertainty_model,
    strike_price=strike_price,
    c_approx=c_approx,
    i_state=list(range(num_uncertainty_qubits)),
    i_compare=num_uncertainty_qubits,
    i_objective=num_uncertainty_qubits + 1
)
# set number of evaluation qubits (samples)
m = 5
# construct amplitude estimation
ae = AmplitudeEstimation(m, european_call)
backend = Aer.get_backend('statevector_simulator')
ae.setup_quantum_backend(backend=backend, shots=100, skip_transpiler=False)
result = ae.run()
print(result)
exit(0)
