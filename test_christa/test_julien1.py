from qiskit import BasicAer, QuantumRegister, QuantumCircuit
from qiskit.aqua.algorithms import AmplitudeEstimation
from qiskit.aqua.components.uncertainty_models import NormalDistribution
from test_christa.init_shift_a_factory import IntShiftAFactory

# Parameters for the optimisation problem
m = 2               # number of qubits
mu = 2              # parameter `mu` of the probability distribution
low, high = 0, 10   # interval for prob dist
c_approx = 0.1      # approx factor in sin^2(cx) = (cx)^2

# Define probability distribution
uncertainty_model = NormalDistribution(num_target_qubits=3,
                                       mu=mu,
                                       low=low,
                                       high=high)

# Test case: y is simply zero
y_num_bits = 3
y_qr = QuantumRegister(y_num_bits)
y_qc = QuantumCircuit(y_qr)

# Run QAE
a_factory = IntShiftAFactory(uncertainty_model, c_approx, y_qc)
ae = AmplitudeEstimation(m, a_factory=a_factory)
result = ae.run(quantum_instance=BasicAer.get_backend("statevector_simulator"))

# Print QAE result
print(result)