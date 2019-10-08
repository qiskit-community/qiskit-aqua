from qiskit.aqua.algorithms import VQE, ExactEigensolver
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
from qiskit.aqua.components.variational_forms import RYRZ, VariationalForm
from qiskit.aqua.components.optimizers import COBYLA, SPSA, SLSQP, AQGD
from qiskit.chemistry.drivers import PySCFDriver, UnitsType
from qiskit.chemistry import FermionicOperator
from qiskit import IBMQ, Aer, execute, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.providers.aer import noise
from qiskit.aqua import QuantumInstance
from qiskit.quantum_info import Pauli
from qiskit.aqua.operators.weighted_pauli_operator import WeightedPauliOperator

#backend = Aer.get_backend("qasm_simulator")

def run(operator, optimizer, varform, backend, num_qubits, initialParameters=None, shotsPerPoint=10000):
    global shots
    global values
    global plottingTime
    global qubitOp
    global qr_size
    global var_form
    global backend
    backend = backend
    var_form = varform
    qr_size = num_qubits
    plottingTime = True
    qubitOp = operator
    shots = shotsPerPoint
    values = []
    numberOfParameters = varform.num_parameters
    if initialParameters is None:
        initialParameters = np.random.rand(numberOfParameters)

    # Start optimization
    sol_opt = optimizer.optimize(numberOfParameters, energy_opt, gradient_function=None,
                                 variable_bounds=None, initial_point=initialParameters)
    return(sol_opt[1])


# Create Parallelized Measurement Circuits from single measurement circuits
def opToCircs (circuit = QuantumCircuit, operator = WeightedPauliOperator, qr_size = int):
    if(qr_size < operator.num_qubits):
        raise Exception('Error: Not enough qubits, enter at least QubitOp.num_qubits qubits.')
    qr = []
    cr = []
    for i in range(len(operator.paulis)):
        qr.append(QuantumRegister(operator.num_qubits))
        cr.append(ClassicalRegister(operator.num_qubits))
    meascircuits = operator.construct_evaluation_circuit(circuit, statevector_mode=False,
                                                        qr=None, cr=None, use_simulator_operator_mode=False,
                                                        circuit_name_prefix='')
    paulis_per_register = math.floor(qr_size/operator.num_qubits)
    numRegisters = math.ceil(len(operator.paulis)/paulis_per_register)
    output_circ = []
    for j in range(numRegisters):
        l = j*paulis_per_register
        for k in range(paulis_per_register-1):
            if(l+k+1<len(qr)):
                meascircuits[l].add_register(qr[j*(paulis_per_register-1)+k+1], cr[j*(paulis_per_register-1)+k+1])
                meascircuits[l].append(meascircuits[l+k+1].to_instruction(),
                                       qr[j*(paulis_per_register-1)+k+1], cr[j*(paulis_per_register-1)+k+1])
        output_circ.append(meascircuits[l].decompose())
    return output_circ


# Function that is used by the optimizer, takes parameters and returns respective energy
def energy_opt(parameters):
    # Create variational form with RYRZ model and build corresponding circuit with parameters
    #var_form = RYRZ(qubitOp.num_qubits, depth=3, entanglement="linear")
    circuit = var_form.construct_circuit(parameters)

    # Calculate and output the energy
    energy = E(circuit, qubitOp, qr_size)
    if plottingTime:
        values.append(energy)
    #print(energy)
    return energy



# Function that calculates energy
def E(circuit=QuantumCircuit, qubitOp=WeightedPauliOperator, qr_size=int):
    # Initialize energy and calculate number of Paulis that fit on quantum device
    energy = 0
    pauli_size = qubitOp.num_qubits
    paulis_per_register = math.floor(qr_size/pauli_size)

    # Call opToCircs to obtain parallelized circuits
    output_circuits = opToCircs(circuit, qubitOp, pauli_size*paulis_per_register)
    counter = 0

    # Sum over the energies of the different measurement circuits
    for circuit in output_circuits:
        job = execute(circuit, backend, shots=shots, optimization_level=3)
        result = job.result()

        # Use the following for noisy calculation:
        # result=quantum_instance.execute(circuit)

        counts = result.get_counts(circuit)

        # Separate dictionaries for the different Pauli Operators
        sep_counts = []
        for key in counts:
            string = []
            for i in range(paulis_per_register):
                if (i*(pauli_size + 1) + pauli_size <= len(key)):
                    string.append(key[i * (pauli_size + 1):i * (pauli_size + 1) + pauli_size])
            string.append(counts.get(key))
            sep_counts.append(string)
        for i in range(len(sep_counts[0])-1):
            newdict = {}
            for j in range(2**pauli_size):
                b ='{0:b}'.format(j)
                b = b.zfill(pauli_size)
                newdict[b]=0
            for k in range(len(sep_counts)):
                newdict[sep_counts[k][len(sep_counts[0])-2-i]] += sep_counts[k][-1]
            energy += qubitOp.paulis[counter*paulis_per_register+i][0] \
                     * sum_binary(newdict, qubitOp.paulis[counter*paulis_per_register+i][1])
        counter += 1
    return energy

# Calculate the energy for a given Pauli operator and measurement outcome given as bitstring
def sum_binary(counts, pauli = Pauli):
    sum = 0
    total = 0
    # countOperator tracks which parts of the Pauli Operator consist of the identity (which does not affect the parity)
    countOperator =  list(np.logical_or(pauli.x, pauli.z))
    for key in counts:
        parity = 0
        counter = 0
        for i in key:
            if int(i) == 1 and countOperator[counter]:
                parity += 1
            counter += 1
        sum += ((-1) ** parity) * counts[key]
        total += counts[key]
    return sum / total

global plottingTime
plottingTime= False
noisy = False

# Noisy Backend
if noisy:
    provider = IBMQ.load_account()
    device = provider.get_backend("ibmq_16_melbourne")
    coupling_map = device.configuration().coupling_map
    noise_model = noise.device.basic_device_noise_model(device.properties())
    quantum_instance = QuantumInstance(backend=backend, shots=100,
                                       noise_model=noise_model, coupling_map=coupling_map)
