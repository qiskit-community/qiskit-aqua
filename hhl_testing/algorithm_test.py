from qiskit import QuantumRegister, QuantumCircuit, execute
import qiskit.extensions.simulator

from qiskit_aqua import Operator, run_algorithm
from qiskit_aqua.input import get_input_instance
from qiskit_aqua import get_eigs_instance, get_reciprocal_instance

from qiskit_aqua.algorithms.single_sample.hhl.hhl import HHL

from qiskit_aqua.utils import random_hermitian, random_non_hermitian

from qiskit.tools.visualization import plot_circuit

# from qiskit_aqua.algorithms.components.reciprocals.lookup_rotation import LookupRotation

params = {
    "evo_time": 1,
    "reg_size": 4,
    "pat_length": 3,
    "subpat_length": 2
}

def init_int(num, qc, q):
    s = "{:b}".format(num).rjust(len(q), "0")
    for i, b in enumerate(reversed(s)):
        if b == "1":
            qc.x(q[i])

for i in range(1, 2**params["reg_size"]):
    rp = get_reciprocal_instance("LOOKUP_ROT")
    rp.init_params(params)
    q = QuantumRegister(params["reg_size"])
    qc = QuantumCircuit(q)
    init_int(i, qc, q)
    qc += rp.construct_circuit("circuit", q)
    qc.snapshot("1")

    res = execute(qc, "local_qasm_simulator", config={"data":
        ["quantum_state_ket"]}, shots=1).result()
    print(res.get_snapshot("1").get("quantum_state_ket"))

# params = {
#     "algorithm": {
#         "name": "HHL"
#     },
#     "eigs": {
#         "name": "QPE",
#         "num_time_slices": 1,
#         "expansion_mode": "trotter",
#         "negative_evals": False,
#         "num_ancillae": 3,
#     }
# }
#
# matrix = random_hermitian(2, trunc=2)
#
# qc = run_algorithm(params, matrix)
