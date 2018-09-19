from qiskit import QuantumRegister, QuantumCircuit, execute
import qiskit.extensions.simulator
import ncu1
import ncx

n = 1

q = QuantumRegister(n)
tgt = QuantumRegister(1)

qc = QuantumCircuit(q, tgt)

qc.x(q)
#qc.x(q[0])

qc.ncx([], tgt, global_phase=-1j)

qc.snapshot(1)

res = execute(qc, "local_qasm_simulator", config={"data":
    ["hide_statevector", "quantum_state_ket"]}).result()
print(res.get_snapshot("1").get("quantum_state_ket"))
