from qiskit import register, available_backends, get_backend, least_busy
from qiskit import ClassicalRegister, QuantumRegister
from qiskit import QuantumCircuit, execute
from qiskit.tools.visualization import plot_histogram, matplotlib_circuit_drawer as drawer
import matplotlib.pyplot as plt

def maj(p, a, b, c):
    p.cx(c, b)
    p.cx(c, a)
    p.ccx(a, b, c)

def uma(p, a, b, c):
    p.ccx(a, b, c)
    p.cx(c, a)
    p.cx(a, b)

n = 3

a = QuantumRegister(n)
b = QuantumRegister(n)
z = QuantumRegister(1)
c = QuantumRegister(1)
measa = ClassicalRegister(n)
measb = ClassicalRegister(n)
measc = ClassicalRegister(1)
measz = ClassicalRegister(1)

qc = QuantumCircuit(a, b, c, z, measa, measb, measc, measz)

qc.x(a[2])
qc.x(b[1])
qc.x(a[0])
qc.x(b[2])

for i in range(n):
    qc.x(a[i])

maj(qc, c[0], b[0], a[0])

for i in range(n-1):
    maj(qc, a[i], b[i+1], a[i+1])

qc.cx(a[n-1], z)

for i in range(1, n):
    uma(qc, a[n-i-1], b[n-i], a[n-i])

uma(qc, c[0], b[0], a[0])

for i in range(n):
    qc.x(a[i])
    qc.x(b[i])

qc.measure(a, measa)
qc.measure(b, measb)
qc.measure(c, measc)
qc.measure(z, measz)

job = execute(qc, backend='local_qasm_simulator')
results = job.result()
print(results.get_counts())
#plot_histogram(results.get_counts())
#drawer(qc)
#plt.show()