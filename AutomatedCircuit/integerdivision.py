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

def subtract(q, a, b, c, z, n):
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

n = 3

inputreg = QuantumRegister(n)
divident = QuantumRegister(n+1)
addbit = QuantumRegister(1)
c = QuantumRegister(1)
z = QuantumRegister(1)