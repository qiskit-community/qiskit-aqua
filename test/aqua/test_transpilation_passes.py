import unittest

from qiskit.test import ReferenceCircuits
from qiskit import transpile, QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler import PassManager

from qiskit.aqua.components.extrapolation_pass_managers import RedundantCNOT
from test.aqua.common import QiskitAquaTestCase

class TranspilationPassesTest(unittest.TestCase):
    def setUp(self):
        pass

    def test_redundant_cnot_insertion(self):
        circ = ReferenceCircuits.bell()
        pm = PassManager(passes=RedundantCNOT(1))
        result = transpile([circ], pass_manager=pm)

        qr = QuantumRegister(2, name='qr')
        cr = ClassicalRegister(2, name='qc')
        qc = QuantumCircuit(qr, cr, name='bell')
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[1])
        qc.measure(qr, cr)

        self.assertEqual(result, qc)

    def test_redundant_cnot_insertion_reversed(self):
        pm = PassManager(passes=RedundantCNOT(1))
        qr = QuantumRegister(2, name='qr')
        cr = ClassicalRegister(2, name='qc')

        qc = QuantumCircuit(qr, cr, name='bell')
        qc.h(qr[0])
        qc.cx(qr[1], qr[0])
        qc.cx(qr[1], qr[0])
        qc.cx(qr[1], qr[0])
        qc.measure(qr, cr)

        qc_1 = QuantumCircuit(qr, cr, name='bell')
        qc_1.h(qr[0])
        qc_1.cx(qr[1], qr[0])
        qc_1.measure(qr, cr)

        result = transpile([qc_1], pass_manager=pm)

        self.assertEqual(result, qc)

        qc = QuantumCircuit(qr, cr, name='bell')
        qc.h(qr[0])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[1])
        qc.cx(qr[0], qr[1])
        qc.measure(qr, cr)

        qc_1 = QuantumCircuit(qr, cr, name='bell')
        qc_1.h(qr[0])
        qc_1.cx(qr[0], qr[1])
        qc_1.measure(qr, cr)

        result = transpile([qc_1], pass_manager=pm)

        self.assertEqual(result, qc)

if __name__ == '__main__':
    unittest.main()
