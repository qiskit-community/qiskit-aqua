import sys
import os

qiskit_acqua_directory = os.path.dirname(os.path.realpath(__file__))
qiskit_acqua_directory = os.path.join(qiskit_acqua_directory, '..')
sys.path.insert(0, 'qiskit_acqua')
sys.path.insert(0, qiskit_acqua_directory)
