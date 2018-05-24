import sys
import os

qiskit_acqua_chemistry_directory = os.path.dirname(os.path.realpath(__file__))
qiskit_acqua_chemistry_directory = os.path.join(qiskit_acqua_chemistry_directory, '..')
sys.path.insert(0, 'qiskit_acqua_chemistry')
sys.path.insert(0, qiskit_acqua_chemistry_directory)
# hack untils qiskit-acqua is installable
qiskit_acqua_directory = os.path.dirname(os.path.realpath(__file__))
qiskit_acqua_directory = os.path.join(qiskit_acqua_directory,'../../qiskit-acqua')
sys.path.insert(0,qiskit_acqua_directory)
# ---
