# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister

import sys
sys.path.append('../')
from qiskit_aqua.algorithms.components.reciprocals import Reciprocal
from qiskit_aqua.utils import cnx_na, cnu3
from qiskit_aqua.utils.cnx_no_anc import CNXGate
from qiskit import execute 

from .Long_division_rotation import LongDivision

import numpy as np
import math
import os

n = 3

for i in range(2):
    for j in range(2):
        for k in range(2):
            ev = QuantumRegister(n)
            ANC = QuantumRegister(1)
            qc = QuantumCircuit(ev, ANC)
            if k ==1:
                qc.x(ev[0])
            if j ==1:
                qc.x(ev[1])
            if i ==1:
                qc.x(ev[2])
            
            qd = LongDivision()            
            qc += qd.construct_circuit('random', ev, 4)            

            job = execute(qc, backend = 'local_qasm_simulator_cpp', shots = 1)
            re = job.result()
            print(re.get_counts())