# -*- coding: utf-8 -*-

# Copyright 2018 IBM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import sys
import os

qiskit_acqua_chemistry_directory = os.path.dirname(os.path.realpath(__file__))
qiskit_acqua_chemistry_directory = os.path.join(qiskit_acqua_chemistry_directory,'..')
sys.path.insert(0,'qiskit_acqua_chemistry')
sys.path.insert(0,qiskit_acqua_chemistry_directory)
# hack untils qiskit-acqua is installable
qiskit_acqua_directory = os.path.dirname(os.path.realpath(__file__))
qiskit_acqua_directory = os.path.join(qiskit_acqua_directory,'../../qiskit-acqua')
sys.path.append(qiskit_acqua_directory)
# ---

import qiskit_acqua_chemistry

input_min = {
    'driver': {'name': 'PYSCF', 'hdf5_output': 'molecule.hdf5' },
    'PYSCF': {'atom': 'H .0 .0 .0; H .0 .0 0.735', 'unit': 'Angstrom', 'charge': 0, 'spin': 0, 'basis': 'sto3g'}
}

input_vqe = {
    'name': 'A two line description\nof my experiment',
    'driver': {'name':'PYSCF'},
    'PYSCF': {'atom': 'H .0 .0 .0; H .0 .0 0.735', 'unit': 'Angstrom', 'charge': 0, 'spin': 0, 'basis': 'sto3g'},
    "algorithm" : {'name': 'VQE', 'operator_mode': 'matrix'},
    "backend": {'name': 'local_statevector_simulator'}
}

psi4_cfg = """
molecule h2 {
   0 1
   H       .0000000000       0.0          0.0
   H       .0000000000       0.0           .2
}

set {
  basis sto-3g
  scf_type pk
}
"""
input_psi4 = {
    'driver': {'name':'PSI4'},
    'PSI4': psi4_cfg,
    "algorithm": {'name': 'VQE', 'operator_mode': 'paulis'}
}

# Here this is a list of strings one for each line instead of a multiline string
# Was thinking this might be an alternate useful was to supply (e.g. I could
# read the lines from a file or something
psi4_alt_cfg = [
    'molecule h2 {',
    '   0 1',
    '   H       .0000000000       0.0          0.0',
    '   H       .0000000000       0.0           .2',
    '}',
    '',
    'set {',
    '  basis sto-3g',
    '  scf_type pk',
    '}'
]

input_psi4 = {
    'driver': {'name':'PSI4'},
    'PSI4': psi4_alt_cfg,
    "algorithm": {'name': 'VQE', 'operator_mode': 'paulis'}
}

# =============================================================
# An example of using in a loop to vary interatomic distance

distance = 0.5
molecule = 'H .0 .0 -{0}; H .0 .0 {0}'
energies = []
for i in range(100):
    atoms = molecule.format((distance + i*0.5/100)/2) # From 0.5 to 1.0 in steps of 0.5/100. Each atom at half distance - and +
    solver = qiskit_acqua_chemistry.ACQUAChemistry()
    input_loop = {
        'driver': {'name':'PYSCF'},
        'PYSCF': {'atom': atoms, 'unit': 'Angstrom', 'charge': 0, 'spin': 0, 'basis': 'sto3g'},
        "algorithm": {'name': 'ExactEigensolver'},
    }
    e = solver.run(input_loop) # Assumes here this will construct inputparser using dict. No Output specified here
    print(e['energy'])
    energies.append(e['energy'])

# Can now use energies on y-axis of plot where x axis is distance 0.5 to 1.0
