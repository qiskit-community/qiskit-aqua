import numpy as np
from qiskit import Aer
from qiskit.chemistry import QiskitChemistry
import warnings
warnings.filterwarnings('ignore')

# setup qiskit_chemistry logging
import logging
from qiskit.chemistry import set_qiskit_chemistry_logging
set_qiskit_chemistry_logging(logging.ERROR) # choose among DEBUG, INFO, WARNING, ERROR, CRITICAL and NOTSET

# chemistry related modules
from qiskit.chemistry import FermionicOperator
from qiskit.chemistry.drivers import PySCFDriver, UnitsType

# Use PySCF, a classical computational chemistry software
# package, to compute the one-body and two-body integrals in
# molecular-orbital basis, necessary to form the Fermionic operator
driver = PySCFDriver(atom='H .0 .0 .0; H .0 .0 0.735',
                    unit=UnitsType.ANGSTROM,
                    basis='sto3g')
molecule = driver.run()
num_particles = molecule.num_alpha + molecule.num_beta
num_spin_orbitals = molecule.num_orbitals * 2

# Build the qubit operator, which is the input to the VQE algorithm in Aqua
ferOp = FermionicOperator(h1=molecule.one_body_integrals, h2=molecule.two_body_integrals)
map_type = 'PARITY'
qubitOp = ferOp.mapping(map_type)
qubitOp = qubitOp.two_qubit_reduced_operator(num_particles)
num_qubits = qubitOp.num_qubits

# set the backend for the quantum computation
from qiskit import Aer
backend = Aer.get_backend('statevector_simulator')

# setup a classical optimizer for VQE
from qiskit.aqua.components.optimizers import L_BFGS_B
optimizer = L_BFGS_B()

# setup the initial state for the variational form
from qiskit.chemistry.aqua_extensions.components.initial_states import HartreeFock
init_state = HartreeFock(num_qubits, num_spin_orbitals, num_particles)

# setup the variational form for VQE
from qiskit.chemistry.aqua_extensions.components.variational_forms import UCCSD
var_form = UCCSD(num_qubits, 1, num_spin_orbitals, num_particles, initial_state=init_state)

from qiskit.chemistry.aqua_extensions.components.variational_forms.ucc import UCCSD_single_operator
var_form_base = UCCSD_single_operator(num_qubits, 1, num_spin_orbitals, num_particles, [], [], [], init_state, 0)

from qiskit.aqua.algorithms.adaptive import VQEAdapt
algorithm = VQEAdapt(qubitOp, var_form_base, 0.0000001, optimizer, var_form._hopping_ops)
result = algorithm.run(backend)
