import matplotlib.pyplot as plt
import numpy as np
import qiskit.extensions.simulator
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, execute
from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer
from scipy import linalg

from qiskit_aqua import (Operator, get_eigs_instance,
                         get_initial_state_instance, get_iqft_instance,
                         get_qft_instance, get_reciprocal_instance,
                         run_algorithm)
from qiskit_aqua.input import get_input_instance
from qiskit_aqua.utils import random_hermitian


class HHLDEMO():
    def __init__(self):
        self._matrix = None
        self._invec = None
        self._eigs = None
        self._init_state = None
        self._reciprocal = None
        self._evo_time = None
        self._negative_evals = False
        self._operator = None

        self._qpe_circuit = None
        self._circuit = None
        self._io_register = None
        self._eigenvalue_register = None
        self._ancilla_register = None
        self._ne_qfts = [None, None]

        self._num_q = 0
        self._num_a = 0

        self._mode = None

    def getstate(self, name,invec):
        num_q = int(np.log2(len(invec)))
        self._num_q = num_q
        self._invec = invec
        init_state_params = {"name": "CUSTOM", "num_qubits": num_q, "state_vector": self._invec}
        init_state = get_initial_state_instance(init_state_params["name"])
        init_state.init_params(init_state_params)
        q = QuantumRegister(num_q)
        qc = QuantumCircuit(q)
        qc = init_state.construct_circuit("circuit", q)
        self._circuit = qc
        self._io_register = q
        return(qc, q)

    def getoperator(self, matrix):
        self._matrix = matrix
        self._operator = Operator(matrix=self._matrix)
        return(self._operator)

    def construct_qpe_circuit(self, num_time_slices, num_ancillae, expansion_mode, expansion_order, negative_evals):
        paulis_grouping = 'random'
        self._negative_evals = negative_evals
        if self._negative_evals:
            num_ancillae += 1
        iqft_params = {}
        iqft_params['num_qubits'] = num_ancillae
        iqft_params['name'] = "STANDARD"
        iqft = get_iqft_instance(iqft_params['name'])
        iqft.init_params(iqft_params)
        if self._negative_evals:
            ne_qft_params = iqft_params
            ne_qft_params['num_qubits'] -= 1
            ne_qfts = [ get_qft_instance(ne_qft_params['name']),
                    get_iqft_instance(ne_qft_params['name'])]
            ne_qfts[0].init_params(ne_qft_params)
            ne_qfts[1].init_params(ne_qft_params)
        else:
            ne_qfts = [None, None]

        self._operator._check_representation('paulis')
        paulis = self._operator.paulis
        if self._evo_time == None:
            lmax = sum([abs(p[0]) for p in self._operator.paulis])
            if not self._negative_evals:
                self._evo_time = (1-2**-num_ancillae)*2*np.pi/lmax
            else:
                self._evo_time = (1/2-2**-num_ancillae)*2*np.pi/lmax

            # check for identify paulis to get its coef for applying global phase shift on ancillae later
        num_identities = 0
        for p in self._operator.paulis:
            if np.all(p[1].v == 0) and np.all(p[1].w == 0):
                num_identities += 1
                if num_identities > 1:
                    raise RuntimeError('Multiple identity pauli terms are present.')
                ancilla_phase_coef = p[0].real if isinstance(p[0], complex) else p[0]
        
        a = QuantumRegister(num_ancillae)
        q = self._io_register

        qc = QuantumCircuit(a, q)

        # Put all ancillae in uniform superposition
        qc.u2(0, np.pi, a)

        # phase kickbacks via dynamics
        pauli_list = self._operator.reorder_paulis(grouping=paulis_grouping)
        if len(pauli_list) == 1:
            slice_pauli_list = pauli_list
        else:
            if expansion_mode == 'trotter':
                slice_pauli_list = pauli_list
            elif expansion_mode == 'suzuki':
                slice_pauli_list = Operator._suzuki_expansion_slice_pauli_list(
                    pauli_list,
                    1,
                    expansion_order
                )
            else:
                raise ValueError('Unrecognized expansion mode {}.'.format(expansion_mode))
        for i in range(num_ancillae):
            qc += self._operator.construct_evolution_circuit(
                slice_pauli_list, -self._evo_time, num_time_slices, q, a,
                ctl_idx=i, use_basis_gates=True
            )
            # global phase shift for the ancilla due to the identity pauli term
            if ancilla_phase_coef != 0:
                qc.u1(self._evo_time * ancilla_phase_coef * (2 ** i), a[i])

        # inverse qft on ancillae
        iqft.construct_circuit('circuit', a, qc)

        # handle negative eigenvalues
        if self._negative_evals:
            sgn = a[0]
            qs = [a[i] for i in range(1, len(a))]
            for qi in qs:
                qc.cx(sgn, qi)
            ne_qfts[0].construct_circuit('circuit', qs, qc)
            for i, qi in enumerate(reversed(qs)):
                qc.cu1(2*np.pi/2**(i+1), sgn, qi)
            ne_qfts[1].construct_circuit('circuit', qs, qc)

        circuit = qc
        output_register = a
        input_register = q
        self._qpe_circuit = qc
        self._io_register = q
        self._eigenvalue_register = a
        self._circuit += circuit

    def construct_reciprocal_circuit(self, mode):
        reciprocal_params = {}
        reciprocal_params["negative_evals"] = False
        reciprocal_params["scale"] = 1
        reciprocal_params["evo_time"] = self._evo_time
        reciprocal_params["name"] = mode
        reciprocal = get_reciprocal_instance(reciprocal_params["name"])
        reciprocal.init_params(reciprocal_params)

        qc = reciprocal.construct_circuit("circuit", self._eigenvalue_register)
        anc = reciprocal._anc
        self._circuit += qc

    def construct_inverse_circuit(self):
        circuit = self._qpe_circuit #QuantumCircuit(input_register, output_register)
        qc = QuantumCircuit(self._io_register, self._eigenvalue_register)
        for gate in reversed(circuit.data):
            gate.reapply(qc)
            qc.data[-1].inverse()
        inverse = qc
        self._circuit += qc

    def result(self, backend):
        ret = {}
        if backend == "local_statevector_simulator":
            res = execute(self._circuit, backend=backend).result()
            sv = res.get_statevector()        
        elif backend == "local_qasm_simulator":
            import qiskit.extensions.simulator
            self._circuit.snapshot("-1")
            res = execute(self._circuit, backend=backend).result()
            sv = res.get_snapshot("-1")
        half = int(len(sv)/2)
        vec = sv[half:half+2**self._num_q]
        ret["probability"] = vec.dot(vec.conj())
        vec = vec/np.linalg.norm(vec)
        ret["result"] = vec
        theo = np.linalg.solve(self._matrix, self._invec)
        theo = theo/np.linalg.norm(theo)
        ret["fidelity"] = abs(theo.dot(vec.conj()))**2
        tmp_vec = self._matrix.dot(vec)
        f1 = np.linalg.norm(self._invec)/np.linalg.norm(tmp_vec)
        f2 = sum(np.angle(self._invec*tmp_vec.conj()))/self._num_q
        ret["solution"] = f1*vec*np.exp(-1j*f2)
        ret["gate_count"] = self._circuit.number_atomic_gates()
        ret["matrix"] = self._matrix
        ret["invec"] = self._invec
        ret["eigenvalues"] = np.linalg.eig(self._matrix)[0]
        return ret, self._circuit
