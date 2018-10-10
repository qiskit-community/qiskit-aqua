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
"""
The HHL algorithm.
"""

import logging

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

from qiskit_aqua import QuantumAlgorithm
from qiskit_aqua import get_eigs_instance, get_reciprocal_instance, get_initial_state_instance
import numpy as np
from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer


import qiskit.extensions.simulator

logger = logging.getLogger(__name__)


class HHL(QuantumAlgorithm):
    """The HHL algorithm."""

    PROP_MODE = 'mode'

    HHL_CONFIGURATION = {
        'name': 'HHL',
        'description': 'The HHL Algorithm for Solving Linear Systems of equations',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'hhl_schema',
            'type': 'object',
            'properties': {
                PROP_MODE: {
                    'type': 'string',
                    'oneOf': [
                        {'enum': [
                            'circuit', 
                            'state_tomography',
                            'debug',
                            'swap_test'
                        ]}
                    ],
                    'default': 'circuit'
                }
            },
            'additionalProperties': False
        },
        'problems': ['energy'],
        'depends': ['eigs', 'reciprocal'],
        'defaults': {
            'eigs': {
                'name': 'QPE',
                'num_ancillae': 6,
                'num_time_slices': 50,
                'expansion_mode': 'suzuki',
                'expansion_order': 2,
                'qft': {'name': 'STANDARD'}
            },
            'reciprocal': {
                'name': 'LOOKUP'
            },
        }
    }

    def __init__(self, configuration=None):
        super().__init__(configuration or self.HHL_CONFIGURATION.copy())
        self._matrix = None
        self._invec = None

        self._eigs = None
        self._init_state = None
        self._reciprocal = None

        self._circuit = None
        self._io_register = None
        self._eigenvalue_register = None
        self._ancilla_register = None
        self._success_bit = None

        self._num_q = 0
        self._num_a = 0

        self._mode = None
        self._exact = False
        self._debug = False

        self._ret = {}


    def init_params(self, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: list or tuple of np.ndarray (matrix, vector)
        """
        if algo_input is None:
            raise AlgorithmError("Matrix, Vector instance is required.")
        if not isinstance(algo_input, (list, tuple)):
            raise AlgorithmError("(matrix, vector) pair is required.")
        matrix, invec = algo_input
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        if not isinstance(invec, np.ndarray):
            invec = np.array(invec)

        hhl_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM) or {}
        mode = hhl_params.get(HHL.PROP_MODE)

        # Handle different modes
        if self._backend == "local_qasm_simulator":
            from qiskit.backends.local import QasmSimulatorCpp
            try:
                QasmSimulatorCpp()
                cpp = True
            except FileNotFoundError:
                cpp = False
        
        exact = False
        if mode == 'state_tomography':
            if ((self._backend == "local_statevector_simulator" or 
                    (self._backend == "local_qasm_simulator" and cpp)) and
                    self._execute_config.get("shots") == 1):
                exact = True
                ############### not always
                self._debug = True
                ###############
                import qiskit.extensions.simulator

        if mode == 'debug':
            if self._backend != 'local_qasm_simulator' and not cpp:
                raise AlgorithmError("Debug mode only possible with"
                        "C++ local_qasm_simulator.")
            import qiskit.extensions.simulator
            self._debug = True

        if mode == 'swap_test':
            if self._backend == 'local_statevector_simulator':
                raise AlgorithmError("Measurement requred")
    
        # Initialize eigenvalue finding module
        eigs_params = params.get(QuantumAlgorithm.SECTION_KEY_EIGS) or {}
        eigs = get_eigs_instance(eigs_params["name"])
        eigs.init_params(eigs_params, matrix)

        num_q, num_a = eigs.get_register_sizes()


        # Fix invec for nonhermitian/non 2**n size matrices
        assert(matrix.shape[0] == len(invec), "Check input vector size!")

        tmpvec = np.append(invec, (2**num_q - len(invec))*[0])
        init_state_params = {"name": "CUSTOM"}
        init_state_params["num_qubits"] = num_q
        init_state_params["state_vector"] = tmpvec
        init_state = get_initial_state_instance(init_state_params["name"])
        init_state.init_params(init_state_params)

        # Initialize reciprocal rotation module
        reciprocal_params = params.get(QuantumAlgorithm.SECTION_KEY_RECIPROCAL) or {}
        reciprocal_params["negative_evals"] = eigs._negative_evals
        reciprocal_params["evo_time"] = eigs._evo_time
        reci = get_reciprocal_instance(reciprocal_params["name"])
        reci.init_params(reciprocal_params)

        # Initialize self
        self.init_args(matrix, invec, eigs, init_state, reci, mode, exact, num_q, num_a)


    def init_args(self, matrix, invec, eigs, init_state, reciprocal, mode,
            exact, num_q, num_a):
        self._matrix = matrix
        self._invec = invec
        self._eigs = eigs
        self._init_state = init_state
        self._reciprocal = reciprocal
        self._num_q = num_q
        self._num_a = num_a
        self._mode = mode
        self._exact = exact

       
    def _construct_circuit(self):
        """ Constructing the HHL circuit """

        q = QuantumRegister(self._num_q, name="io")
        qc = QuantumCircuit(q)

        # InitialState
        qc += self._init_state.construct_circuit("circuit", q)

        if self._debug: qc.snapshot("0")

        # EigenvalueEstimation (QPE)
        qc += self._eigs.construct_circuit("circuit", q)
        a = self._eigs._output_register

        if self._debug: qc.snapshot("1")

        # Reciprocal calculation with rotation
        qc += self._reciprocal.construct_circuit("circuit", a)
        s = self._reciprocal._anc

        if self._debug: qc.snapshot("2")

        # Inverse EigenvalueEstimation
        qc += self._eigs.construct_inverse("circuit")

        if self._debug: qc.snapshot("3")

        # Measurement of the ancilla qubit
        if not self._exact:
            c = ClassicalRegister(1)
            qc.add(c)
            qc.measure(s, c)
            self._success_bit = c

        if self._debug: qc.snapshot("-1")

        self._io_register = q
        self._eigenvalue_register = a
        self._ancilla_register = s
        self._circuit = qc

    
    def _exact_simulation(self):
        """ 
        The exact simulation mode: The result of the HHL gets extracted from
        the statevector. Only possible with statevector simulators
        """
        # Handle different backends
        if self._backend == "local_statevector_simulator":
            res = self.execute(self._circuit)
            sv = res.get_statevector()
        elif self._backend == "local_qasm_simulator":
            import qiskit.extensions.simulator
            self._circuit.snapshot("5")
            self._execute_config["config"]["data"] = ["quantum_state_ket"]
            res = self.execute(self._circuit)
            sv = res.get_snapshot("5").get("statevector")[0]

        # Extract output vector
        half = int(len(sv)/2)
        vec = sv[half:half+2**self._num_q]
        self._ret["probability"] = vec.dot(vec.conj())
        vec = vec/np.linalg.norm(vec)
        self._ret["result"] = vec

        # Calculating the fidelity
        theo = np.linalg.solve(self._matrix, self._invec)
        theo = theo/np.linalg.norm(theo)
        self._ret["fidelity"] = abs(theo.dot(vec.conj()))**2
        tmp_vec = self._matrix.dot(vec)
        f1 = np.linalg.norm(self._invec)/np.linalg.norm(tmp_vec)
        f2 = sum(np.angle(self._invec*tmp_vec.conj()))/self._num_q
        self._ret["solution"] = f1*vec*np.exp(-1j*f2)

    
    def _state_tomography(self):
        """
        Extracting the solution vector information via state tomography.
        Inefficient, uses 3**n*shots executions of the circuit.
        """
        # Preparing the state tomography circuits
        import qiskit.tools.qcvv.tomography as tomo
        from qiskit import QuantumProgram
        c = ClassicalRegister(self._num_q)
        self._circuit.add(c)
        qp = QuantumProgram()
        qp.add_circuit("master", self._circuit)
        tomo_set = tomo.state_tomography_set(list(range(self._num_q)))
        tomo_names = tomo.create_tomography_circuits(qp, "master",
                self._io_register, c, tomo_set)
        config = {k: v for k, v in self._execute_config.items() if k != "qobj_id"}

        # Handling the results
        res = qp.execute(tomo_names, backend=self._backend, **config)
        probs = []
        for circ in res._result.get("result"):
            counts = circ.get("data").get("counts")
            new_counts = {}
            s, f = 0, 0
            for k, v in counts.items():
                if k[-1] == "1":
                    new_counts[k[:-2]] = v
                    s += v
                else:
                    f += v
            probs.append(s/(f+s))
            circ["data"]["counts"] = new_counts

        # Fitting the tomography data
        tomo_data = tomo.tomography_data(res, 'master', tomo_set)
        rho_fit = tomo.fit_tomography_data(tomo_data)
        vec = rho_fit[:, 0]/np.sqrt(rho_fit[0, 0])
        self._ret["result"] = vec

        # Calculating the fidelity with the classical solution
        theo = np.linalg.solve(self._matrix, self._invec)
        theo = theo/np.linalg.norm(theo)
        self._ret["fidelity"] = abs(theo.dot(vec.conj()))**2

        # Rescaling the output vector to the real solution vector
        tmp_vec = self._matrix.dot(vec)
        f1 = np.linalg.norm(self._invec)/np.linalg.norm(tmp_vec)
        f2 = sum(np.angle(self._invec*tmp_vec.conj()))/self._num_q
        self._ret["solution"] = f1*vec*np.exp(-1j*f2)


    def _swap_test(self):
        """
        Making a swap test calculating the fidelity between the HHL and
        classical result (normalized).
        """
        # Preparing the circuit
        c = ClassicalRegister(1)
        self._circuit.add(c)

        # using free qubits
        if (self._num_q + 1) > self._num_a:
            qx = QuantumRegister(self._num_q+1-self._num_a)
            self._circuit.add(qx)
            qubits = [qi for qi in self._eigenvalue_register] + [qi for qi in qx]
        else:
            qubits = [self._eigenvalue_register[i] for i in
                    range(self._num_q + 1)]
        test_bit = qubits[0]
        x_state = qubits[1:]

        # Initializeing the solution state vector
        init_state = get_initial_state_instance("CUSTOM")
        sol = list(np.linalg.solve(self._matrix, self._invec))
        init_state.init_params({"num_qubits": self._num_q, "state_vector": sol})

        qc = self._circuit
        qc += init_state.construct_circuit('circuit', x_state)

        # Making a swap test
        qc.h(test_bit)
        for i in range(self._num_q):
            qc.cswap(test_bit, self._io_register[i], x_state[i])
        qc.h(test_bit)

        qc.measure(test_bit, c[0])
        
        # Execution and calculation of the fidelity
        res = self.execute(self._circuit)
        counts = res.get_counts()
        failed = 0
        probs = [0, 0]
        for key, val in counts.items():
            if key[-1] == "1":
                probs[int(key[0])] = val
            else:
                failed += val
        self._ret["probability"] = sum(probs)/(sum(probs)+failed)
        probs = np.array(probs)/sum(probs)
        self._ret["fidelity"] = probs[0]*2-1
        self._ret["probs"] = probs
        self._ret["solution"] = sol
        self._ret["counts"] = res.get_counts()


    def __filter(self, qsk, reg=None, qubits=None):
        # WORK IN PROGRESS
        qregs = list(self._circuit.get_qregs().values())
        if reg:
            idx = qregs.index(reg)
            none = 0
            for i in range(idx):
                none += len(qregs[i]) + 1
            mask = list(range(none, none+len(reg)))
        if qubits:
            mask = []
            for qubit in qubits:
                idx = qregs.index(qubit[0])
                none = 0
                for i in range(idx):
                    none += len(qregs[i]) + 1
                mask += [none + qubit[1]]
        ret = {}
        mask = np.array(mask)
        for key, val in qsk.items():
            nkey = "".join(np.array(list(key))[len(key)-1-mask])
            if nkey in ret:
                ret[nkey] += complex(*val)
            else:
                ret[nkey] = complex(*val)
        n = np.linalg.norm(np.array(list(ret.values())))
        ret = {k: v/n for k, v in ret.items()}
        return ret

    #####################################################
    def _exec_debug(self):
        # WORK IN PROGRESS
        print(" HHL - Debug Mode ")
        print("##################\n")
        print("Matrix:\t", str(self._matrix).replace("\n", "\n\t "))
        w, v = np.linalg.eig(self._matrix)
        print("-> Eigenvalues", w)
        print("-> Eigenvectors", *v)
        print("-> Condition", np.linalg.cond(self._matrix))

        print("Input:", self._invec)

        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1])
        fig = plt.figure(figsize=(16,7))
        ax_eig = plt.subplot(gs[0])
        ax_rec = plt.subplot(gs[1])
        ax_dev = plt.subplot(gs[2])

        self._execute_config["config"] = {"noise_params": None,
                "data": ["quantum_state_ket"]}
        self._execute_config["shots"] = 1
        res = self.execute(self._circuit)

        # Plot eigenvalues
        eigs = res.get_snapshot("1").get("quantum_state_ket")[0]
        eigs = self.__filter(eigs, reg=self._eigenvalue_register)
        nums = np.array(list(map(lambda x: int(x, 2), eigs.keys())))
        nums = nums/2**self._num_a*2*np.pi/self._eigs._evo_time
        vals = np.array(list(eigs.values()))
        ax_eig.bar(nums, abs(vals)**2, width=2*np.pi/self._eigs._evo_time
                / 2**self._num_a)

        # Theoretical eigenvalues
        w, v = np.linalg.eig(self._matrix)
        vt = v.T.conj().dot(self._invec)
         
        ty = np.arange(0, 2**self._num_a)
        tmp = 1j*(2**self._num_a*np.outer(w, np.ones(len(ty)))*self._eigs._evo_time - 
                2*np.pi*np.outer(np.ones(len(w)), ty))
        tmp[tmp == 0] = 2j*np.pi/self._eigs._evo_time/2**self._num_a
        ty = np.abs(vt.dot((1-np.exp(tmp))/(1-np.exp(tmp/2**self._num_a))
                * 2**-self._num_a))**2
        tx = np.arange(0, 2**self._num_a)/2**self._num_a*2*np.pi/self._eigs._evo_time
        ty /= sum(ty)

        if self._eigs._negative_evals:
            h = int(len(tx)/2)
            tx1 = tx[:h]
            tx2 = tx[h:]
            tx2 -= 2*tx2[0]
            tx = np.concatenate(tx2, tx1)
            ty = np.concatenate(ty[h:], ty[:h])

        ax_eig.plot(tx, ty, "r")
        
        # Plot reciprocals
        rec = res.get_snapshot("2").get("quantum_state_ket")[0]
        list(map(lambda x: print(x[0], x[1]), rec.items()))
        rec = self.__filter(rec, qubits=[qi for qi in self._eigenvalue_register]
                + [self._ancilla_register[0]])
        list(map(lambda x: print(x[0], x[1]), rec.items()))
        getrec = lambda s: rec[s] if s in rec else 0
        rec = [[getrec(key+"0"), getrec(key+"1")] for key in eigs.keys()]
        y = [np.linalg.norm(i)**2 for i in rec]
        x = [i[1]/np.linalg.norm(i) for i in rec]
        ax_rec.scatter(x, y)

        # Plot theoretical reciprocals (dependend on QPE results)
        tx = np.arange(0, 2**self._num_a)/2**self._num_a
        tx = self._reciprocal._scale/tx
        ax_rec.plot(tx, ty, "r")

        # Solution deviation
        sv = res.get_snapshot("3").get("statevector")[0]
        half = int(len(sv)/2)
        vec = sv[half:half+2**self._num_q]
        self._ret["probability"] = vec.dot(vec.conj())
        vec = vec/np.linalg.norm(vec)
        self._ret["result"] = vec
        solution = np.linalg.solve(self._matrix, self._invec)
        self._ret["fidelity"] = abs(vec.conj().dot(solution/np.linalg.norm(solution)))**2
        print(self._ret["fidelity"])
        dev = np.abs(solution/np.linalg.norm(solution)-vec)**2
        ax_dev.barh(np.arange(len(dev)), dev)
        sa = solution/np.linalg.norm(solution)
        print(sa, vec)
        print(abs(sa[0]*vec[0].conj()+sa[1]*vec[1].conj())**2)

        # Decoration
        ax_eig.set_title("Eigenvalue results")
        ax_eig.set_ylabel("Probability")
        ax_eig.set_xlabel("Value $\lambda$")
        ax_eig.set_xlim(0, 2.1*np.pi/self._eigs._evo_time)
        ax_eig.axvline(x=self._reciprocal._scale*2*np.pi/self._eigs._evo_time,
                color="g")

        ax_rec.set_title("Reciprocal results")
        ax_rec.set_ylabel("Probability")
        ax_rec.set_xlabel("$C/\lambda$")
        ax_rec.set_xlim(0, 1)

        ax_dev.set_title("Deviation of Solution")
        ax_dev.set_xlabel("Amount")
        ax_dev.set_ylabel("Entry")
        lim = max(abs(dev)*1.5)
        ax_dev.set_xlim(0, lim)

        plt.show()
    #################################### 

    def run(self):
        self._construct_circuit()
        # Handling the modes
        if self._mode == "circuit":
            self._ret["circuit"] = self._circuit
            regs = {
                "io_register": self._io_register, 
                "eigenvalue_register": self._eigenvalue_register,
                "ancilla_register": self._ancilla_register,
                "self._success_bit": self._success_bit
            }
            self._ret["regs"] = regs
        elif self._mode == "state_tomography":
            if self._exact:
                sv = self._exact_simulation()
            else:
                self._state_tomography()
        elif self._mode == "debug":
            self._exec_debug()
        elif self._mode == "swap_test":
            self._swap_test()
        # Adding few general informations
        self._ret["gate_count"] = self._circuit.number_atomic_gates()
        self._ret["matrix"] = self._matrix
        self._ret["invec"] = self._invec
        self._ret["eigenvalues"] = np.linalg.eig(self._matrix)[0]
        return self._ret

    ############################################### 
    def number_atomic_gates_2(self, qc=None):
        from qiskit import CompositeGate,Gate
        from qiskit.extensions.standard.ccx import ToffoliGate
        from qiskit.extensions.standard.cu1 import Cu1Gate
        from qiskit.extensions.standard.cu3 import Cu3Gate
        from qiskit.extensions.standard.cswap import FredkinGate
        from qiskit.extensions.standard.cy import CyGate
        from qiskit.extensions.standard.cz import CzGate
        from qiskit.extensions.standard.ch import CHGate
        from qiskit.extensions.standard.crz import CrzGate

        """Count the number of leaf gates. """
        #worth 6 basic gates
        gate_list=[Cu1Gate,Cu3Gate,FredkinGate,CyGate,CzGate,CHGate,CrzGate]
        num = 0
        if qc == None:
            qc = self._circuit
        for gate in qc.data:
            if isinstance(gate, CompositeGate):
                num += self.number_atomic_gates_2(gate)
            else:
                if isinstance(gate, Gate):
                    if isinstance(gate,ToffoliGate):
                        num+=15
                    inlist = [isinstance(gate,i) for i in gate_list]
                    if any(inlist):
                        num+=5
                    num += 1
        return num
    ##############################################


