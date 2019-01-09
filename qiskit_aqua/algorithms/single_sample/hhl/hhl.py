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
import numpy as np

from qiskit.providers.aer.backends import QasmSimulator
from qiskit.extensions.simulator import snapshot
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit_aqua.algorithms import QuantumAlgorithm
from qiskit_aqua import AquaError, PluggableType, get_pluggable_class

logger = logging.getLogger(__name__)


class HHL(QuantumAlgorithm):
    """The HHL algorithm."""

    PROP_MODE = 'mode'

    CONFIGURATION = {
        'name': 'HHL',
        'description': 'The HHL Algorithm for Solving Linear Systems of '
                       'equations',
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
        'problems': ['linear_system'],
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
                'name': 'Lookup'
            },
        }
    }

    def __init__(self, matrix=None, vector=None, eigs=None, init_state=None,
                 reciprocal=None, mode='circuit', num_q=0, num_a=0):
        super().__init__()
        super().validate({
            HHL.PROP_MODE: mode
        })
        self._matrix = matrix
        self._vector = vector
        self._eigs = eigs
        self._init_state = init_state
        self._reciprocal = reciprocal
        self._mode = mode
        self._num_q = num_q
        self._num_a = num_a
        self._circuit = None
        self._io_register = None
        self._eigenvalue_register = None
        self._ancilla_register = None
        self._success_bit = None
        self._ret = {}

    @classmethod
    def init_params(cls, params, algo_input):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
            algo_input: LinearSystemInput instance
        """
        if algo_input is None:
            raise AquaError("LinearSystemInput instance is required.")
        matrix = algo_input.matrix
        vector = algo_input.vector
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)

        # extending the matrix and the vector
        if np.log2(matrix.shape[0]) % 1 != 0:
            next_higher = int(np.ceil(np.log2(matrix.shape[0])))
            new_matrix = np.identity(2**next_higher)
            new_matrix = np.array(new_matrix, dtype=complex)
            new_matrix[:matrix.shape[0], :matrix.shape[0]] = matrix[:, :]
            matrix = new_matrix

            new_vector = np.ones((1, 2**next_higher))
            new_vector[0, :vector.shape[0]] = vector
            vector = new_vector.reshape(np.shape(new_vector)[1])

        hhl_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM) or {}
        mode = hhl_params.get(HHL.PROP_MODE)

        # Fix vector for nonhermitian/non 2**n size matrices
        if matrix.shape[0] != len(vector):
            raise ValueError("Input vector dimension does not match input "
                             "matrix dimension!")

        # Initialize eigenvalue finding module
        eigs_params = params.get(QuantumAlgorithm.SECTION_KEY_EIGS) or {}
        eigs = get_pluggable_class(PluggableType.EIGENVALUES,
                                   eigs_params['name']).init_params(eigs_params, matrix)
        num_q, num_a = eigs.get_register_sizes()

        # Initialize initial state module
        tmpvec = vector
        init_state_params = {"name": "CUSTOM"}
        init_state_params["num_qubits"] = num_q
        init_state_params["state_vector"] = tmpvec
        init_state = get_pluggable_class(PluggableType.INITIAL_STATE,
                                         init_state_params['name']).init_params(init_state_params)

        # Initialize reciprocal rotation module
        reciprocal_params = \
            params.get(QuantumAlgorithm.SECTION_KEY_RECIPROCAL) or {}
        reciprocal_params["negative_evals"] = eigs._negative_evals
        reciprocal_params["evo_time"] = eigs._evo_time
        reci = get_pluggable_class(PluggableType.RECIPROCAL,
                                   reciprocal_params['name']).init_params(reciprocal_params)

        return cls(matrix, vector, eigs, init_state, reci, mode, num_q, num_a)

    def _handle_modes(self):
        # Handle different modes
        try:
            QasmSimulator()
            cpp = True
        except FileNotFoundError:
            cpp = False
        exact = False
        debug = False
        if self._mode == 'state_tomography':
            if self._quantum_instance.is_statevector_backend(
                    self._quantum_instance._backend):
                exact = True
        if self._mode == 'debug':
            if not cpp:
                raise AquaError("Debug mode only possible with C++ simulator.")
            debug = True
        if self._mode == 'swap_test':
            if self._quantum_instance.is_statevector_backend(
                    self._quantum_instance._backend):
                raise AquaError("Measurement required")
        self._debug = debug
        self._exact = exact

    def _construct_circuit(self):
        """ Constructing the HHL circuit """

        q = QuantumRegister(self._num_q, name="io")
        qc = QuantumCircuit(q)

        # InitialState
        qc += self._init_state.construct_circuit("circuit", q)

        if self._debug:
            qc.snapshot("0")

        # EigenvalueEstimation (QPE)
        qc += self._eigs.construct_circuit("circuit", q)
        a = self._eigs._output_register

        if self._debug:
            qc.snapshot("1")

        # Reciprocal calculation with rotation
        qc += self._reciprocal.construct_circuit("circuit", a)
        s = self._reciprocal._anc

        if self._debug:
            qc.snapshot("2")

        # Inverse EigenvalueEstimation
        qc += self._eigs.construct_inverse("circuit")

        if self._debug:
            qc.snapshot("3")

        # Measurement of the ancilla qubit
        if not self._exact:
            c = ClassicalRegister(1)
            qc.add_register(c)
            qc.measure(s, c)
            self._success_bit = c

        if self._debug:
            qc.snapshot("-1")

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
        if self._quantum_instance.is_statevector:
            res = self._quantum_instance.execute(self._circuit)
            sv = res.get_statevector()
        elif self._quantum_instance.backend_name == "qasm_simulator":
            self._circuit.snapshot("5")
            res = self._quantum_instance.execute(self._circuit)
            sv = res.data(0)['snapshots']['5']['statevector'][0]

        # Extract output vector
        half = int(len(sv)/2)
        vec = sv[half:half+2**self._num_q]
        self._ret["probability_result"] = vec.dot(vec.conj())
        vec = vec/np.linalg.norm(vec)
        self._ret["result_hhl"] = vec
        # Calculating the fidelity
        theo = np.linalg.solve(self._matrix, self._vector)
        theo = theo/np.linalg.norm(theo)
        self._ret["fidelity_hhl_to_classical"] = abs(theo.dot(vec.conj()))**2
        tmp_vec = self._matrix.dot(vec)
        f1 = np.linalg.norm(self._vector)/np.linalg.norm(tmp_vec)
        f2 = sum(np.angle(self._vector*tmp_vec.conj()-1+1))/self._num_q # "-1+1" to fix angle error for -0.+0.j
        self._ret["solution_scaled"] = f1*vec*np.exp(-1j*f2)

    def _state_tomography(self):
        """
        Extracting the solution vector information via state tomography.
        Inefficient, uses 3**n*shots executions of the circuit.
        """
        # Preparing the state tomography circuits
        import qiskit.tools.qcvv.tomography as tomo
        from qiskit import QuantumCircuit
        from qiskit import execute

        c = ClassicalRegister(self._num_q)
        self._circuit.add_register(c)
        qc = QuantumCircuit(c, name="master")
        tomo_set = tomo.state_tomography_set(list(range(self._num_q)))
        tomo_circuits = \
            tomo.create_tomography_circuits(qc, self._io_register, c, tomo_set)

        # Handling the results
        result = self._quantum_instance.execute(tomo_circuits)
        probs = []
        for circ in tomo_circuits:
            counts = result.get_counts(circ)
            s, f = 0, 0
            for k, v in counts.items():
                if k[-1] == "1":
                    s += v
                else:
                    f += v
            probs.append(s/(f+s))
        self._ret["probability_result"] = probs

        # Fitting the tomography data
        tomo_data = tomo.tomography_data(result, "master", tomo_set)
        rho_fit = tomo.fit_tomography_data(tomo_data)
        vec = rho_fit[:, 0]/np.sqrt(rho_fit[0, 0])
        self._ret["result_hhl"] = vec
        # Calculating the fidelity with the classical solution
        theo = np.linalg.solve(self._matrix, self._vector)
        theo = theo/np.linalg.norm(theo)
        self._ret["fidelity_hhl_to_classical"] = abs(theo.dot(vec.conj()))**2
        # Rescaling the output vector to the real solution vector
        tmp_vec = self._matrix.dot(vec)
        f1 = np.linalg.norm(self._vector)/np.linalg.norm(tmp_vec)
        f2 = sum(np.angle(self._vector*tmp_vec.conj()-1+1))/self._num_q # "-1+1" to fix angle error for -0.+0.j
        self._ret["solution_scaled"] = f1*vec*np.exp(-1j*f2)

    def _swap_test(self):
        """
        Making a swap test calculating the fidelity between the HHL and
        classical result (normalized).
        """
        # Preparing the circuit
        c = ClassicalRegister(1)
        self._circuit.add_register(c)

        # using free qubits
        if (self._num_q + 1) > self._num_a:
            qx = QuantumRegister(self._num_q+1-self._num_a)
            self._circuit.add_register(qx)
            qubits = [qi for qi in self._eigenvalue_register] + \
                     [qi for qi in qx]
        else:
            qubits = [self._eigenvalue_register[i] for i in
                      range(self._num_q + 1)]
        test_bit = qubits[0]
        x_state = qubits[1:]

        # Initializing the solution state vector
        init_state_params = {"name": "CUSTOM"}
        sol = list(np.linalg.solve(self._matrix, self._vector))
        init_state_params["num_qubits"] = self._num_q
        init_state_params["state_vector"] = sol
        init_state = get_pluggable_class(PluggableType.INITIAL_STATE,
                                         init_state_params['name']).init_params(init_state_params)

        qc = self._circuit
        qc += init_state.construct_circuit('circuit', x_state)

        # Making a swap test
        qc.h(test_bit)
        for i in range(self._num_q):
            qc.cswap(test_bit, self._io_register[i], x_state[i])
        qc.h(test_bit)

        qc.measure(test_bit, c[0])

        # Execution and calculation of the fidelity
        res = self._quantum_instance.execute(self._circuit)
        counts = res.get_counts()
        failed = 0
        probs = [0, 0]
        for key, val in counts.items():
            if key[-1] == "1":
                probs[int(key[0])] = val
            else:
                failed += val
        self._ret["probability_result"] = sum(probs)/(sum(probs)+failed)
        probs = np.array(probs)/sum(probs)
        self._ret["fidelity_hhl_to_classical"] = probs[0]*2-1
        self._ret["solution_scaled"] = sol
        self._ret["result_counts"] = res.get_counts()

    #####################################################

    def __filter(self, qsk, reg=None, qubits=None):
        # WORK IN PROGRESS
        qregs = self._circuit.qregs
        mask = []
        if reg:
            idx = qregs.index(reg)
            none = 0
            for i in range(idx):
                none += len(qregs[i]) + 1
            mask = list(range(none, none+len(reg)))
        if qubits:
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

    def _exec_debug(self):
        # WORK IN PROGRESS
        print(" HHL - Debug Mode ")
        print("##################\n")
        print("Matrix:\t", str(self._matrix).replace("\n", "\n\t "))
        w, v = np.linalg.eig(self._matrix)
        print("-> Eigenvalues", w)
        print("-> Eigenvectors", *v)
        print("-> Condition", np.linalg.cond(self._matrix))

        print("Input:", self._vector)

        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 1])
        fig = plt.figure(figsize=(16, 7))
        ax_eig = plt.subplot(gs[0])
        ax_rec = plt.subplot(gs[1])
        ax_dev = plt.subplot(gs[2])

        res = self._quantum_instance.execute(self._circuit)

        # Plot eigenvalues
        eigs = res.data(0)['snapshots']['1']['quantum_state_ket'][0]
        eigs = self.__filter(eigs, reg=self._eigenvalue_register)
        nums = np.array(list(map(lambda x: int(x, 2), eigs.keys())))
        nums = nums/2**self._num_a*2*np.pi/self._eigs._evo_time
        vals = np.array(list(eigs.values()))
        ax_eig.bar(nums, abs(vals)**2,
                   width=2*np.pi/self._eigs._evo_time/2**self._num_a)

        # Theoretical eigenvalues
        w, v = np.linalg.eig(self._matrix)
        vt = v.T.conj().dot(self._vector)

        ty = np.arange(0, 2**self._num_a)
        tmp = 1j*(2**self._num_a*np.outer(w, np.ones(len(ty))) *
                  self._eigs._evo_time - 2*np.pi*np.outer(np.ones(len(w)), ty))
        tmp[tmp == 0] = 2j*np.pi/self._eigs._evo_time/2**self._num_a
        ty = np.abs(vt.dot((1-np.exp(tmp))/(1-np.exp(tmp/2**self._num_a)) *
                           2**-self._num_a))**2
        tx = np.arange(0, 2**self._num_a)/2**self._num_a * \
             2*np.pi/self._eigs._evo_time

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
        rec = res.data(0)['snapshots']['2']['quantum_state_ket'][0]
        list(map(lambda x: print(x[0], x[1]), rec.items()))
        rec = self.__filter(rec, qubits=[qi for qi in self._eigenvalue_register] +
                                        [self._ancilla_register[0]])
        list(map(lambda x: print(x[0], x[1]), rec.items()))
        getrec = lambda s: rec[s] if s in rec else 0
        rec = [[getrec(key+"0"), getrec(key+"1")] for key in eigs.keys()]
        y = [np.linalg.norm(i)**2 for i in rec]
        x = [i[1]/np.linalg.norm(i) for i in rec]
        ax_rec.scatter(np.abs(x), np.abs(y))

        # Plot theoretical reciprocals (derived from QPE results)
        tx = np.arange(0, 2**self._num_a)/(2**(self._num_a+1))
        ax_rec.plot(tx, ty, "r")

        # Solution deviation
        sv = res.data(0)['snapshots']['3']['statevector'][0]
        half = int(len(sv)/2)
        vec = sv[half:half+2**self._num_q]
        self._ret["probability_result"] = vec.dot(vec.conj())
        vec = vec/np.linalg.norm(vec)
        self._ret["result_hhl"] = vec
        solution = np.linalg.solve(self._matrix, self._vector)
        self._ret["fidelity_hhl_to_classical"] = \
            abs(vec.conj().dot(solution/np.linalg.norm(solution)))**2
        tmp_vec = self._matrix.dot(vec)
        f1 = np.linalg.norm(self._vector)/np.linalg.norm(tmp_vec)
        f2 = sum(np.angle(self._vector*tmp_vec.conj()-1+1))/self._num_q # "-1+1" to fix angle error for -0.+0.j
        self._ret["solution_scaled"] = f1*vec*np.exp(-1j*f2)
        dev = np.abs(solution/np.linalg.norm(solution)-vec)**2
        ax_dev.barh(np.arange(len(dev)), dev)
        sa = solution/np.linalg.norm(solution)
        fid = abs(sa[0]*vec[0].conj()+sa[1]*vec[1].conj())**2
        print("result expected: {}".format(sa))
        print("result HHL:      {}".format(vec))
        print("fidelity:        {}".format(fid))

        # Decoration
        ax_eig.set_title("Eigenvalue results")
        ax_eig.set_ylabel("Probability")
        ax_eig.set_xlabel("Value $\\lambda$")
        ax_eig.set_xlim(0, 2.1*np.pi/self._eigs._evo_time)
        ax_eig.axvline(x=self._reciprocal._scale*2*np.pi/self._eigs._evo_time,
                       color="g")

        ax_rec.set_title("Reciprocal results")
        ax_rec.set_ylabel("Probability")
        ax_rec.set_xlabel("$C/\\lambda$")
        ax_rec.set_xlim(0, 1)

        ax_dev.set_title("Deviation of Solution")
        ax_dev.set_xlabel("Amount")
        ax_dev.set_ylabel("Entry")
        lim = max(abs(dev)*1.5)
        ax_dev.set_xlim(0, lim)

        plt.show()

    ####################################

    def _run(self):
        self._handle_modes()
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
                self._exact_simulation()
            else:
                self._state_tomography()
        elif self._mode == "debug":
            self._exec_debug()
        elif self._mode == "swap_test":
            self._swap_test()
        # Adding a bit of general information
        self._ret["input_matrix"] = self._matrix
        self._ret["input_vector"] = self._vector
        self._ret["eigenvalues_calculated"] = np.linalg.eig(self._matrix)[0]
        self._ret["qubits_used_total"] = self._io_register.size + \
                                         self._eigenvalue_register.size + \
                                         self._ancilla_register.size
        self._ret["gate_count_total"] = self._circuit.number_atomic_gates()
        # TODO print depth of worst qubit
        return self._ret
