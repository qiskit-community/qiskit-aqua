import numpy as np
from math import log
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, register, execute#, ccx
from qiskit import register, available_backends, get_backend, least_busy
from qiskit.tools.qi.pauli import Pauli
from qiskit_aqua import Operator, QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_initial_state_instance, get_iqft_instance

from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer, plot_histogram
import matplotlib.pyplot as plt
import logging

from functools import reduce



logger = logging.getLogger(__name__)


#@TODO replace the creation routine for m-controlled not gates to not use ancillar qubits
#@TODO information in literature survey
def create_cn_x_gate(n,c,t):
    """Create a gate that takes the first n quibits of register c as control
    quibits and t as the target of the not operation"""
    return


class C_ROT():
    """Perform controlled rotation to invert matrix in HHL"""
    PROP_PREV_CIRCUIT = 'previous_circuit'
    PROP_INV_EIG_PRECISION = 'num_qbits_precision_ev'
    PROP_USE_BASIS_GATES = 'use_basis_gates'
    PROP_NEGATIVE_EVALS = 'negative_evals'
    PROP_BACKEND = 'backend'

    ROT_CONFIGURATION = {
        'name': 'ROT_HHL',
        'description': 'Rotation for HHL',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'rotation_schema',
            'type': 'object',
            'properties': {
                PROP_PREV_CIRCUIT: {
                    #@TODO double-check if class name or arbitrary descr.
                    'type': 'QuantumCircuit',
                    'default': None
                },

                PROP_INV_EIG_PRECISION: {
                    'type' : 'int',
                    'default': 5
                },

                PROP_NEGATIVE_EVALS: {
                    'type': 'boolean',
                    'default': False
                },

                PROP_USE_BASIS_GATES: {
                    'type': 'boolean',
                    'default': True,
                },

                PROP_BACKEND: {
                    'type': 'string',
                    'default': 'local_qasm_simulator'
                },

            },
            'additionalProperties': False
        },

    }
    def __init__(self,configuration=None):
        self._configuration = configuration or self.ROT_CONFIGURATION.copy()


    def init_params(self,params):
        """Initialize via parameters dictionary and algorithm input instance
        Args:
            qc: circuit that rotation will be added to
        """

        rot_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)

        vector = init_state_params['state_vector']
        if len(vector) > 0:

            init_state_params['num_qubits'] = int(np.log2(len(vector)))
            init_state = get_initial_state_instance(init_state_params['name'])
            init_state.init_params(init_state_params)
        else:
            init_state= None

        for k, p in self._configuration.get("input_schema").get("properties").items():
            if rot_params.get(k) == None:
                rot_params[k] = p.get("default")

        qc = rot_params.get(C_ROT.PROP_PREV_CIRCUIT)
        num_quibits_inv_ev = rot_params.get(C_ROT.PROP_INV_EIG_PRECISION)
        negative_evals = rot_params.get(C_ROT.PROP_NEGATIVE_EVALS)

        use_basis_gates = rot_params.get(C_ROT.PROP_USE_BASIS_GATES)
        backend = rot_params.get(C_ROT.PROP_BACKEND)


        self.init_args(qc,init_state,num_quibits_inv_ev
                       ,use_basis_gates,
                       negative_evals=negative_evals, backend=backend)

    def init_args(self,quantum_circuit, state_in = None,num_quibits_inv_ev=5,use_basis_gates=True,
                  negative_evals=False,  backend='local_qasm_simulator'):
        if isinstance(quantum_circuit,QuantumCircuit):
            logger.info('C_Rot circuit is added to existing circuit')
            print('C_Rot circuit is added to existing circuit')
            self._circuit = quantum_circuit
            self._standalone = False
        else:
            logger.info('C_Rot circuit is initialized from scratch')
            print('C_Rot circuit is initialized from scratch')
            self._circuit = None
            self._standalone = True
        self._state_in = state_in
        self._num_quibits_inv_ev = num_quibits_inv_ev
        self._use_basis_gates = use_basis_gates
        self._negative_evals = negative_evals

        self._backend = backend
        self._ret = {}


    def _construct_controlled_rotation_circuit(self, measure=False):
        """Implement the controlled rotation algorithm for HHL"""

        #############################################################################
        ## Compute multiplicative inverse of EV via first step of Newton iteration ##
        #############################################################################


        #if circuit exists, get registers and check if eigenvalue register is included
        if isinstance(self._circuit,QuantumCircuit):
            qregs_dict = self._circuit.get_qregs()
            try:
                inputregister = qregs_dict['eigenvalue_reg']
                n = len(inputregister)
                print("length of register for eigenvalues:",n)
            #@TODO: better Error catching
            except:
                raise RuntimeError('The circuit passed to C_Rot does not contain register with label'
                                   '\'eigenvalue_reg\''
                                   )
        elif self._circuit is None:
            n = self._num_quibits_inv_ev
            print("Number of quibits:",n)
            inputregister = QuantumRegister(n, name="eigenvalue_reg")

        else:
            raise RuntimeError('Quantum circuit passed to C_Rot instance not understood')
        flagbit = QuantumRegister(1, name="flagbit_reg")
        # plus 1 for 2**0
        outputregister = QuantumRegister(n+1, name="inv_eigenvalue_reg")

        anc = QuantumRegister(n-1, name="anc_reg_crot")

        if measure:
            classreg1 = ClassicalRegister(n)
            classreg2 = ClassicalRegister(1)
            classreg3 = ClassicalRegister(n+1)

        # add registers to existing circuit
        if isinstance(self._circuit, QuantumCircuit):
            qc = self._circuit

            for reg in (flagbit, anc, outputregister):
                qc.add(reg)
        else:
            qc = QuantumCircuit(inputregister, flagbit, anc, outputregister)
        if measure:
            for reg in (classreg1, classreg2, classreg3):
                qc.add(reg)


        # initialize state_in if starting from scratch
        if self._circuit is None and self._state_in is not None:
            qc += self._state_in.construct_circuit('circuit', inputregister)
            qc.barrier(inputregister)

        # Starting from the highest qbit, the circuit checks if only this qbit is 1
        # and sets the inverse. If there are smaller qbits at 1, the next highest power
        # is taken so that 2**p-1 <= Eigenvalue <= 2**p and the inverse is taken as 2**-p

        if not self._negative_evals:
            for _ in range(0,n-1):
                qc.x(inputregister[_+1])
                qc.ccx(inputregister[_], inputregister[_+1], anc[0])
                qc.x(inputregister[_+1])
                for i in range(2+_, n):
                    qc.x(inputregister[i])
                    qc.ccx(inputregister[i], anc[i - 2-_], anc[i - 1 - _])
                    qc.x(inputregister[i])
                # copy
                qc.x(flagbit)
                qc.ccx(flagbit[0],anc[n - 2-_], outputregister[n - 1 - _])
                qc.x(flagbit)
                # uncompute
                for i in range(n - 1 , 1+_, -1):
                    qc.x(inputregister[i])
                    qc.ccx(inputregister[i], anc[i - 2 - _], anc[i - 1 - _])
                    qc.x(inputregister[i])
                qc.x(inputregister[_+1])
                qc.ccx(inputregister[_], inputregister[_+1], anc[0])
                qc.x(inputregister[_+1])
                qc.cx(outputregister[n - 1 - _], flagbit)
                qc.x(flagbit)
                qc.ccx(inputregister[_], flagbit[0], outputregister[n - _])
                qc.x(flagbit)

                qc.cx(outputregister[n - _], flagbit)

            qc.x(flagbit[0])
            qc.ccx(inputregister[n-1], flagbit[0], outputregister[0])
            qc.x(flagbit[0])
            #@TODO need to uncompute flagbit?
        else:
            # for negative EV, the sign qbit is copied to the new register and the circuit for finding
            # 2**p is run on the other qbits

            #copy sign bit
            qc.cx(inputregister[0],outputregister[0])
            for _ in range(1,n-1):
                qc.x(inputregister[_+1])
                qc.ccx(inputregister[_], inputregister[_+1], anc[0])
                qc.x(inputregister[_+1])
                for i in range(2+_, n):
                    qc.x(inputregister[i])
                    qc.ccx(inputregister[i], anc[i - 2-_], anc[i - 1 - _])
                    qc.x(inputregister[i])
                # copy
                qc.x(flagbit)
                qc.ccx(flagbit[0],anc[n - 2-_], outputregister[n - _])
                qc.x(flagbit)
                # uncompute
                for i in range(n - 1 , 1+_, -1):
                    qc.x(inputregister[i])
                    qc.ccx(inputregister[i], anc[i - 2 - _], anc[i - 1 - _])
                    qc.x(inputregister[i])
                qc.x(inputregister[_+1])
                qc.ccx(inputregister[_], inputregister[_+1], anc[0])
                qc.x(inputregister[_+1])
                qc.cx(outputregister[n  - _], flagbit)
                qc.x(flagbit)
                qc.ccx(inputregister[_], flagbit[0], outputregister[n + 1 - _])
                qc.x(flagbit)

                qc.cx(outputregister[n +1 - _], flagbit)

            qc.x(flagbit[0])
            qc.ccx(inputregister[n-1], flagbit[0], outputregister[1])
            qc.x(flagbit[0])
            #@TODO need to uncompute flagbit?


        if measure:
            qc.barrier(inputregister, flagbit, outputregister)
            qc.measure(inputregister, classreg1)
            qc.measure(flagbit, classreg2)
            qc.measure(outputregister, classreg3)


        #@TODO: Add controlled rotation on ancillar quibit


        self._circuit = qc
        return qc

    def _setup_crot(self, measure=False):

        self._construct_controlled_rotation_circuit(measure=measure)
        logger.info('C_Rot circuit qasm length is roughly {}.'.format(
            len(self._circuit.qasm().split('\n'))
        ))
        print('C_Rot circuit circuit qasm length is roughly {}.'.format(
            len(self._circuit.qasm().split('\n'))
        ))
        return self._circuit



    def _compute_inverse_eigenvalue(self, shots=1024):
        self._setup_crot(measure=True)
        result = execute(self._circuit, backend=self._backend, shots=shots).result()
        counts = result.get_counts(self._circuit)

        rd = result.get_counts(self._circuit)
        rets = sorted([[rd[k], k, k] for k in rd])[::-1]

        for d in rets:
            d[0] /= shots
            #split registers which are white space seperated and decode
            c1,c2,c3 = d[2].split()
            if self._negative_evals and c1[-1] == "1":
                c1_ = -(sum([2**(len(c1[:-1])-i-1) for i, e in enumerate(reversed(c1[:-1])) if e ==
                    "1"]))
            else:
                if self._negative_evals:
                    c1_ = sum([2 ** (i) for i, e in enumerate((c1[:-1])) if e ==
                               "1"])
                else:
                    c1_ = sum([2 **(i) for i, e in enumerate((c1)) if e ==
                               "1"])
            if self._negative_evals and c3[-1] == "1":
                c3_ = -(sum([2**-(i+1) for i, e in enumerate(reversed(c3[:-1])) if e ==
                    "1"]))
            else:
                if self._negative_evals:
                    c3_ = sum([2 ** -(i + 1) for i, e in enumerate(reversed(c3[:-1])) if e ==
                               "1"])
                else:
                    c3_ = sum([2 ** -(i + 1) for i, e in enumerate(reversed(c3)) if e ==
                                     "1"])
            d[2] = ' '.join([str(c1_),c2,str(c3_)])


        self._ret['measurements'] = rets

    def run(self):
        self._compute_inverse_eigenvalue()
        return self._ret
