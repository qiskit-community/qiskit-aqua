import numpy as np
from math import log
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, register, execute#, ccx
from qiskit import register, available_backends, get_backend, least_busy
from qiskit.tools.qi.pauli import Pauli
from qiskit_aqua import Operator, QuantumAlgorithm, AlgorithmError
from qiskit_aqua import get_initial_state_instance, get_iqft_instance

from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer, plot_histogram
import matplotlib.pyplot as plt



def create_cn_x_gate(n,c,t):
    """Create a gate that takes the first n quibits of register c as control
    quibits and t as the target of the not operation"""
    return


class C_ROT():
    """Perform controlled rotation to invert matrix in HHL"""

    PROP_INV_EIG_PRECISION = 'number_quibits_precision_ev'
    PROP_USE_BASIS_GATES = 'use_basis_gates'
    PROP_BACKEND = 'backend'

    ROT_CONFIGURATION = {
        'name': 'ROT_HHL',
        'description': 'Rotation for HHL',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'rotation_schema',
            'type': 'object',
            'properties': {
                PROP_INV_EIG_PRECISION: {
                    'type' : 'int',
                    'default': 5
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


    def init_params(self,qc):
        """Initialize via parameters dictionary and algorithm input instance
        Args:
            qc: circuit that rotation will be added to
        """

        rot_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM)
        for k, p in self._configuration.get("input_schema").get("properties").items():
            if rot_params.get(k) == None:
                rot_params[k] = p.get("default")

        num_quibits_inv_ev = rot_params.get(C_ROT.PROP_INV_EIG_PRECISION)
        use_basis_gates = rot_params.get(C_ROT.PROP_USE_BASIS_GATES)
        backend = rot_params.get(C_ROT.PROP_BACKEND)

        self.init_args(num_quibits_inv_ev,use_basis_gates,backend)

    def init_args(self,num_quibits_inv_ev=5,use_basis_gates=True, backend='local_qasm_simulator'):
        self._num_quibits_inv_ev = num_quibits_inv_ev
        self._use_basis_gates = use_basis_gates
        self._backend = backend

    def _construct_controlled_rotation_circuit(self, measure=False):
        """Implement the controlled rotation algorithm for HHL"""
        #@TODO Implement passing name of input register



        #############################################################################
        ## Compute multiplicative inverse of EV via first step of Newton iteration ##
        #############################################################################

        n = self._num_quibits_inv_ev
        inputregister = QuantumRegister(n, name="inputregister")
        flagbit = QuantumRegister(1, name="flagbit")
        outputregister = QuantumRegister(n, name="outputregister")
        anc = QuantumRegister(n - 1, name="anc")
        if measure:
            classreg1 = ClassicalRegister(n)
            classreg2 = ClassicalRegister(1)
            classreg3 = ClassicalRegister(n)

            qc = QuantumCircuit(inputregister, flagbit, anc, outputregister, classreg1, classreg2, classreg3)
        else:
            qc = QuantumCircuit(inputregister, flagbit, anc, outputregister)

        qc.cx(inputregister[0], flagbit)
        qc.cx(inputregister[0], outputregister[n - 1])

        for i in range(1, n):
            qc.x(flagbit[0])
            qc.ccx(inputregister[i], flagbit[0], outputregister[n - 1 - i])
            qc.x(flagbit)
            qc.cx(outputregister[n - 1 - i], flagbit)

        # several control registers for the not gate of the flagbit at the end
        qc.ccx(outputregister[0], outputregister[1], anc[0])
        for i in range(2, n):
            qc.ccx(outputregister[i], anc[i - 2], anc[i - 1])

        # copy
        qc.cx(anc[n - 2], flagbit)

        # uncompute
        for i in range(n - 1, 1, -1):
            qc.ccx(outputregister[i], anc[i - 2], anc[i - 1])
        qc.ccx(outputregister[0], outputregister[1], anc[0])

        #@TODO: Why x gate
        qc.x(flagbit)


        if measure:
            qc.barrier(inputregister, flagbit, outputregister)
            qc.measure(inputregister, classreg1)
            qc.measure(flagbit, classreg2)
            qc.measure(outputregister, classreg3)


        #@TODO: Add controlled rotation on ancillar quibit


        self._circuit = qc
        return qc

    def _setup_qpe(self, measure=False):

        self._construct_controlled_rotation_circuit(measure=measure)
        logger.info('C_Rot circuit qasm length is roughly {}.'.format(
            len(self._circuit.qasm().split('\n'))
        ))
        print('C_Rot circuit circuit qasm length is roughly {}.'.format(
            len(self._circuit.qasm().split('\n'))
        ))
        return self._circuit