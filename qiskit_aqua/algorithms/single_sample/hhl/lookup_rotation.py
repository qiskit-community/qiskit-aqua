"""Controlled rotation for the HHL algorithm based on partial table lookup"""

import numpy as np
import itertools
import logging
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit_aqua import QuantumAlgorithm
from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer
from qiskit_aqua import get_initial_state_instance

logger = logging.getLogger(__name__)
class LUP_ROTATION(object):
    """Partial table lookup to rotate ancillar qubit"""
    PROP_REG_SIZE = 'reg_size'
    PROP_PAT_LENGTH = 'pat_length'
    PROP_SUBPAT_LENGTH = 'subpat_length'
    PROP_NEGATIVE_EVALS = 'negative_evals'
    PROP_BACKEND = 'backend'

    ROT_CONFIGURATION = {
        'name' : 'LUP_ROTATION',
        'description' : 'eigenvalue inversion for HHL',
        'input_schema': {
            '$schema': 'http://json-schema.org/schema#',
            'id': 'qpe_schema',
            'type': 'object',
            'properties': {
                PROP_REG_SIZE: {
                    'type': 'integer',
                    'default': 2,
                    'minimum': 2
                },
                PROP_PAT_LENGTH: {
                    'type': 'integer',
                    'default': 4,
                    'minimum': 0
                },
                PROP_SUBPAT_LENGTH: {
                    'type': 'integer',
                    'default': 3,
                    'minimum': 0
                },
                PROP_NEGATIVE_EVALS: {
                    'type': 'bool',
                    'default': False
                },
                PROP_BACKEND: {
                    'type': 'string',
                    'default': 'local_qasm_simulator'
                }
            },
            'additionalProperties': False
    },
        'depends': ['initial_state', 'qpe_hhl'],
        'defaults': {
            'initial_state': {
                'name': 'ZERO'
            },
            #@TODO what
            'qpe_hhl': {
                'name': 'STANDARD',
                'circuit': None,
                'ev_register': None,
            },

        }
    }

    def __init__(self, configuration=None):#k, n, initial_params, measure):
        self._configuration = configuration or self.ROT_CONFIGURATION.copy()
        self._k = 0#int(k)
        self._n = 0#int(n)
        self._anc = None#QuantumRegister(1, 'anc')
        self._workq = None#QuantumRegister(1, 'workq')
        self._msb = None#QuantumRegister(1, 'msb')  # QuantumRegister(k-n+1,'msb')
        self._ev = None#QuantumRegister(k, 'ev')
        self._circuit = None#QuantumCircuit(self._ev, self._msb, self._anc, self._workq)  # ,self._workq,self._msb,self._anc)
        self._measure = False
        self._state_in = None
        self._reg_size = 0
        self._pat_length = 0
        self._subpat_length = 0
        self._negative_evals = True
        self._backend = None

    def init_params(self, params):
        """
        Initialize via parameters dictionary and algorithm input instance
        Args:
            params: parameters dictionary
        """
        rot_params = params.get(QuantumAlgorithm.SECTION_KEY_ALGORITHM) or {}
        for k, p in self._configuration.get("input_schema").get("properties").items():
            if rot_params.get(k) == None:
                rot_params[k] = p.get("default")

        for k, p in self._configuration.get("defaults").items():
            if k not in params:
                params[k] = p

        reg_size = rot_params.get(LUP_ROTATION.PROP_REG_SIZE)
        pat_length = rot_params.get(LUP_ROTATION.PROP_PAT_LENGTH)
        subpat_length = rot_params.get(LUP_ROTATION.PROP_SUBPAT_LENGTH)
        negative_evals = rot_params.get(LUP_ROTATION.PROP_NEGATIVE_EVALS)
        backend = rot_params.get(LUP_ROTATION.PROP_BACKEND)





        # Set up initial state, we need to add computed num qubits to params,
        # check the length of the vector
        init_state_params = params.get(QuantumAlgorithm.SECTION_KEY_INITIAL_STATE)
        if init_state_params.get("name") == "CUSTOM":
            vector = init_state_params['state_vector']
            if len(vector) > 0:
                init_state_params['state_vector'] = vector
            init_state_params['num_qubits'] = reg_size
            assert (len(vector)==2**reg_size,
                    "The supplied init vector does not fit with the register size")

            init_state = get_initial_state_instance(init_state_params['name'])
        else:
            init_state = None

        # Set up inclusion of existing circuit
        init_circuit_params = params.get('qpe_hhl')
        if init_circuit_params.get("name") == "STANDARD":
            circuit = init_circuit_params['circuit']
            ev_register = init_circuit_params['ev_register']
            assert ev_register in circuit.qregs, "The EV register is not found in the circuit"

        else:
            ev_register = None
            circuit = None

        self.init_args(ev_register,circuit,
            reg_size=reg_size,pat_length=pat_length,
            subpat_length=subpat_length,negative_evals=negative_evals,
            backend=backend,state_in=init_state,
        )

    def init_args(self,ev_register,circuit,reg_size, pat_length, subpat_length,
            negative_evals=True, backend='local_qasm_simulator', state_in=None):

        self._ev = ev_register
        self._circuit = circuit
        self._state_in = state_in
        self._reg_size = reg_size
        self._pat_length = pat_length
        self._subpat_length = subpat_length
        self._negative_evals = negative_evals
        self._backend = backend


    @staticmethod
    def classic_approx(k, n, m):
        '''Calculate error of arcsin rotation using k bits fixed point numbers and n bit accuracy'''

        def bin_to_num(binary):
            num = np.sum([2 ** -(n + 1) for n, i in enumerate(reversed(binary)) if i == '1'])
            return num

        def max_sign_bit(binary):
            return len(binary) - list(reversed(binary)).index('1')

        def min_lamb(k):
            return 2 ** -k

        def get_est_lamb(pattern, msb, n):
            '''Estimate the bin mid point and return the float value'''
            if msb - n > 0:
                pattern[msb - n - 1] = '1'
                return bin_to_num(pattern)
            else:
                return bin_to_num(pattern)

        pattern_array = []
        from collections import OrderedDict
        output = OrderedDict()
        n_ = n  # copy of n
        for msb in range(k - 1, n - 1, -1):
            # if msb!=k-2: continue
            vec = ['0'] * k
            vec[msb] = '1'

            for pattern_ in itertools.product('10', repeat=m):
                app_pattern_array = []
                lambda_array = []
                msb_array = []

                for appendpat in itertools.product('10', repeat=n - m):
                    pattern = pattern_ + appendpat
                    # print(pattern)
                    if not '1' in pattern:
                        continue
                    vec[msb - n:msb] = pattern
                    e_l = get_est_lamb(vec.copy(), msb, n)
                    l = bin_to_num(vec)
                    lambda_array.append(get_est_lamb(vec.copy(), msb, n))
                    msb_array.append(msb)
                    app_pattern_array.append(list(reversed(appendpat)))
                try:
                    prev_res = output[k - msb - 1]
                except:
                    prev_res = []

                output.update(
                    {k - msb - 1: prev_res + [(list(reversed(list(pattern_))), app_pattern_array, lambda_array)]})

        vec = ['0'] * k

        # last iterations
        for pattern_ in itertools.product('10', repeat=m):
            app_pattern_array = []
            lambda_array = []
            msb_array = []
            for appendpat in itertools.product('10', repeat=n - m):
                pattern = pattern_ + appendpat
                if not '1' in pattern:
                    continue
                vec[msb - n:msb] = list(pattern)
                e_l = get_est_lamb(vec.copy(), msb, n)
                l = bin_to_num(vec)
                lambda_array.append(get_est_lamb(vec.copy(), msb, n))
                msb_array.append(msb - 1)
                app_pattern_array.append(list(reversed(appendpat)))
            try:
                prev_res = output[k - msb]
            except:
                prev_res = []

            output.update({(k - msb): prev_res + [(list(reversed(list(pattern_))), app_pattern_array, lambda_array)]})

        return output

    def nc_toffoli(self, ctl, tgt, n, offset):
        '''Implement n+1-bit toffoli using the approach in Elementary gates'''

        assert n >= 3, "This method works only for more than 2 control bits"

        from sympy.combinatorics.graycode import GrayCode
        gray_code = list(GrayCode(n).generate_gray())
        last_pattern = None
        qc = self._circuit

        # angle to construct nth square root of diagonlized pauli x matrix
        # via u3(0,lam_angle,0)
        lam_angle = np.pi / (2 ** (n - 1))
        # transform to eigenvector basis of pauli X
        qc.h(tgt[0])
        for pattern in gray_code:

            if not '1' in pattern:
                continue
            if last_pattern is None:
                last_pattern = pattern
            # find left most set bit
            lm_pos = list(pattern).index('1')

            # find changed bit
            comp = [i != j for i, j in zip(pattern, last_pattern)]
            if True in comp:
                pos = comp.index(True)
            else:
                pos = None
            if pos is not None:
                if pos != lm_pos:
                    qc.cx(ctl[offset + pos], ctl[offset + lm_pos])
                else:
                    indices = [i for i, x in enumerate(pattern) if x == '1']
                    for idx in indices[1:]:
                        qc.cx(ctl[offset + idx], ctl[offset + lm_pos])
            # check parity
            if pattern.count('1') % 2 == 0:
                # inverse
                qc.cu3(0, -lam_angle, 0, ctl[offset + lm_pos], tgt)
            else:
                qc.cu3(0, lam_angle, 0, ctl[offset + lm_pos], tgt)
            last_pattern = pattern
        qc.h(tgt[0])

    def set_msb(self, msb, ev, msb_num, last_iteration=False):
        qc = self._circuit
        if last_iteration:
            print("finishing soon", msb_num)
            if msb_num == 1:
                qc.x(ev[0])
                qc.cx(ev[0], msb[0])
                qc.x(ev[0])
            elif msb_num == 2:
                qc.x(ev[0])
                qc.x(ev[1])
                qc.ccx(ev[0], ev[1], msb[0])
                qc.x(ev[1])
                qc.x(ev[0])
            elif msb_num > 2:
                for idx in range(msb_num):
                    qc.x(ev[idx])
                self.nc_toffoli(ev, msb[0], int(msb_num), 0)
                for idx in range(msb_num):
                    qc.x(ev[idx])
            else:
                raise ValueError()
        elif msb_num == 0:
            qc.cx(ev[0], msb)
        elif msb_num == 1:
            qc.x(ev[0])
            qc.ccx(ev[0], ev[1], msb[0])
            qc.x(ev[0])

        elif msb_num > 1:
            for idx in range(msb_num):
                qc.x(ev[idx])
            self.nc_toffoli(ev, msb[0], int(msb_num + 1), 0)
            for idx in range(msb_num):
                qc.x(ev[idx])

    def _set_measurement(self):
        qc = self._circuit
        try:
            qc['c_ev']
        except:
            self.c_ev = ClassicalRegister(self._reg_size, 'c_ev')
            # self.c_msb = ClassicalRegister(int(np.ceil(np.log2(self.k-self.n+2))),'c_msb')
            self.c_anc = ClassicalRegister(1, 'c_anc')

            qc.add(self.c_ev)
            qc.add(self.c_anc)
            # qc.add(self.c_msb)
        # qc.measure(self._msb,self.c_msb)
        qc.measure(self._anc, self.c_anc)

        qc.measure(self._ev, self.c_ev)

    def _construct_initial_state(self):
        qc = self._circuit

        self._state_in = get_initial_state_instance('CUSTOM')
        self._state_in.init_params(self.initial_params)
        qc += self._state_in.construct_circuit('circuit', self._ev)

    def set_bit_pattern(self, pattern, tgt, offset):
        qc = self._circuit
        for c, i in enumerate(pattern):
            if i == '0':
                qc.x(self._ev[int(c + offset)])

        if len(pattern) > 2:
            self.nc_toffoli(self._ev, tgt, len(pattern), int(offset))
        elif len(pattern) == 2:
            qc.ccx(self._ev[offset], self._ev[offset + 1], tgt)
        elif len(pattern) == 1:
            qc.cx(self._ev[offset], tgt)

        for c, i in enumerate(pattern):
            if i == '0':
                qc.x(self._ev[int(c + offset)])

    def ccry(self, theta, control1, control2, target):
        '''Implement ccRy gate using no additional ancillar qubits'''
        # double angle because the rotation is defined with theta/2
        theta = 2 * theta
        qc = self._circuit
        theta_half = theta / 2
        qc.cu3(theta_half, 0, 0, control2, target)
        qc.cx(control1, control2)
        qc.cu3(- theta_half, 0, 0, control2, target)
        qc.cx(control1, control2)
        qc.cu3(theta_half, 0, 0, control1, target)

    def _construct_rotation_circuit(self):
        if self._circuit is None:
            self._ev = QuantumRegister(self._reg_size,'ev')
            self._circuit = QuantumCircuit(self._ev)
        self._workq = QuantumRegister(1, 'work')
        self._msb = QuantumRegister(1, 'msb')
        self._anc = QuantumRegister(1, 'anc')
        self._circuit.add(self._anc)
        self._circuit.add(self._msb)
        self._circuit.add(self._work)

        qc = self._circuit
        if self._state_in is not None:
            qc += self._state_in.construct_circuit('circuit', self._ev)
        m = self._subpat_length
        n = self._pat_length
        k = self._reg_size
        approx_dict = LUP_ROTATION.classic_approx(k, n, m)

        old_msb = None
        for _, msb in enumerate(list(approx_dict.keys())):  # enumerate(list(reversed(list(approx_dict.keys())))):
            pattern_map = approx_dict[msb]
            if old_msb != msb:
                if old_msb != None:
                    self.set_msb(self._msb, self._ev, int(old_msb))
                old_msb = msb
                if msb + n == k:
                    self.set_msb(self._msb, self._ev, int(msb), last_iteration=True)
                else:
                    self.set_msb(self._msb, self._ev, int(msb), last_iteration=False)
            offset_mpat = msb + (n - m) if msb < k - n else msb + n - m - 1

            for mainpat, subpat, lambda_ar in pattern_map:
                self.set_bit_pattern(mainpat, self._workq[0], offset_mpat + 1)
                for subpattern, lambda_ in zip(subpat, lambda_ar):
                    theta = np.arcsin(2 ** int(-k) / lambda_)
                    offset = msb + 1 if msb < k - n else msb

                    self.ccry(theta / 2, self._workq[0], self._msb[0], self._anc[0])

                    self.set_bit_pattern(subpattern, self._anc[0], offset)

                    self.ccry(-theta / 2, self._workq[0], self._msb[0], self._anc[0])

                    self.set_bit_pattern(subpattern, self._anc[0], offset)
                    # break
                    #print(msb, mainpat, subpattern, lambda_, np.sin(theta))
                    vec = [0] * k
                    if k != n + msb:
                        vec[msb] = 1
                    vec[offset:offset + (n - m)] = subpattern
                    vec[offset_mpat:offset_mpat + len(mainpat)] = mainpat
                self.set_bit_pattern(mainpat, self._workq[0], offset_mpat + 1)

        self.set_msb(self._msb, self._ev, int(msb), last_iteration=True)
        return qc

    def _execute_rotation(self):

        self._construct_rotation_circuit()
        self._set_measurement()

        shots = 8000
        backend = self._backend
        result = execute(self._circuit, backend=backend, shots=shots).result()
        counts = result.get_counts(self._circuit)
        rd = result.get_counts(self._circuit)
        rets = sorted([[rd[k], k, k] for k in rd])[::-1]
        # return rets
        for d in rets:
            print(d)
            d[0] /= shots
            d[0] = np.sqrt(d[0])
            # split registers which are white space separated and decode
            c1, c2 = d[2].split()
            c2_ = sum([2 ** -(i + 1) for i, e in enumerate(reversed(c2)) if e == "1"])
            d[2] = ' '.join([c1, str(c2_)])
        self._ret = rets
        return rets
    def run(self):
        self._execute_rotation()
        return self._ret
    def _draw(self):
        drawer(self._circuit)
        plt.show()

    @staticmethod
    def get_statevector_representation(bitpattern):
        '''Using the tensorproduct of the qubits representing the bitpattern, estimate
        the input state vector to the rotation routine
        Args:
            bitpattern (list)'''
        state_vector = None
        for bit in bitpattern:
            vec = np.array([0, 1]) if bit == '1' else np.array([1, 0])
            if state_vector is None:
                state_vector = vec
                continue
            state_vector = np.tensordot(state_vector, vec, axes=0)
        print(state_vector.flatten().shape)

        return (state_vector.flatten())

    def test_value_range(k,n):

        params = {
            'algorithm': {
                'reg_size':k,
                'pat_length':n,
                'subpat_length':int(np.ceil(n/2)),
                'negative_evals':False,
                'backend':'local_qasm_simulator'
            },
            # for running the rotation seperately, supply input
            "initial_state": {
                "name": "CUSTOM",
                "state_vector": []
            },
            "qpe_hhl":{
                "name": "ZERO"
            }
        }
        res_dict = {}
        counter = 0
        for pattern in itertools.product('01', repeat=k):
            if not '1' in pattern:
                continue
            state_vector = LUP_ROTATION.get_statevector_representation(list(pattern))
            draw = False  # True
            params["initial_state"]["state_vector"] = state_vector
            # state_vector = np.zeros(2**k)
            # state_vector[4] = 1

            print(state_vector)
            print("Next pattern:", pattern)
            obj = LUP_ROTATION(params)

            res = obj.run()
            # break
            for d in res:
                if d[1][0] == '1':
                    res_dict.update({float(d[2].split()[-1]): d[0]})
            counter += 1
            # if counter == 4:
            #    break
        vals = list(res_dict.keys())
        inverse = [res_dict[_] for _ in vals]
        vals = np.array(vals)
        inverse = np.array(inverse)
        plt.scatter(vals, inverse / 2 ** (-len(res[0][1].split()[-1])))
        plt.plot(np.linspace(2 ** -k, 1, 1000), 1 / np.linspace(2 ** -k, 1, 1000))
        plt.show()


    def test_stat_error(k, n):
        params = {
            'algorithm': {
                'reg_size': k,
                'pat_length': n,
                'subpat_length': int(np.ceil(n / 2)),
                'negative_evals': False,
                'backend': 'local_qasm_simulator'
            },
            # for running the rotation seperately, supply input
            "initial_state": {
                "name": "CUSTOM",
                "state_vector": []
            },
            "qpe_hhl": {
                "name": "ZERO"
            }
        }
        counts = []
        pattern = ['1'] * k
        state_vector = LUP_ROTATION.get_statevector_representation(list(pattern))
        params["initial_state"]["state_vector"] = state_vector

        for i in range(100):
            obj = LUP_ROTATION(params)
            res = obj.run()
            # break
            for d in res:
                if d[1][0] == '1':
                    counts.append(d[0])
        plt.hist(counts)
        plt.show()
# classic_approx_v2(5,3,2)
