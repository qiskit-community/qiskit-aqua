import numpy as np
import itertools
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer
from qiskit_aqua import get_initial_state_instance
def bin_to_num(binary):
    num = np.sum([2 ** -(n + 1) for n, i in enumerate(reversed(binary)) if i == '1'])
    return num


def max_sign_bit(binary):
    return len(binary) - list(reversed(binary)).index('1')


def min_lamb(k):
    return 2 ** -k


def get_error(lamb, msb, accuracy, k, choose):
    if choose == 'max':
        er_lamb = 1 / 2 ** (k - msb + accuracy + 1)  # error of lambda = 0.5 bin width
        er_lamb[msb <= accuracy] = 0
        max_error = np.max(np.abs((np.arcsin(min_lamb(k) / (lamb - er_lamb)) -
                                   np.arcsin(min_lamb(k) / lamb))))
        print(lamb, er_lamb, msb)
        return max_error
    elif choose == 'maxrel':
        er_lamb = 1 / 2 ** (k - msb + accuracy + 1)  # error of lambda = 0.5 bin width
        er_lamb[msb <= accuracy] = 0
        maxrel_error = np.max(np.abs((np.arcsin(min_lamb(k) / (lamb - er_lamb)) -
                                      np.arcsin(min_lamb(k) / lamb)) /
                                     np.arcsin(min_lamb(k) / (lamb - er_lamb))))
        # print(lamb,er_lamb,msb)
        return maxrel_error

    else:
        raise ValueError('{} not yet implemented'.format(choose))


def get_error_2(lamb, msb, accuracy, k, choose):
    if choose == 'max':
        er_lamb = 1 / 2 ** (k - msb + accuracy + 1)  # error of lambda = 0.5 bin width
        er_lamb[msb <= accuracy] = 0
        max_error = np.max(np.abs(((min_lamb(k) / (lamb - er_lamb)) -
                                   (min_lamb(k) / lamb))))
        # print(lamb,er_lamb,msb)
        return max_error
    elif choose == 'maxrel':
        er_lamb = 1 / 2 ** (k - msb + accuracy + 1)  # error of lambda = 0.5 bin width
        er_lamb[msb <= accuracy] = 0
        maxrel_error = np.max(np.abs(((min_lamb(k) / (lamb - er_lamb)) -
                                      (min_lamb(k) / lamb)) /
                                     (min_lamb(k) / (lamb - er_lamb))))
        # print(lamb,er_lamb,msb,accuracy)
        return maxrel_error

    else:
        raise ValueError('{} not yet implemented'.format(choose))


def get_est_lamb(pattern, msb, n):
    '''Estimate the bin mid point and return the float value'''
    if msb - n > 0:
        pattern[msb - n - 1] = '1'
        return bin_to_num(pattern)
    else:
        return bin_to_num(pattern)


def error_analysis(k, n):
    '''Calculate error of arcsin rotation using k bits fixed point numbers and n bit accuracy'''
    lambda_array = []
    msb_array = []
    n_ = n  # copy of n
    for msb in range(k - 1, -1, -1):
        vec = ['0'] * k
        vec[msb] = '1'
        if msb <= n_ - 1:
            n -= 1
            # print(msb,n,n_)
        for pattern in itertools.product('10', repeat=n):
            vec[msb - n:msb] = pattern
            e_l = get_est_lamb(vec.copy(), msb, n)
            l = bin_to_num(vec)
            # print(vec,e_l,l,get_error_2(np.array([e_l]),np.array([msb]),n_,k,'maxrel'))
            lambda_array.append(get_est_lamb(vec.copy(), msb, n))
            msb_array.append(msb)
    # print("finished here")
    return (get_error_2(np.array(lambda_array), np.array(msb_array), n_, k, 'maxrel'))

def classic_approx(k, n):
    '''Calculate error of arcsin rotation using k bits fixed point numbers and n bit accuracy'''
    lambda_array = []
    msb_array = []
    pattern_array = []
    n_ = n  # copy of n
    for msb in range(k - 1, n-1, -1):
        vec = ['0'] * k
        vec[msb] = '1'

        for pattern in itertools.product('10', repeat=n):
            vec[msb - n:msb] = pattern
            e_l = get_est_lamb(vec.copy(), msb, n)
            l = bin_to_num(vec)
            # print(vec,e_l,l,get_error_2(np.array([e_l]),np.array([msb]),n_,k,'maxrel'))
            lambda_array.append(get_est_lamb(vec.copy(), msb, n))
            msb_array.append(msb)
            if len(pattern) < n_:
                pattern = list(pattern) + ['0']*(n_-len(pattern))
            #pattern_array.append(pattern)
            pattern_array.append(list(reversed(list(pattern))))
            print(vec,get_est_lamb(vec.copy(), msb, n))
    vec = ['0'] * k

    #last iterations
    for pattern in itertools.product('10', repeat=n):
        if not '1' in pattern:
            continue
        vec[msb - n:msb] = list(pattern)
        e_l = get_est_lamb(vec.copy(), msb, n)
        l = bin_to_num(vec)
        # print(vec,e_l,l,get_error_2(np.array([e_l]),np.array([msb]),n_,k,'maxrel'))
        lambda_array.append(get_est_lamb(vec.copy(), msb, n))
        msb_array.append(msb-1)
        if len(pattern) < n_:
            pattern = list(pattern) + ['0'] * (n_ - len(pattern))
        pattern_array.append(list(reversed(list(pattern))))#pattern)
        print(vec,pattern, get_est_lamb(vec.copy(), msb, n))
    # print("finished here")
    return pattern_array,np.array(lambda_array), k-np.array(msb_array)-1

def ccry(theta,control1,control2,target,qc):
    '''Implement ccRy gate using no additional ancillar qubits'''
    theta_half = theta / 2
    qc.cu3(theta_half,0,0,control2,target)
    qc.cx(control1,control2)
    qc.cu3(- theta_half, 0, 0, control2, target)
    qc.cx(control1, control2)
    qc.cu3(theta_half, 0, 0, control1, target)

class hybrid_rot(object):
    def __init__(self,k,n,initial_params,measure):
        self.k = int(k)
        self.n = int(n)
        self.anc = QuantumRegister(1,'anc')
        self.workq = QuantumRegister(n,'workq')
        self.msb = QuantumRegister(k-n+1,'msb')
        self.ev = QuantumRegister(k,'ev')
        self._circuit = QuantumCircuit(self.ev,self.workq,self.msb,self.anc)
        self.initial_params = initial_params
        self.measure = measure
        msb_num = 3
        #self.set_msb(self.msb,self.ev,msb_num)
        #self.n_controlled_rotation(self.ev,self.msb,self.workq,self.anc,['0']*n,0.5,msb_num)
        #self.draw()
        #self.set_up_circuit(k,n)

    def n_controlled_rotation(self,bitpat,msb,workq,anc,pattern,theta,msb_num):
        '''Construct a n_controlled rotation where the bit pattern fitting to the n-1 bitpat qubits perform
        a rotation by an angle theta dependent on the msb being set.

        Args:
            bitpat : Subset of register storing the n-1 long bit pattern
            msb : Qubit storing most-significant-bit
            workq : register with n-1 qubits that are used to construct the gate
            anc : qubit on which rotation is performed
            pattern (list) : bit pattern
            theta (float) : angle
            '''
        #assert len(bitpat)==len(pattern), "The specified bit sequence does not fit with the number of qubits provided"
        assert len(pattern) >= 2, "Only n >= 2 supported, use normal Toffoli gate instead"
        #print(len(bitpat))
        try:
            assert len(bitpat) >= msb_num+1+len(pattern), "Not enough qubits in the EV register to map the bit pattern"
            last_iteration = False
        except:
            last_iteration = True
            msb_num -=1
        qc = self._circuit

        if pattern[0] == '0': qc.x(bitpat[msb_num + 1])
        if pattern[1] == '0': qc.x(bitpat[msb_num + 2])
        qc.ccx(bitpat[msb_num+1],bitpat[msb_num+2],workq[0])
        idx = 1
        for idx in range(2,len(pattern)):
            if pattern[idx] == '0': qc.x(bitpat[idx+msb_num+1])
            qc.ccx(bitpat[idx+msb_num+1],workq[idx-2],workq[idx-1])

        qc.ccx(msb[msb_num if not last_iteration else msb_num+1],workq[idx-1],workq[idx])

        qc.cu3(theta,0,0,workq[idx],anc[0])
        qc.ccx(msb[msb_num if not last_iteration else msb_num+1], workq[idx -  1], workq[idx])
        #ccry(theta,workq[idx-1],msb[msb_num if not last_iteration else msb_num+1],anc[0],qc)

        for idx in range(len(pattern)-1,1,-1):
            qc.ccx(bitpat[idx + msb_num+1], workq[idx - 2], workq[idx - 1])
            if pattern[idx] == '0': qc.x(bitpat[idx + msb_num + 1])

        qc.ccx(bitpat[msb_num + 1], bitpat[msb_num + 2], workq[0])
        if pattern[1] == '0': qc.x(bitpat[msb_num + 2])
        if pattern[0] == '0': qc.x(bitpat[msb_num + 1])

    def set_msb(self,msb,ev,msb_num):
        print("MSB ",msb_num)
        qc = self._circuit
        if msb_num == 0: qc.cx(ev[0],msb[0])
        elif msb_num == 1:
            qc.x(ev[0])
            qc.ccx(ev[0],ev[1],msb[1])
            qc.x(ev[0])

        else:
            qc.x(ev[0])
            qc.x(ev[1])
            qc.ccx(ev[0],ev[1], msb[0])
            idx = 1
            for idx in range(2, msb_num):

                qc.x(ev[idx])
                qc.ccx(ev[idx], msb[idx - 2], msb[idx - 1])

            print(idx,msb_num)
            print(ev[2])
            print(msb[idx- 1])
            print(msb[idx])
            qc.ccx(ev[msb_num],msb[idx- 1], msb[msb_num])


            for idx in range(msb_num-1, 1, -1):
                qc.ccx(ev[idx ], msb[idx - 2], msb[idx - 1])
                qc.x(ev[idx])

            qc.ccx(ev[0], ev[1], msb[0])
            qc.x(ev[1])
            qc.x(ev[0])
    def _set_measurement(self):
        qc = self._circuit
        try:
            qc['c_ev']
        except:
            self.c_ev = ClassicalRegister(self.k,'c_ev')
            self.c_anc = ClassicalRegister(1, 'c_anc')
            qc.add(self.c_ev)
            qc.add(self.c_anc)
        qc.measure(self.anc, self.c_anc)

        qc.measure(self.ev,self.c_ev)

    def _uncompute_msbreg(self,max_msb):
        while max_msb >= 0:
            self.set_msb(self.msb,self.ev,int(max_msb))
            max_msb -= 1
    def _construct_initial_state(self):
        qc = self._circuit

        self._state_in = get_initial_state_instance('CUSTOM')
        self._state_in.init_params(self.initial_params)
        qc += self._state_in.construct_circuit('circuit', self.ev)
        #qc.x(self.ev[0])
        #qc.x(self.ev[2])
    def set_up_and_execute_circuit(self,k,n):
        if len(self.initial_params['state_vector'])!=0:
            self._construct_initial_state()
        pattern_ar,lambda_ar,msb_ar = classic_approx(k, n)
        old_msb = None
        for _,msb in enumerate(msb_ar):
            #if k-msb < 2:
            #    break
            if old_msb != msb:
                if old_msb != None:
                    1
                    #self.set_msb(self.msb, self.ev, int(old_msb))
                    #break
                old_msb = msb

                self.set_msb(self.msb,self.ev,int(msb))
            theta = 2*np.arcsin(np.min(lambda_ar)/lambda_ar[_])
            print("Pattern",pattern_ar[_],"Theta:",theta,"###",np.sin(theta/2),"lambda:",lambda_ar[_],"Minimal:",np.min(lambda_ar))
            #if lambda_ar[_] != 0.625:
            #    continue
            #break
            #self._circuit.ry(2*np.arcsin(0.5),self.anc[0])
            self.n_controlled_rotation(self.ev, self.msb, self.workq, self.anc, pattern_ar[_], theta, int(msb))
        self._uncompute_msbreg(msb)
        
        if self.measure:
            self._set_measurement()
            #self.draw()
            print('Circuit length is roughly: {}'.format(len(self._circuit.qasm().split('\n'))))
            return self._execute_rotation()
        if len(self.initial_params['state_vector']) == 0:
            self.draw()

    def _execute_rotation(self):
        shots = 8000
        result = execute(self._circuit, backend='local_qasm_simulator', shots=shots).result()
        counts = result.get_counts(self._circuit)
        #print(np.argmax(np.square(result.get_statevector())),np.max(np.square(result.get_statevector())))
        rd = result.get_counts(self._circuit)
        rets = sorted([[rd[k], k, k] for k in rd])[::-1]

        for d in rets:
            print(d)
            d[0] /= shots
            d[0] = np.sqrt(d[0])
            # split registers which are white space separated and decode
            c1, c2 = d[2].split()
            print(c1,c2)
            c2_ = sum([2**-(i+1) for i, e in enumerate(reversed(c2)) if e =="1"])
            print(c2_)


            d[2] = ' '.join([c1, str(c2_)])
        """
        for d in rets:
            d[0] /= shots
        """
        return rets

    def draw(self):
        drawer(self._circuit)
        plt.show()
#print(classic_approx(3,1))
