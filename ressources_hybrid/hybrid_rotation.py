import numpy as np
import itertools
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer
from qiskit_aqua import get_initial_state_instance
from qiskit_addon_jku import JKUProvider
from qiskit.wrapper._wrapper import _DEFAULT_PROVIDER

_DEFAULT_PROVIDER.add_provider(JKUProvider())

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
        # #print(lamb,er_lamb,msb)
        return maxrel_error

    else:
        raise ValueError('{} not yet implemented'.format(choose))


def get_error_2(lamb, msb, accuracy, k, choose):
    if choose == 'max':
        er_lamb = 1 / 2 ** (k - msb + accuracy + 1)  # error of lambda = 0.5 bin width
        er_lamb[msb <= accuracy] = 0
        max_error = np.max(np.abs(((min_lamb(k) / (lamb - er_lamb)) -
                                   (min_lamb(k) / lamb))))
        # #print(lamb,er_lamb,msb)
        return max_error
    elif choose == 'maxrel':
        er_lamb = 1 / 2 ** (k - msb + accuracy + 1)  # error of lambda = 0.5 bin width
        er_lamb[msb <= accuracy] = 0
        maxrel_error = np.max(np.abs(((min_lamb(k) / (lamb - er_lamb)) -
                                      (min_lamb(k) / lamb)) /
                                     (min_lamb(k) / (lamb - er_lamb))))
        # #print(lamb,er_lamb,msb,accuracy)
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
            # #print(msb,n,n_)
        for pattern in itertools.product('10', repeat=n):
            vec[msb - n:msb] = pattern
            e_l = get_est_lamb(vec.copy(), msb, n)
            l = bin_to_num(vec)
            # #print(vec,e_l,l,get_error_2(np.array([e_l]),np.array([msb]),n_,k,'maxrel'))
            lambda_array.append(get_est_lamb(vec.copy(), msb, n))
            msb_array.append(msb)
    # #print("finished here")
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
            # #print(vec,e_l,l,get_error_2(np.array([e_l]),np.array([msb]),n_,k,'maxrel'))
            lambda_array.append(get_est_lamb(vec.copy(), msb, n))
            msb_array.append(msb)
            if len(pattern) < n_:
                pattern = list(pattern) + ['0']*(n_-len(pattern))
            #pattern_array.append(pattern)
            pattern_array.append(list(reversed(list(pattern))))
            ##print(vec,get_est_lamb(vec.copy(), msb, n))
    vec = ['0'] * k

    #last iterations
    for pattern in itertools.product('10', repeat=n):
        if not '1' in pattern:
            continue
        vec[msb - n:msb] = list(pattern)
        e_l = get_est_lamb(vec.copy(), msb, n)
        l = bin_to_num(vec)
        # #print(vec,e_l,l,get_error_2(np.array([e_l]),np.array([msb]),n_,k,'maxrel'))
        lambda_array.append(get_est_lamb(vec.copy(), msb, n))
        msb_array.append(msb-1)
        if len(pattern) < n_:
            pattern = list(pattern) + ['0'] * (n_ - len(pattern))
        pattern_array.append(list(reversed(list(pattern))))#pattern)
        ##print(vec,pattern, get_est_lamb(vec.copy(), msb, n))
    # #print("finished here")
    return pattern_array,np.array(lambda_array), k-np.array(msb_array)-1

def classic_approx_v2(k, n,m):
    '''Calculate error of arcsin rotation using k bits fixed point numbers and n bit accuracy'''
    pattern_array = []
    from collections import OrderedDict
    output = OrderedDict()
    n_ = n  # copy of n
    for msb in range(k - 1, n-1, -1):
            vec = ['0'] * k
            vec[msb] = '1'

            for pattern_ in itertools.product('10', repeat=m):
                app_pattern_array = []
                lambda_array = []
                msb_array = []
    
                for appendpat in itertools.product('10',repeat=n-m):
                    pattern = pattern_+appendpat
                    print(pattern)
                    vec[msb - n:msb] = pattern
                    print(vec)
                    e_l = get_est_lamb(vec.copy(), msb, n)
                    l = bin_to_num(vec)
                    # #print(vec,e_l,l,get_error_2(np.array([e_l]),np.array([msb]),n_,k,'maxrel'))
                    lambda_array.append(get_est_lamb(vec.copy(), msb, n))
                    msb_array.append(msb)
                    if len(pattern) < n_:
                        print("Still active, this part")
                        pattern = list(pattern) + ['0']*(n_-len(pattern))
                   
                    app_pattern_array.append(list(reversed(appendpat)))
                try:
                    
                    prev_res = output[k-msb-1]
                    
                except:
                    prev_res = []
                output.update({k-msb-1:prev_res+[list(reversed(list(pattern_))),app_pattern_array,lambda_array]})
                #pattern_array.append([list(reversed(list(pattern_))),app_pattern_array,lambda_array,msb_array])
            ##print(vec,get_est_lamb(vec.copy(), msb, n))
            vec = ['0'] * k

            #last iterations
            for pattern_ in itertools.product('10', repeat=m):
                app_pattern_array = []
                lambda_array = []
                msb_array = []
                
                for appendpat in itertools.product('10',repeat=n-m):
                    pattern = pattern_+appendpat
                
                    if not '1' in pattern:
                        continue
                    vec[msb - n:msb] = list(pattern)
                    e_l = get_est_lamb(vec.copy(), msb, n)
                    l = bin_to_num(vec)
                    # #print(vec,e_l,l,get_error_2(np.array([e_l]),np.array([msb]),n_,k,'maxrel'))
                    lambda_array.append(get_est_lamb(vec.copy(), msb, n))
                    msb_array.append(msb-1)
                    if len(pattern) < n_:
                        print("Still active, this part")
                        pattern = list(pattern) + ['0'] * (n_ - len(pattern))
                    app_pattern_array.append(list(reversed(appendpat)))
                try:
                    prev_res = output[k-msb]
                    print("found")
                except:
                    prev_res = []
                output.update({(k-msb):prev_res+[list(reversed(list(pattern_))),app_pattern_array,lambda_array]})
                
                    #pattern_array.append(list(reversed(list(pattern))))#pattern)
                #pattern_array.append([list(reversed(list(pattern_))),app_pattern_array,lambda_array,msb_array])
        ##print(vec,pattern, get_est_lamb(vec.copy(), msb, n))
    # #print("finished here")
    print(output)
    return output#,np.array(lambda_array), k-np.array(msb_array)-1

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
        #self.workq = QuantumRegister(n-2,'workq')
        self.msb = QuantumRegister(int(np.ceil(np.log2(k-n+2))),'msb')#QuantumRegister(k-n+1,'msb')
        self.ev = QuantumRegister(k,'ev')
        self._circuit = QuantumCircuit(self.ev,self.msb,self.anc)#,self.workq,self.msb,self.anc)
        self.initial_params = initial_params
        self.measure = measure
        msb_num = 3
        #self.set_msb(self.msb,self.ev,msb_num)
        #self.n_controlled_rotation(self.ev,self.msb,self.workq,self.anc,['0']*n,0.5,msb_num)
        #self.draw()
        #self.set_up_circuit(k,n)
        

    def nc_toffoli(self,ctl,tgt,n,offset):
        '''Implement n+1-bit toffoli using the approach in Elementary gates'''
        
        assert n>=3,"This method works only for more than 2 control bits"
        
        from sympy.combinatorics.graycode import GrayCode
        gray_code = list(GrayCode(n).generate_gray())
        last_pattern = None
        qc = self._circuit

        #angle to construct nth square root of diagonlized pauli x matrix
        #via u3(0,lam_angle,0)
        lam_angle = np.pi/(2**(self.n-1))
        #transform to eigenvector basis of pauli X
        qc.h(tgt[0])
        for pattern in gray_code:
            
            if not '1' in pattern:
                continue
            if last_pattern is None:
                last_pattern = pattern
            #find left most set bit
            lm_pos = list(pattern).index('1')

            #find changed bit
            comp = [i!=j for i,j in zip(pattern,last_pattern)]
            if True in comp:
                pos = comp.index(True)
            else:
                pos = None
            if pos is not None:
                if pos != lm_pos:
                    qc.cx(ctl[offset+pos],ctl[offset+lm_pos])
                else:
                    indices = [i for i, x in enumerate(pattern) if x == '1']
                    for idx in indices[1:]:
                        qc.cx(ctl[offset+idx],ctl[offset+lm_pos])
            #check parity
            if pattern.count('1') % 2 == 0:
                #inverse
                qc.cu3(0,-lam_angle,0,ctl[offset+lm_pos],tgt)
            else:
                qc.cu3(0,lam_angle,0,ctl[offset+lm_pos],tgt)
            last_pattern = pattern
        qc.h(tgt[0])

    def n_controlled_rotation_optimized_v2(self,bitpat,msb,anc,pattern,theta,msb_num):
        '''Construct a n_controlled rotation where the bit pattern fitting to the n-1 bitpat qubits perform
        a rotation by an angle theta dependent on the msb being set.

        This implementation requires 1 qubit less than the n_controlled_rotation function.

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
        ##print(len(bitpat))
        last_iteration = False
        try:
            assert len(bitpat) >= msb_num+1+len(pattern), "Not enough qubits in the EV register to map the bit pattern"
        
        except:
            #return
            last_iteration = True
            #msb_num -=1
        #print("MSB NUMEMR IN ROT:",msb_num)
        if (self.k-self.n)==msb_num:
            last_iteration = True
            msb_num -= 1
            #print("last iteration")
       
        qc = self._circuit

        #if pattern[0] == '0': qc.x(bitpat[msb_num + 1])
        #if pattern[1] == '0': qc.x(bitpat[msb_num + 2])
        #qc.ccx(bitpat[msb_num+1],bitpat[msb_num+2],workq[0])
        #idx = 1
        #for idx in range(2,len(pattern)-1):
        #    if pattern[idx] == '0': qc.x(bitpat[idx+msb_num+1])
        #    qc.ccx(bitpat[idx+msb_num+1],workq[idx-2],workq[idx-1])
        for n,_ in enumerate(pattern):
            if _ == '0':
                qc.x(bitpat[msb_num+n+1])
        if 1:# idx > 1:
            #qc.cu3(theta/2,0,0,msb[msb_num if not last_iteration else msb_num+1],anc[0])
            if not last_iteration:
                self.num_controlled_Ry(self.msb,self.anc[0],msb_num,theta/2)
            else:
                self.num_controlled_Ry(self.msb,self.anc[0],msb_num+1,theta/2)
            

            #if pattern[idx+1] == '0': qc.x(bitpat[idx+msb_num+2])
            #qc.ccx(bitpat[idx+msb_num+2],workq[idx-1],anc[0])
            self.nc_toffoli(self.ev,self.anc[0],self.n,msb_num+1)
            #qc.cu3(-theta/2,0,0,msb[msb_num if not last_iteration else msb_num+1],anc[0])
            if not last_iteration:
                self.num_controlled_Ry(self.msb,self.anc[0],msb_num,-theta/2)
            else:
                self.num_controlled_Ry(self.msb,self.anc[0],msb_num+1,-theta/2)
            self.nc_toffoli(self.ev,self.anc[0],self.n,msb_num+1)
            #qc.ccx(bitpat[idx+msb_num+2],workq[idx-1],anc[0])
            #if pattern[idx+1] == '0': qc.x(bitpat[idx+msb_num+2])
        for n,_ in enumerate(pattern):
            if _ == '0':
                qc.x(bitpat[msb_num+n+1])
       
            
        #qc.cu3(theta,0,0,workq[idx],anc[0])
        #qc.ccx(msb[msb_num if not last_iteration else msb_num+1], workq[idx -  1], workq[idx])
        #ccry(theta,workq[idx-1],msb[msb_num if not last_iteration else msb_num+1],anc[0],qc)

        #for idx in range(len(pattern)-2,1,-1):
        #    qc.ccx(bitpat[idx + msb_num+1], workq[idx - 2], workq[idx - 1])
        #    if pattern[idx] == '0': qc.x(bitpat[idx + msb_num + 1])

        #qc.ccx(bitpat[msb_num + 1], bitpat[msb_num + 2], workq[0])
        #if pattern[1] == '0': qc.x(bitpat[msb_num + 2])
        #if pattern[0] == '0': qc.x(bitpat[msb_num + 1])

        
    def n_controlled_rotation_optimized(self,bitpat,msb,anc,pattern,theta,msb_num):
        '''Construct a n_controlled rotation where the bit pattern fitting to the n-1 bitpat qubits perform
        a rotation by an angle theta dependent on the msb being set.

        This implementation requires 1 qubit less than the n_controlled_rotation function.

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
        ##print(len(bitpat))
        try:
            assert len(bitpat) >= msb_num+1+len(pattern), "Not enough qubits in the EV register to map the bit pattern"
            #return
            last_iteration = False
        except:
            #return
            #print("last iteration!!!!")
            last_iteration = True
            msb_num -=1
        qc = self._circuit
         
        #if pattern[0] == '0': qc.x(bitpat[msb_num + 1])
        #if pattern[1] == '0': qc.x(bitpat[msb_num + 2])
        #qc.ccx(bitpat[msb_num+1],bitpat[msb_num+2],workq[0])
        #idx = 1
        #for idx in range(2,len(pattern)-1):
        #    if pattern[idx] == '0': qc.x(bitpat[idx+msb_num+1])
        #    qc.ccx(bitpat[idx+msb_num+1],workq[idx-2],workq[idx-1])
        for n,_ in enumerate(pattern):
            if _ == '0':
                qc.x(bitpat[msb_num+n+1])
        if 1:# idx > 1:
            qc.cu3(theta/2,0,0,msb[msb_num if not last_iteration else msb_num+1],anc[0])
            #if pattern[idx+1] == '0': qc.x(bitpat[idx+msb_num+2])
            #qc.ccx(bitpat[idx+msb_num+2],workq[idx-1],anc[0])
            self.nc_toffoli(self.ev,self.anc[0],self.n,msb_num+1)
            qc.cu3(-theta/2,0,0,msb[msb_num if not last_iteration else msb_num+1],anc[0])

            self.nc_toffoli(self.ev,self.anc[0],self.n,msb_num+1)
            #qc.ccx(bitpat[idx+msb_num+2],workq[idx-1],anc[0])
            #if pattern[idx+1] == '0': qc.x(bitpat[idx+msb_num+2])
        for n,_ in enumerate(pattern):
            if _ == '0':
                qc.x(bitpat[msb_num+n+1])
       
            
        #qc.cu3(theta,0,0,workq[idx],anc[0])
        #qc.ccx(msb[msb_num if not last_iteration else msb_num+1], workq[idx -  1], workq[idx])
        #ccry(theta,workq[idx-1],msb[msb_num if not last_iteration else msb_num+1],anc[0],qc)

        #for idx in range(len(pattern)-2,1,-1):
        #    qc.ccx(bitpat[idx + msb_num+1], workq[idx - 2], workq[idx - 1])
        #    if pattern[idx] == '0': qc.x(bitpat[idx + msb_num + 1])

        #qc.ccx(bitpat[msb_num + 1], bitpat[msb_num + 2], workq[0])
        #if pattern[1] == '0': qc.x(bitpat[msb_num + 2])
        #if pattern[0] == '0': qc.x(bitpat[msb_num + 1])
                

    def n_controlled_rotation_(self,bitpat,msb,workq,anc,pattern,theta,msb_num):
        '''Construct a n_controlled rotation where the bit pattern fitting to the n-1 bitpat qubits perform
        a rotation by an angle theta dependent on the msb being set.

        This implementation requires 1 qubit less than the n_controlled_rotation function.

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
        ##print(len(bitpat))
        try:
            assert len(bitpat) >= msb_num+1+len(pattern), "Not enough qubits in the EV register to map the bit pattern"
            #return
            last_iteration = False
        except:
            #return
            last_iteration = True
            msb_num -=1
        qc = self._circuit

        if pattern[0] == '0': qc.x(bitpat[msb_num + 1])
        if pattern[1] == '0': qc.x(bitpat[msb_num + 2])
        qc.ccx(bitpat[msb_num+1],bitpat[msb_num+2],workq[0])
        idx = 1
        for idx in range(2,len(pattern)-1):
            if pattern[idx] == '0': qc.x(bitpat[idx+msb_num+1])
            qc.ccx(bitpat[idx+msb_num+1],workq[idx-2],workq[idx-1])
        if 1:# idx > 1:
            qc.cu3(theta/2,0,0,msb[msb_num if not last_iteration else msb_num+1],anc[0])
            if pattern[idx+1] == '0': qc.x(bitpat[idx+msb_num+2])
            qc.ccx(bitpat[idx+msb_num+2],workq[idx-1],anc[0])
            qc.cu3(-theta/2,0,0,msb[msb_num if not last_iteration else msb_num+1],anc[0])
            qc.ccx(bitpat[idx+msb_num+2],workq[idx-1],anc[0])
            if pattern[idx+1] == '0': qc.x(bitpat[idx+msb_num+2])
        
            
        #qc.cu3(theta,0,0,workq[idx],anc[0])
        #qc.ccx(msb[msb_num if not last_iteration else msb_num+1], workq[idx -  1], workq[idx])
        #ccry(theta,workq[idx-1],msb[msb_num if not last_iteration else msb_num+1],anc[0],qc)

        for idx in range(len(pattern)-2,1,-1):
            qc.ccx(bitpat[idx + msb_num+1], workq[idx - 2], workq[idx - 1])
            if pattern[idx] == '0': qc.x(bitpat[idx + msb_num + 1])

        qc.ccx(bitpat[msb_num + 1], bitpat[msb_num + 2], workq[0])
        if pattern[1] == '0': qc.x(bitpat[msb_num + 2])
        if pattern[0] == '0': qc.x(bitpat[msb_num + 1])
        

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
        ##print(len(bitpat))
        try:
            assert len(bitpat) >= msb_num+1+len(pattern), "Not enough qubits in the EV register to map the bit pattern"
            #return
            last_iteration = False
        except:
            #return
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

        #qc.ccx(msb[msb_num if not last_iteration else msb_num+1],workq[idx-1],workq[idx])

        #qc.cu3(theta,0,0,workq[idx],anc[0])
        #qc.ccx(msb[msb_num if not last_iteration else msb_num+1], workq[idx -  1], workq[idx])
        ccry(theta,workq[idx-1],msb[msb_num if not last_iteration else msb_num+1],anc[0],qc)

        for idx in range(len(pattern)-1,1,-1):
            qc.ccx(bitpat[idx + msb_num+1], workq[idx - 2], workq[idx - 1])
            if pattern[idx] == '0': qc.x(bitpat[idx + msb_num + 1])

        qc.ccx(bitpat[msb_num + 1], bitpat[msb_num + 2], workq[0])
        if pattern[1] == '0': qc.x(bitpat[msb_num + 2])
        if pattern[0] == '0': qc.x(bitpat[msb_num + 1])

    def num_controlled_Ry(self,ctl,tgt,number,theta):
        '''Check ctl register for fixed point repr. of number and perform rotation on target

        Args:
        ctl : QRegister storing number
        number (int): number mapped to rotation
        tgt : Qubit that is rotated
        theta (float) : rotation angles theta, lambda, phi'''

        def generate_bit_string(number,size):
            bitstring = "{:b}".format(number+1)
            #enlong with 0s 
            bitstring = (size-len(bitstring))*'0' + bitstring
            return list(reversed(bitstring))

        def ccry(theta,control1,control2,target,qc):
            '''Implement ccRy gate using no additional ancillar qubits'''
            theta_half = theta / 2
            qc.cu3(theta_half,0,0,control2,target)
            qc.cx(control1,control2)
            qc.cu3(- theta_half, 0, 0, control2, target)
            qc.cx(control1, control2)
            qc.cu3(theta_half, 0, 0, control1, target)
        
        #generate bitstring of length len(tgt)
        bitstring = generate_bit_string(number,len(ctl))
        #print("Will perform rotation with bitstring {} and angle {}".format(bitstring,theta))
        qc = self._circuit
        for count,_ in enumerate(bitstring):
            if _ == '0':
                qc.x(ctl[count])

        n = len(ctl)
        
        #assert n>=3,"This method works only for more than 2 control bits"
        if n >= 3:
            # use n controlled U gate
            from sympy.combinatorics.graycode import GrayCode
            gray_code = list(GrayCode(n).generate_gray())
            last_pattern = None
            qc = self._circuit
            #angle to construct nth square root of Ry(Theta)
            #via u3(theta_angle,0,0)
            theta_angle = theta/(2**(n-1))
            
            
            for pattern in gray_code:
            
                if not '1' in pattern:
                    continue
                if last_pattern is None:
                    last_pattern = pattern
                #find left most set bit
                lm_pos = list(pattern).index('1')

                #find changed bit
                comp = [i!=j for i,j in zip(pattern,last_pattern)]
                if True in comp:
                    pos = comp.index(True)
                else:
                    pos = None
                    if pos is not None:
                        if pos != lm_pos:
                            qc.cx(ctl[offset+pos],ctl[offset+lm_pos])
                        else:
                            indices = [i for i, x in enumerate(pattern) if x == '1']
                            for idx in indices[1:]:
                                qc.cx(ctl[offset+idx],ctl[offset+lm_pos])
                #check parity
                if pattern.count('1') % 2 == 0:
                    #inverse
                    qc.cu3(-2*theta_angle,0,0,ctl[offset+lm_pos],tgt[count])
                else:
                    qc.cu3(2*theta_angle,0,0,ctl[offset+lm_pos],tgt[count])
                last_pattern = pattern
            
        
        elif n == 2:
            #print("Bit string of length {}".format(n))
            ccry(2*theta,ctl[0],ctl[1],tgt[0],qc)

        elif n == 1:
            qc.cu3(2*theta,0,0,ctl[0],tgt[0])
        else:
            raise ValueError("Register size storing MSB must be of size > 1")
        for count,_ in enumerate(bitstring):
            if _ == '0':
                qc.x(ctl[count])

        
    def set_num(self,ctl,tgt,number,ctl_nums):
        '''Write a number encoded as a binary string to the tgt register
        
        Args:
        ctl : QRegister with controlqubits
        tgt : QRegister that number is written to
        number (int): number to encode
        ctl_nums (array): range of integers that spec. idx of ctl qubits'''
        def generate_bit_string(number,size):
            bitstring = "{:b}".format(number+1)
            #enlong with 0s 
            bitstring = (size-len(bitstring))*'0' + bitstring
            return list(reversed(bitstring))
        #generate bitstring of length len(tgt)
        bitstring = generate_bit_string(number,len(tgt))
        
        n = len(ctl_nums)#np.max(ctl_nums)-np.min(ctl_nums)
        offset = int(np.min(ctl_nums))
        #print(ctl_nums,"generated bitstring for number {}".format(number),":",bitstring,"offset",offset)
        qc = self._circuit
        if n>=3:
            from sympy.combinatorics.graycode import GrayCode
            gray_code = list(GrayCode(n).generate_gray())
            last_pattern = None
            #angle to construct nth square root of diagonlized pauli x matrix
            #via u3(0,lam_angle,0)
            lam_angle = np.pi/(2**(n-1))
            #transform to eigenvector basis of pauli X
            for count,_ in enumerate(bitstring):
                if _ == '1':
                    qc.h(tgt[count])
            for pattern in gray_code:
                if not '1' in pattern:
                    continue
                if last_pattern is None:
                    last_pattern = pattern
                #find left most set bit
                lm_pos = list(pattern).index('1')
                #find changed bit
                comp = [i!=j for i,j in zip(pattern,last_pattern)]
                if True in comp:
                    pos = comp.index(True)
                else:
                    pos = None
                if pos is not None:
                    if pos != lm_pos:
                        qc.cx(ctl[offset+pos],ctl[offset+lm_pos])
                    else:
                        indices = [i for i, x in enumerate(pattern) if x == '1']
                        for idx in indices[1:]:
                            qc.cx(ctl[offset+idx],ctl[offset+lm_pos])
                #check parity
                if pattern.count('1') % 2 == 0:
                     #inverse
                    for count,_ in enumerate(bitstring):
                        if _ == '1':
                            qc.cu3(0,-lam_angle,0,ctl[offset+lm_pos],tgt[count])
                else:
                    for count,_ in enumerate(bitstring):
                        if _ == '1':
                            qc.cu3(0,lam_angle,0,ctl[offset+lm_pos],tgt[count])
                last_pattern = pattern
            for count,_ in enumerate(bitstring):
                if _ == '1':
                    qc.h(tgt[count])
           

        elif n == 2:
            for count,_ in enumerate(bitstring):
                if _ == '1':
                    qc.ccx(ctl[offset],ctl[offset+1],tgt[count])

        elif n == 1:
            for count,_ in enumerate(bitstring):
                 if _ == '1':
                     qc.cx(ctl[offset],tgt[count])

        else:
            raise ValueError("could not set MSB register")
               
               
        
        

    def set_msb(self,msb,ev,msb_num,last_iteration=False):
        ##print("MSB ",msb_num)
        qc = self._circuit
        if last_iteration:
            if msb_num == 1:
                qc.x(ev[0])
                qc.cx(ev[0],msb[1])
                qc.x(ev[0])
            elif msb_num == 2:
                qc.x(ev[0])
                qc.x(ev[1])
                qc.ccx(ev[0],ev[1],msb[2])
                qc.x(ev[1])
                qc.x(ev[0])
            elif msb_num > 2:
                for idx in range(msb_num):
                    qc.x(ev[idx])
                self.nc_toffoli(ev,msb[msb_num],int(msb_num),0)
                for idx in range(msb_num):
                    qc.x(ev[idx])

                #qc.x(ev[0])
                #qc.x(ev[1])
                #qc.ccx(ev[0],ev[1], msb[0])
                #idx = 1
                #for idx in range(2, msb_num):

                #    qc.x(ev[idx])
                #    qc.ccx(ev[idx], msb[idx - 2], msb[idx - 1]) 
                #qc.cx(msb[idx- 1], msb[msb_num])


                #for idx in range(msb_num-1, 1, -1):
                #    qc.ccx(ev[idx ], msb[idx - 2], msb[idx - 1])
                #    qc.x(ev[idx])

                #qc.ccx(ev[0], ev[1], msb[0])
                #qc.x(ev[1])
                #qc.x(ev[0])
    
        elif msb_num == 0: qc.cx(ev[0],msb[0])
        elif msb_num == 1:
            qc.x(ev[0])
            qc.ccx(ev[0],ev[1],msb[1])
            qc.x(ev[0])

        elif msb_num > 1:
            for idx in range(msb_num):
                qc.x(ev[idx])
            self.nc_toffoli(ev,msb[msb_num],int(msb_num+1),0)
            for idx in range(msb_num):
                qc.x(ev[idx])

            #qc.x(ev[0])
            #qc.x(ev[1])
            #qc.ccx(ev[0],ev[1], msb[0])
            #idx = 1
            #for idx in range(2, msb_num):
            #
            #    qc.x(ev[idx])
            #    qc.ccx(ev[idx], msb[idx - 2], msb[idx - 1])

            ##print(idx,msb_num)
            ##print(ev[2])
            ##print(msb[idx- 1])
            ##print(msb[idx])
            #qc.ccx(ev[msb_num],msb[idx- 1], msb[msb_num])


            #for idx in range(msb_num-1, 1, -1):
            #    qc.ccx(ev[idx ], msb[idx - 2], msb[idx - 1])
            #    qc.x(ev[idx])

            #qc.ccx(ev[0], ev[1], msb[0])
            #qc.x(ev[1])
            #qc.x(ev[0])
    def _set_measurement(self):
        qc = self._circuit
        try:
            qc['c_ev']
        except:
            self.c_ev = ClassicalRegister(self.k,'c_ev')
            #self.c_msb = ClassicalRegister(int(np.ceil(np.log2(self.k-self.n+2))),'c_msb')
            self.c_anc = ClassicalRegister(1, 'c_anc')
            
            qc.add(self.c_ev)
            qc.add(self.c_anc)
            #qc.add(self.c_msb)
        #qc.measure(self.msb,self.c_msb)
        qc.measure(self.anc, self.c_anc)

        qc.measure(self.ev,self.c_ev)

    #def _uncompute_msbreg(self,max_msb):
    #    while max_msb >= 0:
    #        self.set_msb(self.msb,self.ev,int(max_msb))
    #        max_msb -= 1
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
                    self.set_msb(self.msb,self.ev,int(old_msb))
                    #self.set_msb(self.msb, self.ev, int(old_msb))
                    #break
                old_msb = msb
                if msb+self.n == self.k:
                    self.set_msb(self.msb,self.ev,int(msb),last_iteration=True)
                else:
                    self.set_msb(self.msb,self.ev,int(msb),last_iteration=False)
            theta = 2*np.arcsin(np.min(lambda_ar)/lambda_ar[_])
            ##print("Pattern",pattern_ar[_],"Theta:",theta,"###",np.sin(theta/2),"lambda:",lambda_ar[_],"Minimal:",np.min(lambda_ar))
            #if lambda_ar[_] != 0.625:
            #    continue
           
            #self._circuit.ry(2*np.arcsin(0.5),self.anc[0])
            self.n_controlled_rotation_optimized(self.ev, self.msb, self.anc, pattern_ar[_], theta, int(msb))
            #if _ == 0: break
        self.set_msb(self.msb,self.ev,int(msb),last_iteration=True)
        
        if self.measure:
            self._set_measurement()
            #self.draw()
            #print('Circuit length is roughly: {}'.format(len(self._circuit.qasm().split('\n'))))
            return self._execute_rotation()
        if len(self.initial_params['state_vector']) == 0:
            self.draw()


    def set_up_and_execute_optimized_circuit(self,k,n):
        if len(self.initial_params['state_vector'])!=0:
            self._construct_initial_state()
        pattern_ar,lambda_ar,msb_ar = classic_approx(k, n)
        old_msb = None
        qc = self._circuit
        for _,msb in enumerate(msb_ar):
            #if k-msb < 2:
            #    break
            #if msb <= 0:
            #    continue
            #print("MSB_:",msb)
            if old_msb != msb:
                if old_msb != None:
                    1
                    for counter in range(old_msb):
                        qc.x(self.ev[counter])
                    self.set_num(self.ev,self.msb,int(old_msb),list(range(old_msb+1)))
                    for counter in range(old_msb):
                        qc.x(self.ev[counter])

                    #self.set_msb(self.msb, self.ev, int(old_msb))
                    #break
                old_msb = msb
                if msb+self.n != self.k:
                    for counter in range(msb+1):
                        qc.x(self.ev[counter])
                    
                    self.set_num(self.ev,self.msb,int(msb),list(range(msb+1)))
                    for counter in range(msb+1):
                        qc.x(self.ev[counter])

                else:
                    for counter in range(msb):
                        qc.x(self.ev[counter])
                    
                    self.set_num(self.ev,self.msb,int(msb),list(range(msb)))
                    for counter in range(msb):
                        qc.x(self.ev[counter])

                
                #self.set_msb(self.msb,self.ev,int(msb),last_iteration=True)
                # else:
                #    self.set_num(self.ev,self.msb,int(msb),range(msb+1))
                #    #self.set_msb(self.msb,self.ev,int(msb),last_iteration=False)
            theta = np.arcsin(np.min(lambda_ar)/lambda_ar[_])
            #if lambda_ar[_] != 2**-self.k:
            #    continue
            ##print("Pattern",pattern_ar[_],"Theta:",theta,"###",np.sin(theta/2),"lambda:",lambda_ar[_],"Minimal:",np.min(lambda_ar))
            #if lambda_ar[_] != 0.625:
            #    continue
           
            #self._circuit.ry(2*np.arcsin(0.5),self.anc[0])
            self.n_controlled_rotation_optimized_v2(self.ev, self.msb, self.anc, pattern_ar[_], theta, int(msb))
            #if _ == 8: break
        #self.set_msb(self.msb,self.ev,int(msb),last_iteration=True)
        for counter in range(msb):
                        qc.x(self.ev[counter])
        self.set_num(self.ev,self.msb,int(msb),list(range(msb)))
        for counter in range(msb):
                        qc.x(self.ev[counter])
                

        if self.measure:
            self._set_measurement()
            #self.draw()
            #print('Circuit length is roughly: {}'.format(len(self._circuit.qasm().split('\n'))))
            return self._execute_rotation()
        if len(self.initial_params['state_vector']) == 0:
            self.draw()


    def _execute_rotation(self):
        shots = 16000
        from qiskit import available_backends
        ###print(available_backends())
        result = execute(self._circuit, backend='local_qasm_simulator', shots=shots).result()
        counts = result.get_counts(self._circuit)
        ###print(np.argmax(np.square(result.get_statevector())),np.max(np.square(result.get_statevector())))
        rd = result.get_counts(self._circuit)
        rets = sorted([[rd[k], k, k] for k in rd])[::-1]
        #return rets
        for d in rets:
            print(d)
            d[0] /= shots
            d[0] = np.sqrt(d[0])
            # split registers which are white space separated and decode
            c1, c2 = d[2].split()
            #print(c1,c2)
            c2_ = sum([2**-(i+1) for i, e in enumerate(reversed(c2)) if e =="1"])
            ###print(c2_)


            d[2] = ' '.join([c1, str(c2_)])
        """
        for d in rets:
            d[0] /= shots
        """
        return rets

    def draw(self):
        drawer(self._circuit)
        plt.show()
###print(classic_approx(3,1))
