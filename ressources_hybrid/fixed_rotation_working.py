import numpy as np
import itertools
import matplotlib.pyplot as plt
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer
from qiskit_aqua import get_initial_state_instance
from qiskit_addon_jku import JKUProvider
from qiskit.wrapper._wrapper import _DEFAULT_PROVIDER

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


def classic_approx_v2(k, n,m):
    '''Calculate error of arcsin rotation using k bits fixed point numbers and n bit accuracy'''
    pattern_array = []
    from collections import OrderedDict
    output = OrderedDict()
    n_ = n  # copy of n
    for msb in range(k - 1, n-1, -1):
            #if msb!=k-2: continue
            vec = ['0'] * k
            vec[msb] = '1'

            for pattern_ in itertools.product('10', repeat=m):
                app_pattern_array = []
                lambda_array = []
                msb_array = []
    
                for appendpat in itertools.product('10',repeat=n-m):
                    pattern = pattern_+appendpat
                    #print(pattern)
                    vec[msb - n:msb] = pattern
                    #print(vec,pattern,pattern_,appendpat,get_est_lamb(vec.copy(),msb,n))
                    e_l = get_est_lamb(vec.copy(), msb, n)
                    l = bin_to_num(vec)
                    lambda_array.append(get_est_lamb(vec.copy(), msb, n))
                    msb_array.append(msb)
                    if len(pattern) < n_:
                        raise ValueError()
                        print("Still active, this part")
                        pattern = list(pattern) + ['0']*(n_-len(pattern))
                   
                    app_pattern_array.append(list(reversed(appendpat)))
                try:
                    
                    prev_res = output[k-msb-1]
                    
                except:
                    prev_res = []
                output.update({k-msb-1:prev_res+[(list(reversed(list(pattern_))),app_pattern_array,lambda_array)]})
               
    vec = ['0'] * k

    #last iterations
    for pattern_ in itertools.product('10', repeat=m):
        app_pattern_array = []
        lambda_array = []
        msb_array = []
        
        for appendpat in itertools.product('10',repeat=n-m):
            pattern = pattern_+appendpat
            #print(pattern)
            if not '1' in pattern:
                continue
            vec[msb - n:msb] = list(pattern)
            e_l = get_est_lamb(vec.copy(), msb, n)
            l = bin_to_num(vec)
            lambda_array.append(get_est_lamb(vec.copy(), msb, n))
            msb_array.append(msb-1)
            if len(pattern) < n_:
                print("Still active, this part")
                raise ValueError()
                pattern = list(pattern) + ['0'] * (n_ - len(pattern))
            app_pattern_array.append(list(reversed(appendpat)))
        try:
            prev_res = output[k-msb]
            
        except:
            prev_res = []
            
        output.update({(k-msb):prev_res+[(list(reversed(list(pattern_))),app_pattern_array,lambda_array)]})      
    #print(output)
    return output




class hybrid_rot(object):
    def __init__(self,k,n,initial_params,measure):
        self.k = int(k)
        self.n = int(n)
        self.anc = QuantumRegister(1,'anc')
        self.workq = QuantumRegister(1,'workq')
        self.msb = QuantumRegister(1,'msb')#QuantumRegister(k-n+1,'msb')
        self.ev = QuantumRegister(k,'ev')
        self._circuit = QuantumCircuit(self.ev,self.msb,self.anc,self.workq)#,self.workq,self.msb,self.anc)
        self.initial_params = initial_params
        self.measure = measure
        msb_num = 3

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



    def set_msb(self,msb,ev,msb_num,last_iteration=False):
        qc = self._circuit
        if last_iteration:
            print("finishing soon",msb_num)
            if msb_num == 1:
                qc.x(ev[0])
                qc.cx(ev[0],msb[0])
                qc.x(ev[0])
            elif msb_num == 2:
                qc.x(ev[0])
                qc.x(ev[1])
                qc.ccx(ev[0],ev[1],msb[0])
                qc.x(ev[1])
                qc.x(ev[0])
            elif msb_num > 2:
                for idx in range(msb_num):
                    qc.x(ev[idx])
                self.nc_toffoli(ev,msb[0],int(msb_num),0)
                for idx in range(msb_num):
                    qc.x(ev[idx])
            else:
                raise ValueError()
        elif msb_num == 0: qc.cx(ev[0],msb)
        elif msb_num == 1:
            qc.x(ev[0])
            qc.ccx(ev[0],ev[1],msb[0])
            qc.x(ev[0])

        elif msb_num > 1:
            for idx in range(msb_num):
                qc.x(ev[idx])
            self.nc_toffoli(ev,msb[0],int(msb_num+1),0)
            for idx in range(msb_num):
                qc.x(ev[idx])


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

   
    def _construct_initial_state(self):
        qc = self._circuit

        self._state_in = get_initial_state_instance('CUSTOM')
        self._state_in.init_params(self.initial_params)
        qc += self._state_in.construct_circuit('circuit', self.ev)
       
    

    def set_bit_pattern(self,pattern,tgt,offset):
        qc = self._circuit
        for c,i in enumerate(pattern):
            if i=='0':
                qc.x(self.ev[int(c+offset)])

        if len(pattern)>2:
            self.nc_toffoli(self.ev,tgt,len(pattern),int(offset))
        elif len(pattern)==2:
            qc.ccx(self.ev[offset],self.ev[offset+1],tgt)
        elif len(pattern)==1:
            qc.cx(self.ev[offset],tgt)

        for c,i in enumerate(pattern):
            if i=='0':
                qc.x(self.ev[int(c+offset)])


    def ccry(self,theta,control1,control2,target):
            '''Implement ccRy gate using no additional ancillar qubits'''
            #double angle because the rotation is defined with theta/2
            theta = 2 * theta
            qc = self._circuit
            theta_half = theta / 2
            qc.cu3(theta_half,0,0,control2,target)
            qc.cx(control1,control2)
            qc.cu3(- theta_half, 0, 0, control2, target)
            qc.cx(control1, control2)
            qc.cu3(theta_half, 0, 0, control1, target)

    def set_up_and_execute_optimized_circuit(self,k,n):
        if len(self.initial_params['state_vector'])!=0 :
            self._construct_initial_state()
        m = int(np.ceil(n/2))
        approx_dict = classic_approx_v2(k,n,m)
        old_msb = None
        allpattern = []
        for _,msb in enumerate(list(approx_dict.keys())):#enumerate(list(reversed(list(approx_dict.keys())))):
            pattern_map = approx_dict[msb] 
            if old_msb != msb:
                if old_msb != None:
                    self.set_msb(self.msb,self.ev,int(old_msb))
                old_msb = msb
                if msb+self.n == self.k:
                    self.set_msb(self.msb,self.ev,int(msb),last_iteration=True)
                else:
                    self.set_msb(self.msb,self.ev,int(msb),last_iteration=False)
            offset_mpat = msb+(n-m) if msb < self.k-self.n else msb+n-m-1 
           
            for mainpat,subpat,lambda_ar in pattern_map:
                self.set_bit_pattern(mainpat,self.workq[0],offset_mpat+1)
                for subpattern,lambda_ in zip(subpat,lambda_ar):
                    theta = np.arcsin(2**int(-self.k)/lambda_)  
                    offset = msb+1 if msb < self.k-self.n else msb
                   
                    self.ccry(theta/2,self.workq[0],self.msb[0],self.anc[0])

                    self.set_bit_pattern(subpattern,self.anc[0],offset)

                    self.ccry(-theta/2,self.workq[0],self.msb[0],self.anc[0])

                    self.set_bit_pattern(subpattern,self.anc[0],offset)
                    #break
                    print(msb,mainpat,subpattern,lambda_,np.sin(theta))
                    vec = [0]*self.k
                    if self.k!=self.n+msb:
                        vec[msb]=1
                    vec[offset:offset+(n-m)] = subpattern
                    vec[offset_mpat:offset_mpat+len(mainpat)] =mainpat
                self.set_bit_pattern(mainpat,self.workq[0],offset_mpat+1)
               
        self.set_msb(self.msb,self.ev,int(msb),last_iteration=True)
        if self.measure:
            self._set_measurement()
            if len(self.initial_params['state_vector'])==0:
                self._draw()
            return self._execute_rotation()
        
        
       




    def _execute_rotation(self):
        shots = 8000
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
            c2_ = sum([2**-(i+1) for i, e in enumerate(reversed(c2)) if e =="1"])
            d[2] = ' '.join([c1, str(c2_)])
        return rets

    def _draw(self):
        drawer(self._circuit)
        plt.show()

#classic_approx_v2(5,3,2)
