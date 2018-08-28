
import numpy as np
import itertools
import collections
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer

def calculate_correction(num,n,p,min_exp):
    """Estimate bitcode of fixed point number with n digits and decimal point at position p that minimizes
        the distance num to the next smallest power of 2

        Args:
            num float: number that should be resembled
            n int : number of digits to represent correction byte code
            p int : decimal point position (p<n)
            """
    out_num = np.zeros(n)
    orig_num = num
    for i in np.arange(n-1,-1,-1):

        if i >= p:
            #print(num, 2 ** (i - p))
            dev = num-2**(i - p)
            #print(num, 2 ** (i - p), 1 / 2 ** p)
        else:
            #print(num, 1 /  2 ** (p - i))
            dev = num - 1 / 2 ** (p - i)
            #print(num, 1 / 2 ** (p - i), 1 / 2 ** p)

        if dev >= 0:
            num = dev
            out_num[i] = 1

        elif abs(dev)<1/2**p and i != min_exp+p:
            out_num[i] = 1
            break
        #print(num)
    #print(out_num)
    #print(sum([2 **(i-p) for i, e in enumerate((out_num)) if e ==
    #                           1]))
    #print(num, 1/2**p)
    return out_num

def calculate_next_biggest_exp(num):
    return -int(np.ceil(np.log2(num))) if num > 0 else 0


def get_approximation_inv(num,n,p):
    nsmallest = calculate_next_biggest_exp(num)
    diff = 1/num - 2**nsmallest if num > 0 else 0
    #print("Difference",diff)
    correction = calculate_correction(diff,n,p,nsmallest)
    #if num == 0.15625:
    #print("Calculated correction:")
    #print(num,nsmallest,correction,diff)
    return list(correction.astype(int))
def get_float_from_bin(bin,exp_range):
    num = 0
    #print(len(bin),len(exp_range))
    assert len(bin)==len(exp_range), "The supplied list of exponents need to be of same size as the binary pattern"
    for b,e in zip(bin,exp_range):
        if e >= 0:
            num = num + 2**e if b else num
        else:
            num = num + 1/ 2 ** abs(e) if b else num

    return num

def generate_circuit(circuit_info,decimals,accuracy,qc=None):
    """Add gates to circuit

    Args:
        circuit_info OrderedDict: Store (in descending order) input and output binary pattern
        qc QuantumCircuit: Gates are added to this circuit"""
    """
    inputregister = QuantumRegister(n, name="eigenvalue_reg")

    flagbit = QuantumRegister(1, name="flagbit_reg")
    # plus 1 for 2**0
    outputregister = QuantumRegister(n + 1+ accuracy, name="inv_eigenvalue_reg")

    anc = QuantumRegister(n - 1, name="anc_reg_crot")
    qc = QuantumCircuit(inputregister, flagbit, anc, outputregister)
    """

    qregs_dict = qc.get_qregs()
    inputregister = qregs_dict['eigenvalue_reg']
    outputregister = qregs_dict['inv_eigenvalue_reg']
    flagbit = qregs_dict["flagbit_reg"]
    anc = qregs_dict["anc_reg_crot"]
    last_max_exp = None

    ###test
    #qc.add(ClassicalRegister(1))
    #counter = 0
    for floatingnumber,(in_pat,out_pat) in zip(circuit_info.keys(),circuit_info.values()):
        in_pat = list(reversed(in_pat))

        #    continue
        #if floatingnumber != 0.5625:
        #    continue
        #if last_max_exp is not None:
        #    break
        out_pat = list(reversed(out_pat))
        #print(len(out_pat),len(outputregister))
        max_exp = in_pat.index(1)#in_pat[::-1].index(1)

        _ = max_exp
        #print("Max exponent:",max_exp)
        n = len(in_pat)
        #print(floatingnumber,in_pat)
        if floatingnumber < 0.1:
            print(max_exp,"doesnt work",in_pat)
        if ((n-1)-max_exp) < 2:

            break
        #break
        #print("Input",n,"Output",len(out_pat),"Accuracy",accuracy)
        #print(n,"QBITS")
        if last_max_exp != max_exp:
            if last_max_exp is not None:
                #break
                qc.cx(outputregister[n - _ + 1], flagbit)
            #if last_max_exp is not None and last_max_exp < 1:
            #    break

            if 0 != max_exp:
                qc.x(inputregister[0])
            if 1 != max_exp:
                qc.x(inputregister[1])
            qc.ccx(inputregister[0], inputregister[ 1], anc[0])
            if 0 != max_exp:
                qc.x(inputregister[0])
            if 1 !=  max_exp:
                qc.x(inputregister[ 1])


            for i in range(2, min(n,2+accuracy)):
                if i != max_exp:
                    qc.x(inputregister[i])
                qc.ccx(inputregister[i], anc[i - 2 ], anc[i - 1 ])
                if i != max_exp:
                    qc.x(inputregister[i])
            # copy
            qc.x(flagbit)
            #print(outputregister[n - _ + accuracy - 2])
            #print("fails for ", _)

            qc.ccx(flagbit[0], anc[i - 1 ], outputregister[n -_ - 1])
            qc.x(flagbit)
            #print("works")
            # uncompute
            for i in range(min(accuracy+1,n - 1), 1 , -1):
                if i != max_exp:
                    qc.x(inputregister[i])
                qc.ccx(inputregister[i], anc[i - 2 ], anc[i - 1])
                if i != max_exp:
                    qc.x(inputregister[i])

            if 0 != max_exp:
                qc.x(inputregister[0])
            if 1 != max_exp:
                qc.x(inputregister[1])
            qc.ccx(inputregister[0], inputregister[1], anc[0])
            if 0 != max_exp:
                qc.x(inputregister[0])
            if 1 != max_exp:
                qc.x(inputregister[1])

            qc.cx(outputregister[n - 1 - _], flagbit)
            qc.x(flagbit)
            qc.ccx(inputregister[_], flagbit[0], outputregister[n - _])
            qc.x(flagbit)

            #if counter == 1:
            #    break
            #counter = counter + 1
            #qc.x(flagbit[0])
            #qc.ccx(inputregister[n - 1], flagbit[0], outputregister[0])
            #qc.x(flagbit[0])

        #store new bit pattern
        if not in_pat[_+1]:
            qc.x(inputregister[_ + 1])
        qc.ccx(inputregister[_], inputregister[_ + 1], anc[0])
        if not in_pat[_ + 1]:
            qc.x(inputregister[_ + 1])
        #print("compute, starting with", _)
        for i in range(2 + _, min(_ + accuracy+1, n)):
            #print(i, "range from",_ + accuracy+1,n)
            if in_pat[i] == 0:
                qc.x(inputregister[i])
            qc.ccx(inputregister[i], anc[i - 2 - _], anc[i - 1 - _])
            if in_pat[i] == 0:
                qc.x(inputregister[i])
        # copy
        #print(max_exp,"MAX exponent")
        #print(inputregister[max_exp])
        #print(anc[i-1 - _])
        #qc.ccx(inputregister[max_exp],flagbit[0], anc[i-1 - _])
        for idx,on_qbit in enumerate(out_pat):
            if on_qbit:
                #print("adding node for ",idx)
                qc.x(flagbit[0])
                print(i - 1 - _)
                qc.ccx(flagbit[0], anc[i - 1 - _], outputregister[idx])
                #qc.cx(anc[i - 1 - _], outputregister[idx])

                qc.x(flagbit[0])
        # uncompute
        #print("uncompute, starting with", _)

        for i in range(min(n-1,_ + accuracy), 1 + _, -1):
            #print(i)
            if in_pat[i] == 0:
                qc.x(inputregister[i])
            qc.ccx(inputregister[i], anc[i - 2 - _], anc[i - 1 - _])
            if in_pat[i] == 0:
                qc.x(inputregister[i])
        if not in_pat[_ + 1]:
            qc.x(inputregister[_ + 1])
        qc.ccx(inputregister[_], inputregister[_ + 1], anc[0])
        if not in_pat[_ + 1]:
            qc.x(inputregister[_ + 1])

        #qc.cx(anc[n - 2 - _], flagbit)
        #qc.cx(flagbit, anc[n - 2 - _])

        #break
        last_max_exp = max_exp


    qc.cx(outputregister[n - _], flagbit)
    #add max_exp for 1 sig bit




    #add last bit
    for count in range(n-1):
        qc.x(inputregister[count])
    #qc.x(inputregister[_ + 1])
    qc.ccx(inputregister[0], inputregister[ 1], anc[0])
    #qc.x(inputregister[_ + 1])

    for i in range(2 , n):
        qc.ccx(inputregister[i], anc[i - 2 ], anc[i - 1])

    # copy

    # print(outputregister[n - _ + accuracy - 2])
    # print("fails for ", _)
    qc.x(flagbit[0])
    qc.ccx(flagbit[0], anc[i - 1 ], outputregister[0])
    qc.x(flagbit[0])
    # print("works")
    # uncompute
    for i in range( n - 1, 1 , -1):
        qc.ccx(inputregister[i], anc[i - 2 ], anc[i - 1 ])


    qc.ccx(inputregister[0], inputregister[ 1], anc[0])

    for count in range(n-1):
        qc.x(inputregister[count])
    #qc.cx(outputregister[n - _ + 1 + accuracy], flagbit)
    #drawer(qc,filename='corrected_inversion')

    return qc
def design_correction_circuit(n_in,decimals,accuracy,qc):
    """
    Args:
        n_in int: number of qbits for input
        decimals int: number of qbits storing subint information in output
        accuracy int: number of qbits behind highest power of 2 that is taken into account
                        for the input
        qc QuantumCircuit: Circuit that gates are added to
    """
    pattern_dict = {} #store floats as keys and bianry pattern of input and output as values
    print("Decimals:",decimals,"Accuracy:",accuracy)
    ################################################################
    ### classically calculate range of numbers and correction ######
    ################################################################
    _accuracy = accuracy
    #iterate over each qbit and get binary patterns for all combination of the $accuracy number of following qbits
    for max_exp in np.arange(n_in-1,0,-1):#,accuracy-1,-1):
        if max_exp < accuracy:
            _accuracy = _accuracy - 1
        for n, i in enumerate(map(''.join, itertools.product('01', repeat=_accuracy))):
            vec = [0]*n_in

            #print(i,max_exp,_accuracy)
            vec[max_exp]=1
            vec[max_exp-_accuracy:max_exp] = [int(_) for _ in list(str(i))]
            if not '1' in i:
                vec[:max_exp-_accuracy] = [1]*(max_exp-_accuracy)
            #print(n_in,vec)
            floating_num = get_float_from_bin(vec,list(np.arange(-n_in,0,1)))
            #print(vec,get_float_from_bin(vec,list(np.arange(-n_in,0,1))))
            #ignore case where only 0s
            if 1:#"1" in i:
                correction_pattern = get_approximation_inv(floating_num,n_in+decimals+1,decimals)
                #if 1 in correction_pattern:
                pattern_dict.update({floating_num:[vec,correction_pattern]})
            #break
            #if n ==2:
            #    break
    """
    for _,dem_accuracy in enumerate(range(accuracy,1,-1)):
        for max_exp in np.arange(accuracy - 1,1, -1):  # ,accuracy-1,-1):
            for n, i in enumerate(map(''.join, itertools.product('01', repeat=dem_accuracy))):
                vec = [0]*n_in
                #print(vec,max_exp,accuracy)
                vec[max_exp]=1
                vec[max_exp-accuracy:max_exp] = [int(_) for _ in list(str(i))]
                #print(n_in,vec)
                floating_num = get_float_from_bin(vec,list(np.arange(-n_in+_,0,1)))
                #print(vec,get_float_from_bin(vec,list(np.arange(-n_in,0,1))))
                #ignore case where only 0s
                if 1:#"1" in i:
                    correction_pattern = get_approximation_inv(floating_num,n_in+decimals+1,decimals)
                    pattern_dict.update({floating_num:[vec,correction_pattern]})
        #break
        #break
    """
    pattern_dict = collections.OrderedDict(sorted(pattern_dict.items(),reverse=True))
    print('\n'.join([' '.join(list((str(key),str(value)))) for key,value in zip(pattern_dict.keys(),pattern_dict.values())]))
    return generate_circuit(pattern_dict,decimals,accuracy,qc)

#get_approximation_inv(0.75,6,2)

#design_correction_circuit(5,3,2)