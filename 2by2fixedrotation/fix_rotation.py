import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, register, execute#, ccx
from qiskit import register, available_backends, get_backend, least_busy
from qiskit.tools.qi.pauli import Pauli
from qiskit_aqua import Operator, QuantumAlgorithm
from qiskit.tools.visualization import matplotlib_circuit_drawer as drawer
import matplotlib.pyplot as plt
import itertools
from collections import OrderedDict
from IPython.display import display, Markdown, Latex

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


def add_eigenvalue_inversion(qc):
    """add controlled rotations for all possible states of the ev_register

    Args:
        qc QuantumCicruit"""

    regs = qc.get_qregs()
    ev_register =regs['eigs']
    rotqbit = regs['control']
    anc = QuantumRegister(len(ev_register)-1,'workq')
    qc.add(anc)


    reg_len = len(ev_register)

    rotation_dict = OrderedDict() # store number with corr. rotation
    for n, i in enumerate(map(''.join, itertools.product('01', repeat=reg_len))):
        #@TODO: maybe inverse binary_pattern
        #if i not in ['11','01']:
        #    continue
        binary_pattern = [int(_) for _ in i]#list(reversed([int(_) for _ in i])) #
        float_num = get_float_from_bin(binary_pattern,np.arange(-reg_len,0))
        #print(binary_pattern,float_num)
        #exclude 0
        if 1 in binary_pattern:
            rotation_dict.update({float_num:binary_pattern})

    minimal_ev = np.min(list(rotation_dict.keys()))

    #update dict with corresponding angles
    for key in rotation_dict.keys():
        rotation_dict[key] = [rotation_dict[key], 2 *  np.arcsin(minimal_ev / key)] # @TODO maybe round here?





    ###########################
    ###### add to circuit #####
    ###########################
    for in_pat,angle in rotation_dict.values():
        #print(in_pat,angle)
        if not in_pat[0]:
            qc.x(ev_register[0])
        if not in_pat[1]:
            qc.x(ev_register[1])
        qc.ccx(ev_register[0], ev_register[1], anc[0])
        if not in_pat[1]:
            qc.x(ev_register[1])
        if not in_pat[0]:
            qc.x(ev_register[0])

        for i in range(2,reg_len):
        
            if in_pat[i] == 0:
                qc.x(ev_register[i])
            qc.ccx(ev_register[i], anc[i - 2], anc[i - 1])
            if in_pat[i] == 0:
                qc.x(ev_register[i])

        if i != reg_len-1:
            i = 1
        qc.cu3(angle,0,0,anc[i - 1], rotqbit[0])


        for i in range(reg_len-1, 1, -1):
            # print(i)
            if in_pat[i] == 0:
                qc.x(ev_register[i])
            qc.ccx(ev_register[i], anc[i - 2], anc[i - 1])
            if in_pat[i] == 0:
                qc.x(ev_register[i])

        if not in_pat[0]:
            qc.x(ev_register[0])
        if not in_pat[1]:
            qc.x(ev_register[1])
        qc.ccx(ev_register[0], ev_register[1], anc[0])
        if not in_pat[1]:
            qc.x(ev_register[1])
        if not in_pat[0]:
            qc.x(ev_register[0])

    return qc


def add_measurement_gates(qc):
    qregs = qc.get_qregs()
    cregs = qc.get_cregs()
    control_qbit = qregs['control']
    sv = qregs['comp']
    #c_ = ClassicalRegister(2,'test')
    #qc.add(c_)
    c1 = cregs['controlbit']
    c2 = cregs['solution_vector']

    qc.barrier()
    qc.measure(sv, c2)
    qc.measure(control_qbit[0],c1[0])

    #qc.measure(qregs['eigs'],c_)
    return


def generate_matrix():
    """Generate 2x2 matrix with commuting Pauli matrices"""

    X = np.array([[0, 1],
                  [1, 0]])
    Z = np.array([[1, 0],
                  [0, -1]])
    I = np.array([[1, 0],
                  [0, 1]])

    pauli = X  # if np.random.choice(['X','Z'],1) == 'X' else Z

    matrix = 2 * I - 1 * pauli

    # print(matrix)
    return matrix


def print_linsystem(A, k, b):
    out_mat = '$\\begin{pmatrix}' + r'\\'.join(
        ['&'.join([str(A[0, i]), str(A[1, i])]) for i in range(A.shape[1])]) + '\\end{pmatrix}$'
    sol_vec = '$\\begin{pmatrix}a_0' + r'\\' + '{}a_0'.format(np.round(1 / k, 5)) + '\\end{pmatrix}$'
    right_side = '$= \\begin{pmatrix}' + str(b[0]) + r'\\' + str(b[1]) + '\\end{pmatrix}$'

    display(Markdown('## ' + out_mat + sol_vec + right_side))
    equiv = '\n ## '.join(
        ['$a_0 = ' + '{}'.format(b[i] / (A[0, i] + (A[1, i] / k))) + '$' for i in range(A.shape[1])])
    print(equiv)
    display(Markdown('## ' + equiv))

    solutions = []
    for i in range(A.shape[1]):
        x0 = b[i] / (A[0, i] + np.round(A[1, i] / k, 1))
        if np.isfinite(x0):
            solutions.append(x0)
        display(Markdown("## Equation {} ".format(i) + "gave a result of $\\begin{pmatrix}" + r'\\'.join([str(x0),str(1/k*x0)]) + "\\end{pmatrix}$"))

    correct = np.linalg.solve(A,b)
    if isinstance(correct,np.ndarray):
        display(Markdown("## Correct $\\begin{pmatrix}" + r'\\'.join([str(correct[i]) for i in range(A.shape[1])]) + "\\end{pmatrix}$"))
    else:
        display(Markdown("## No solution could be estimated classically"))


"""
n = 4
reg = QuantumRegister(n,'a')
anc = QuantumRegister(n-1,'b')
rotqbit = QuantumRegister(1,'c')
qc = QuantumCircuit(reg,anc,rotqbit)
qc = add_eigenvalue_inversion(qc,reg,anc,rotqbit)
drawer(qc)
plt.show()"""

"""The following routine will generate a matrix by adding multiple Pauli matrices.
We restrict our linear system to be real valued (i.e. not complex) and make use of the Pauli matrices $\sigma_{I}=\begin{pmatrix}1 & 0 \\ 0 & 1 \end{pmatrix}$, 
$\sigma_{x}=\begin{pmatrix}0 & 1 \\ 1 & 0 \end{pmatrix}$, $\sigma_{z}=\begin{pmatrix}1 & 0 \\ 0 & -1 \end{pmatrix}$
Now we have a matrix $A=\sum_i \alpha_{i} \sigma_{l_{i}}\ l_{i}\in \{I,x,z\}$ that is going to specify our linear system!"""