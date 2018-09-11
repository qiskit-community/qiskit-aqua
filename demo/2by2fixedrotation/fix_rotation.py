import numpy as np
from qiskit import QuantumRegister
from IPython.display import display, Markdown, Latex

def ccry(theta,control1,control2,target,qc):
    '''construct double controlled rotation'''
    theta_half = theta / 2
    qc.cu3(theta_half,0,0,control2,target)
    qc.cx(control1,control2)
    qc.cu3(- theta_half,0,0,control2,target)
    qc.cx(control1,control2)
    qc.cu3(theta_half,0,0,control1,target)

def add_eigenvalue_inversion(qc):
    regs = qc.get_qregs()
    qr = regs['eigs']

    control_qbit = QuantumRegister(1, 'control')
    qc.add(control_qbit)
    qc.x(qr[0])
    ccry(2*np.arcsin(1),qr[0],qr[1],control_qbit[0],qc)
    qc.x(qr[0])
    ccry(2 * np.arcsin(1/3), qr[0], qr[1], control_qbit[0],qc)

def add_measurement_gates(qc):
    qregs = qc.get_qregs()
    cregs = qc.get_cregs()
    control_qbit = qregs['control']
    sv = qregs['comp']
    c = cregs['c']
    c1 = cregs['controlbit']
    c2 = cregs['solution_vector']

    qc.barrier()
    qc.measure(sv, c2)
    qc.measure(control_qbit[0],c1[0])
    qc.measure(qregs['eigs'],c)
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
    sol_vec = '$\\begin{pmatrix}a_0' + r'\\' + '{}a_0'.format(np.round(1 / k, 1)) + '\\end{pmatrix}$'
    right_side = '$= \\begin{pmatrix}' + str(b[0]) + r'\\' + str(b[1]) + '\\end{pmatrix}$'

    display(Markdown('## ' + out_mat + sol_vec + right_side))
    equiv = '\n ## '.join(
        ['$a_0 = ' + '{}'.format(b[i] / (A[0, i] + np.round(A[1, i] / k, 1))) + '$' for i in range(A.shape[1]) if b[i] != 0])
    # print(equiv)
    display(Markdown('## ' + equiv))

    solutions = []
    for i in range(A.shape[1]):
        if b[i] != 0:
            x0 = b[i] / (A[0, i] + np.round(A[1, i] / k, 1))
            if np.isfinite(x0):
                solutions.append(x0)
            display(Markdown("## Equation {} ".format(i) + "gave a result of $\\begin{pmatrix}" + r'\\'.join([str(x0),str(1/k*x0)]) + "\\end{pmatrix}$"))

    correct = np.linalg.solve(A,b)
    if isinstance(correct,np.ndarray):
        display(Markdown("## Correct $\\begin{pmatrix}" + r'\\'.join([str(correct[i]) for i in range(A.shape[1])]) + "\\end{pmatrix}$"))
    else:
        display(Markdown("## No solution could be estimated classically"))


