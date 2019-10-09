#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  5 17:51:05 2019

@author: henrikdreyer

Indexing convention for all tensors

   2-A-3
     |
     1
     
     1
     |
   2-A-3
"""

from ncon import ncon
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.operators import Operator


def normalize(x):
    """
    INPUT:
        x(vector):input state
    OUPUT
        y(vector): x normalized
    """
    
    y=x/np.linalg.norm(x)
    return y
    
def extract_phase(x):
    """
    INPUT:
        x(vector): input state
    OUPUT
        y(vector): same state but the first coefficient is real positive
    """
    y=x*np.exp(-1j*np.angle(x[0]))
    
    return y

def complement(V):
    """
    INPUT:
        V(array): rectangular a x b matrix with a > b
    OUTPUT
        W(array): columns = orthonormal basis for complement of V
    """
    W=embed_isometry(V)
    W=W[:,V.shape[1]:]
    
    return W


def find_orth(O):
    """
    Given a non-complete orthonormal basis of a vector space O, find
    another vector that is orthogonal to all others
    """
    rand_vec=np.random.rand(O.shape[0],1)
    A=np.hstack((O,rand_vec))
    b=np.zeros(O.shape[1]+1)
    b[-1]=1
    x=np.linalg.lstsq(A.T,b,rcond=None)[0]
    return x/np.linalg.norm(x)


def embed_isometry(V,defect=0):
    """
    V (array): rectangular a x b matrix with a > b with orthogonal columns
    defect (int): if V should not be filled up to a square matrix, set defect
                  s.t. the output matrix is a x (a-defect)
    Complete an isometry V with dimension a x b to a unitary
    with dimension a x a, i.e.
    add a-b normalized columns to V that are orthogonal to the first a columns
    """

    a=V.shape[0]
    b=V.shape[1]
    
    for i in range(a-b-defect):
        x=np.expand_dims(find_orth(V),axis=1)
        V=np.hstack((V,x.conj()))
    return V


def test_isometry(V):
    """
    check if
    
        ---V------         ----
       |   |              |
       |   |           =  |
       |   |              |
        --V.conj--         ----
        
    is true
    """
    VdaggerV=ncon([V,V.conj()],([1,2,-1],[1,2,-2]))
    return np.allclose(VdaggerV,np.eye(V.shape[2]))
        
def create_random_tensors(N,chi,d=2):
    """
    N (int): number of sites
    d (int): physical dimension. Default = qubits
    chi (int): bond dimension
    Creates a list of N random complex tensors with bond dimension chi and
    physical dimension d each
    """
    
    A=[]
    for _ in range(N):
        A.append(np.random.rand(d,chi,chi)+1j*np.random.rand(d,chi,chi))
    return A



def convert_to_isometry(A,phi_initial,phi_final):
    """
    Input:
    A (list): list of tensors of shape (d, chi, chi)
    phi_initial (vector): (chi,1)
    phi_final (vector): (chi,1)
    Takes as input an MPS of the form
    
    PSI =  phi_final - A_N - A_{N-1} - ... - A_1 - phi_initial
                        |      |              |
                 
    and creates a list of isometries V_1, ... V_N and new vectors
    phi_initial_out and phi_final_out,s.t.
    
    PSI = phi_final' - V_N - V_{N-1} - ... - V_1 - phi_final_out
                        |      |              |
    
    the Vs are isometries in the sense that
    
        ---V------         ----
       |   |              |
       |   |           =  |
       |   |              |
        --V.conj--         ----
        
    Return:
    isometries (list): list of isometric tensors that generates the same state
    phi_initial (vector): bew right boundary
    phi_final (vector): bew left boundary
    Ms (list): list of intermediary Schmidt matrices
    """
    
    N=len(A)
    chi=A[0].shape[1]
    d=A[0].shape[0]
    #Construct isometries from N to 1
    A=A[::-1]
    
    
    """
    SVD:
        phi_final - A_N - 2   =  U_N - M_N - 2
                     |            |
                     1            1
                   
        then append phi_final to disentangle
        
        2-U_N-3   =  2-phi_final     U_N - 3
           |                          |
           1                          1
    """
    left_hand_side=ncon([phi_final,A[0]],([1],[-1,1,-2]))
    U,S,VH=np.linalg.svd(left_hand_side,full_matrices=True)
    S=np.append(S,np.zeros(max(0,chi-S.shape[0])))
    M=np.dot(np.diag(S),VH)
    Ms=[M]
    
    U=ncon([U,phi_final],([-1,-3],[-2]))
    U=U.reshape(chi*d,d)
    U=embed_isometry(U,defect=chi*(d-1))
    #First unitary can have dimension d<chi, so add orthogonal vectors
    
    isometries=[U]
    
    """
    from N to 1, SVD successively:
        
       2 - M_k - V_{k-1} - 3  =  2- U_{k-1} - M_{k-1} - 3
                  |                   |
                  1                   1
    """
    for k in range(1,N):
        left_hand_side=ncon([M,A[k]],([-2,1],[-1,1,-3]))
        left_hand_side=left_hand_side.reshape(chi*d,chi)
        U,S,VH=np.linalg.svd(left_hand_side,full_matrices=False)
        M=np.dot(np.diag(S),VH)
        Ms.append(M)
        isometries.append(U)

    phi_initial_out=np.dot(M,phi_initial)
    phi_final_out=phi_final
    isometries=isometries[::-1]
    Ms=Ms[::-1]
    
    return isometries, phi_initial_out, phi_final_out, Ms
    


def create_statevector(A,phi_initial,phi_final,qiskit_ordering=False):
    """
    INPUT:
    A (list): list of N MPS tensors of size (d,chi,chi)
    phi_initial (vector): right boundary condition of size (chi,1)
    phi_final (vector): left boundary condition of size (chi,1)

    RETURNS:
    Psi(vector): the vector 
        phi_final - A_N - A_{N-1} - ... - A_1 - phi_initial
                     |      |              |
                     N      N-1            1   [N.B.: Reversed Qiskit Ordering]
                         B L O C K E D
    
    schmidt_matrix (array, optional): the matrix
            2 - A_N - A_{N-1} - ... - A_1 - phi_initial
                 |      |              |
                             1
            This is useful to check, for the unitary MPS, if after the 
            application of A_N, the state is a product state between ancilla
            and physical space (the schmidt_matrix has rank 1 in this case).
            This must be the case, in order to implement successfully on the
            quantum computer.
    """
    N=len(A)
    chi=A[0].shape[1]
    d=A[0].shape[0]
    
    
    Psi=ncon([phi_initial, A[0]], ([1],[-1,-2,1]))

    for i in range(1,N):
        Psi=ncon([Psi,A[i]],([-1,1],[-2,-3,1]))
        Psi=Psi.reshape((d**(i+1),chi))
    
    if qiskit_ordering:
        Psi=Psi.reshape(np.append(d*np.ones(N,dtype=int),chi))
        Psi=ncon([Psi],-np.append(list(range(N,0,-1)),N+1))
        Psi=Psi.reshape(d**N,chi)
        
    schmidt_matrix=Psi
    Psi=np.dot(Psi,phi_final)
    
    return Psi, schmidt_matrix


def isunitary(U):
    """
    INPUT:
    U (array)
    OUPUT:
    flag (bool)
    """
    
    if np.allclose(np.eye(len(U)), U.dot(U.T.conj())):
        return True
    else:
        return False


def isometry_to_unitary(V):
    """
    INPUT:
    V(tensor):  tensor with indices
    
                   2-V-3
                     |
                     1
    
                that is isometric in the sense that
                    
                ---V------         ----
               |   |              |
               |   |           =  |
               |   |              |
                --V.conj--         ----
                
    OUTPUT:
    U(matrix): unitary matrix that fulfills
            
                |0>
                 |
                -U-    =    -V-
                 |           |
                 
                 with leg ordering
                 
                      3
                      |
                    2-U-4
                      |
                      1
    """
    chi=V.shape[1]
    d=V.shape[0]
    Vp=V.reshape(chi*d,chi)
    W=complement(Vp)

    U=np.zeros((chi*d,d,chi),dtype=np.complex128)
    U[:,0,:]=Vp
    U[:,1,:]=W
    U=U.reshape(chi*d,chi*d)

    return U    

def MPS_to_circuit(A, phi_initial, phi_final):
    """
    INPUT:
    A (list): list of N MPS tensors of size (d,chi,chi)
    phi_initial (vector): right boundary condition of size (chi,1)
    phi_final (vector): left boundary condition of size (chi,1)
    keep_ancilla (bool, optional):  the MPS is generated via an ancilla
                                    that disentangles at the end of the
                                    procedure. By default, the ancilla is
                                    measured out and thrown away at the end.
                                    Set to True to keep the ancilla
                                    (in the first ceil(log(chi)) qubits)

    OUTPUT:
    qc(QuantumCircuit): the circuit that produces the MPS
    
         q0...qn   phi_final - A_{N-1} - A_{N-2} - ... - A_0 - phi_initial
                                 |          |             |
                               q_{n+N}  q_{n+N-1}       q_{n+1}
                               
                   where n = ceil(log2(chi)) is the number of ancilla qubits
    
    reg(QuantumRegister):   the register in which the MPS wavefunction sits
                            (to distinguish from the ancilla)
                            
    N.B. By construction, after applying qc the system will be in a product
         state between the first ceil(log2(chi)) qubits and the rest.
         Those first qubits form the ancilla register and the remaining qubits
         are in the QuantumRegister 'reg'.
         The ancilla is guaranteed to be in phi_final.
    """
    N=len(A)
    chi=A[0].shape[1]
    d=A[0].shape[0]
    
    #Normalize boundary conditions
    phi_final=phi_final/np.linalg.norm(phi_final)
    phi_initial=phi_initial/np.linalg.norm(phi_initial)
    
    #Convert MPS to isometric form
    Vs,phi_initial_U,phi_final_U,Ms=convert_to_isometry(A,phi_initial,phi_final)
    
    
    #Construct circuit
    n_ancilla_qubits=int(np.log2(chi))
    
    ancilla = QuantumRegister(n_ancilla_qubits)
    reg = QuantumRegister(N)
    
    qc=QuantumCircuit(ancilla,reg)
    
    phi_initial_U=phi_initial_U/np.linalg.norm(phi_initial_U)
    qc.initialize(phi_initial_U,range(n_ancilla_qubits))
    
    for i in range(N):
        qubits=list(range(n_ancilla_qubits))
        qubits.append(i+n_ancilla_qubits)
        qc.unitary(Operator(isometry_to_unitary(Vs[i].reshape(d,chi,chi))), qubits)
        
    return qc, reg
        
        

  