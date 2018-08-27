from qpe import QPE
#from qiskit_aqua import Operator
import numpy as np
from qiskit.tools.qi.qi import state_fidelity
from qiskit import execute
import matplotlib.pyplot as plt

from qiskit_addon_jku import JKUProvider

def qpe_test_fidelity(n, k, n_samples, num_time_slices, expansion_mode, expansion_order):
    """
    n - size of the system
    k - number of ancilla qubits
    matrix - operator matrix, size nxn
    """        
    
    for sample in np.arange(1, n_samples +1):
        test_results = np.zeros((n, 7))
        
        matrix_test = np.zeros((n, n*n+ n + 1))
        
        w = [-1, -1]
        while min(w) <= 0:
            matrix = np.random.random([n, n])+1j*np.random.random([n, n])
            matrix = 4*(matrix+matrix.T.conj())
            w, v = np.linalg.eig(matrix)
        
        hermitian_matrix = True    
        
        for i2 in range(0,n):   
            invec = v[:,i2]   
            backend = 'local_statevector_simulator'#_jku'        
            params = {
                    'algorithm': {
                            'name': 'QPE',
                            'num_ancillae': k,
                            'num_time_slices': num_time_slices,
                            'expansion_mode': expansion_mode,
                            'expansion_order': expansion_order,
                            'hermitian_matrix': hermitian_matrix,
                            'backend' : backend
         
                    },
                    "iqft": {
                        "name": "STANDARD"
                    },
                    "initial_state": {
                        "name": "CUSTOM",
                        "state_vector": invec#[1/2**0.5,1/2**0.5]
                    }}
            #generating quantum state with qpe:        
            qpe = QPE()
            qpe.init_params(params, matrix)    
            circuit, t = qpe._setup_qpe(measure = False)
            res =  execute(circuit, backend = backend)
            qvec = res.result().get_statevector() 
                
            # generating theoretical state vector - 
            #calculating the coefficients for the basis vector:
            coeff_l = np.zeros((2**k,1), dtype = complex)
            
            for l in np.arange(2**k):
                for m in np.arange(2**k):       
                    coeff_l[l] = coeff_l[l] + np.exp((-2.j*np.pi*m*l/(2**k)))*np.exp((1.j*w[i2]*t*m))/(2**k)
                    
                    
            # Mapping that accounts for tensor product in reversed order:        
            tmap = np.zeros((2**k, 2**k))       
            for i in range(2**k):
                bv  = format(i, 'b').zfill(8)
                bv = bv[-k:]            
                if int(bv[-1]) == 1:
                    s = [0,1]
                else: 
                    s = [1, 0]           
                for j in range(2, k+1):
                        if int(bv[-j]) == 1:
                            s = np.tensordot(s,[0,1], axes = 0).reshape((2**j))
                        else: 
                            s = np.tensordot(s, [1,0], axes = 0).reshape((2**j)) 
                tmap[:,i] = s
            
            #Generating theoretical vector mapped onto QPE state vector basis 
            #with reversed tensor product)
            coeff_l_swap = (np.matrix(tmap)*coeff_l)   
            C1 = np.tensordot(invec, coeff_l_swap.round(3), axes = 0).reshape((n*2**k,1))            
            test_results[i2, :] = np.array([n, k, num_time_slices, expansion_order,
                                    state_fidelity(qvec, C1.reshape(len(C1))),0,  len(circuit.qasm().split('\n'))])
    
            matrix_test[i2, : ] = np.concatenate((matrix.reshape(len(matrix)**2), invec.reshape(n), [w[i2]]), axis = 0)
        
        if sample ==1:
            T = test_results
            M = matrix_test
        else:
            T = np.vstack((T,test_results))
            M = np.vstack((M,matrix_test))
    
    
    
    return T , M


def run_over_grid(expansion_m, file_out, file_out_M):    
    test1 = ['n', 'k', 'time slices', 'expansion order', 'avg Fidelity', 'std Fidelity', '~number of gates']
    n = 2
    matrix_t = np.zeros((n, n*n+ n +1))
    for expansion_order in [2]:#[1, 2, 3]:
        for expansion_mode in expansion_m:
            for num_time_slices in [5]:#np.arange(5, 25, 5):    
                for k in [4]:
                    for n in [2]: 
                        qpe_re, M = qpe_test_fidelity(n, k, 1000, num_time_slices, 
                                                 expansion_mode, expansion_order)
                        test1 = np.vstack((test1, qpe_re ))
                        #np.savetxt(file_out, test1[1:,:].astype(float) ,fmt='%.18e', delimiter=' ', header = str(test1[0,:]))
                        print('Completed:', 'n:', n, 'k:', k,'exp ord:', expansion_order, 
                              't slices:', num_time_slices )            
                        matrix_t = np.vstack((matrix_t, M))
    print('Test results: \n', test1)            
    np.savetxt(file_out, test1[1:,:].astype(float) ,fmt='%.18e', delimiter=' ', header = str(test1[0,:]))
    np.savetxt(file_out_M, matrix_t[1:,:].astype(float) ,fmt='%.18e', delimiter=' ', header = str(test1[0,:]))

def plot_fidelity(x_name, y_name, x, y, data_series1, label1, data_series2, label2):    
    plt.scatter(data_series1[:,x], np.abs(data_series1[:,y]), label = label1)
    plt.scatter(data_series2[:,x], np.abs(data_series2[:,y]), c = 'r', label = label2)
    #plt.ylim(0, np.max(data_series1[:,4])*1.2, )
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.grid('on')
    plt.legend()
    plt.show()


if True:
    #run_over_grid(['trotter'], 'test_trotter4.txt')
    #run_over_grid(['suzuki'], 'test_suzuki_final.txt')
    #run_over_grid(['trotter'], 'test_trotter_nsize.txt')
    run_over_grid(['trotter'], 'test_matrices2_f.txt','test_matrices2_m.txt' )

#%%
#Postprocessing

trotter = False
suzuki = True    
    
data = np.genfromtxt('test_suzuki_final.txt',delimiter=' ', skip_header=0, )

data[:, -3] = np.abs(data[:,-3] - 1)
data[:, -2] = data[:, 1]

# Correlation matrix
Corr = np.corrcoef(data.T)

d1 = data[np.where(data[:,2]==5)]
d2 = data[np.where(data[:,2]==10)]

ps = ['n', 'number of ancilla', 'time slices', 'expansion order']
for param in range(1,len(ps)):
    plt.figure()
    plot_fidelity(ps[param],'Fidelity eps', param , -3 , d1, 'time slices=5', d2, 't slices=10')


#%%
def plot_hist(data, param, ser, title):  
    plt.figure()       
    for i in range(1, ser+1):
        plt.subplot(ser,1,i)
        plt.hist(data[np.where(data[:,param]==i),4].T, bins = np.linspace(0,1,50))
        plt.xlim([0,1])
        
    plt.suptitle(title) 
    plt.show()

d1 = data[np.where(data[:,2]==5)]   #time slicing = 5
d2 = data[np.where(data[:,2]==15)]

if True:   
    #plot_hist(data, 1, 8, 'Ef distribution for different ancilla')
    plot_hist(d1[np.where(d1[:,0]==2)], 1, 6, 'Ef distribution, N = 2, time slicing = 5')
    plot_hist(d2[np.where(d2[:,0]==2)], 1, 6, 'Ef distribution, N = 2, time slicing = 15')
    #plot_hist(data[np.where(data[:,0]==4)], 1, 8, 'Ef distribution, N = 4')

#%%
    #influence of time slicing ondistribution
    
plot_hist(data[np.where(data[:,2]==5)], 1, 8, 'Ef distribution t = 5')
plot_hist(data[np.where(data[:,2]==10)], 1, 8, 'Ef distribution t = 10')
plot_hist(data[np.where(data[:,2]==15)], 1, 8, 'Ef distribution t = 15')
#plot_hist(data[np.where(data[:,2]==20)], 1, 8, 'Ef distribution t = 20')

#%%

if trotter:
    
    n2 = data[np.where(data[:,0]==2)]
    n4 = data[np.where(data[:,0]==4)]
    
    plt.figure()
    plt.subplot(1,2, 1)
    for i in [5, 10, 15, 20]:
        plt.scatter(n2[np.where(n2[:,2]==i), 1], n2[np.where(n2[:,2]==i), -1], label = 'time slices:'+ str(i))
    plt.legend()
    plt.xlabel('N. Ancilla')
    plt.ylabel('Gates')
    plt.title('N = 2')
    plt.grid('on')    
    plt.subplot(1,2,2)
    
    for i in [5, 10, 15, 20]:
        plt.scatter(n4[np.where(n4[:,2]==i), 1], n4[np.where(n4[:,2]==i), -1], label = 'time slices:'+ str(i))
    plt.legend()
    plt.xlabel('N. Ancilla')
    plt.ylabel('Gates')
    plt.title('N = 4')
    plt.grid('on')   
    
if suzuki: 
    n1 = data[np.where(data[:,3]==1)]
    n2 = data[np.where(data[:,3]==2)]
    n3 = data[np.where(data[:,3]==3)]
    plt.figure()
    plt.subplot(1,3, 1)
    
    for i in [5, 10, 15]:
        plt.scatter(n1[np.where(n1[:,2]==i), 1], n1[np.where(n1[:,2]==i), -1], label = 'time slices:'+ str(i))
    plt.legend()
    plt.xlabel('N. Ancilla')
    plt.ylabel('Gates')
    plt.title('exp order = 1')
    plt.grid('on') 
    
    plt.subplot(1,3,2)
    for i in [5, 10, 15]:
        plt.scatter(n2[np.where(n2[:,2]==i), 1], n2[np.where(n2[:,2]==i), -1], label = 'time slices:'+ str(i))
    plt.legend()
    plt.xlabel('N. Ancilla')
    plt.ylabel('Gates')
    plt.title('exp order = 2')
    plt.grid('on')      

    plt.subplot(1,3,3)
    for i in [5, 10, 15]:
        plt.scatter(n3[np.where(n3[:,2]==i), 1], n3[np.where(n3[:,2]==i), -1], label = 'time slices:'+ str(i))
    plt.legend()
    plt.xlabel('N. Ancilla')
    plt.ylabel('Gates')
    plt.title('exp order = 3')
    plt.grid('on')      
