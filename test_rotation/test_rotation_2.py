def QPE_norm(y,w,k,n,t):
    
    r = np.abs(sum([(1-np.exp(1j*(2**k*wi*t-2*np.pi*y))) /
                     (1-np.exp(1j*(wi*t-2*np.pi*y/2**k))) for wi in w]))
    r[np.isnan(r)]=2**k
    r = 2**(-2*k)*r**2
    return sum(r)
    

def is_resolvable(num,evo_time,num_bits):
    scale_num = evo_time*num/2/np.pi*2**num_bits
    if int(scale_num)!=scale_num:
           return False
    else:
           return True
           
def QPE_theory_ket(y,w,k,n,t,vec,neg_evals=True):
    #output: vector with ampl. of ket for each vector entry
    resolvable = is_resolvable(w,t,k)
    x = None
    if neg_evals:
        y_ = np.arange(-2**(k-1),2**(k-1),1)
    else:
        y_ = y
    x_ =y_/t*2*np.pi/2**k 
    if not resolvable:
        qpe_fac = 1/2**(k)*( (1-np.exp(1j*(2**k*w*t-2*np.pi*y))) /
                            (1-np.exp(1j*(w*t-2*np.pi*y/2**k))) )
    else:
        qpe_fac = np.array([0]*len(y))
        idx = np.where(x_==w)[0]
        qpe_fac[idx] = 1
    outvec = np.zeros((len(y),len(vec))).astype(complex)
    
    
    for i,entr in enumerate(vec.tolist()):
        outvec[:,i] =qpe_fac*entr
        if x is None:
            x = x_
        else:
            x = np.dstack((x,x_))
    if neg_evals and not resolvable:
        c = int(len(y)/2)
        _  = outvec[c:,:].copy()
        outvec[c:,:] = outvec[:c,:]
        outvec[:c,:] = _
    
    return x[0,:,:],outvec

def superpose_QPE_kets(x_in,w_ar,k,n,t,vec_ar,neg_evals=True):
    '''assume equal superposition of vec'''
    output = np.zeros((len(x_in),len(vec_ar[0,:]))).astype(complex)
    norm = np.sqrt(QPE_norm(x_in,w_ar,k,n,t))
    for idx in range(len(w_ar)):
        w_i = w_ar[idx]
        v_i = vec_ar[:,idx]
        x,res = QPE_theory_ket(x_in,w_i,k,n,t,v_i,neg_evals)

        output += res
    output /= norm
    return (x.flatten(),output.flatten())


def test_with_QPE(config):
    config_implemented = [
         'QPE_standalone',
         'QPE_globalphase_test',
         'QPE_ROT',
        ]
    test_config = config['test']
    for setting in test_config:
        if setting not in config_implemented:
            raise ValueError("Invalid configuration for test function: {}".format(setting))


    qpe_config = config['QPE']
    qpe_param =qpe_config['param']
    if 'matrix' not in list(qpe_config.keys()):
        if any([val not in list(qpe_config.keys())
                for val in ['EVmin','EVmax','N','sparsity']]):
               raise ValueError("Missing parameter for matrix generation")
    N = qpe_config['N']
    EVmin = qpe_config['EVmin']
    EVmax = qpe_config['EVmax']
    sparsity = qpe_config['sparsity']

    matrix = random_hermitian(N, eigrange=[EVmin,EVmax], sparsity=sparsity)

    k = qpe_param['algorithm']['num_ancillae']
    w, v = np.linalg.eigh(matrix) 
    if any([w[i]<0 for i in range(n)]):
        nege = True
        print("Negative EV present")
    else:
        nege = False
        print("Only positive EV present")
        
    #set explicit pos / neg EV
    #nege = True
    qpe_param['algorithm']['negative_evals'] = nege
    print("Matrix has Eigenvalues/vector:")
    print('\n\n'.join([str(w[i])+':'+'('+' '
                       .join(map(str,np.round(v[:,i],4).tolist()))+' )' for i in range(n)]))
    print("#"*20)
