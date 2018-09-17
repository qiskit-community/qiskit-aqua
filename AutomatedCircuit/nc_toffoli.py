def nc_toffoli(self,ctl,tgt,n,offset):
    '''Implement n+1-bit toffoli using the approach in Elementary gates'''
        
    assert n>=3 #"This method works only for more than 2 control bits"
        
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