import numpy as np
import matplotlib.pyplot as plt
def num_basic_gates(k,n):
    """Calculate number of basic gates (1 Toffolo = 16 basic gates) for hybrid rotation"""

    frac_bit_pat = 16*(2**n*(k-n+1)*(2*(n-1)+1))
    frac_c_reg = 16 * ( -3 * (k-n-2) + 2 * ((k-n)*(k-n+1)/2 -1)) * 2 + ( -3 * (k-n-2) + 2 * ((k-n)*(k-n+1)/2 -1)) * 2
    frac_x_gates = 2**n*(k-n+1)*2
    print("Basic gates for k={},n={}:".format(k,n))
    print("Storing bit patterns:",frac_bit_pat)
    print("Setting MSB register:", frac_c_reg)
    print("X gates to store bit patterns:",frac_x_gates)
    return frac_bit_pat+frac_c_reg+frac_x_gates
"""
def num_qbits(k,n):
    return 2 * k + 1
k_ = np.arange(6,15)
n = 5
gates = []
qbits = []
for k in k_:
    print(num_basic_gates(k,n))
    gates.append(num_basic_gates(k,n))
    qbits.append(num_qbits(k,n))

ax1 = plt.subplot(2,1,1)
ax1.plot(k_,gates)
plt.title('bit accuracy of {} approx. {}'.format(n,2**-(n+1)))
ax2 = plt.subplot(2,1,2)
ax2.plot(k_,qbits)
ax2.set_xlabel('length of EV register')

plt.show()
#print(num_qbits(5,3))

print(num_basic_gates(7,4))
"""


def nc_u(n,num):
    '''Cost to produce a n controlled operation of some operator whose 2**(n-1) root can be
    calculated with num operations'''
    return (2**(n)-1)*num + (2**n-2)

def num_basic_gates(k,n,m):
    return (k-n+1)*2**(n+1) * (nc_u(n,4) + nc_u(int(np.ceil(np.log2(k-n+2))),4))

def improved_basic_gates(k,n,m):
    return (k-n+1)*2**(m) * (2*nc_u(m,4) + 2**(n-m+1)*( nc_u(int(np.ceil(np.log2(k-n+2))),4)+nc_u(int(n-m+1),4) ))

def outlook_improved_basic_gates(k,n,m):
    return 2**(m) * (2*nc_u(m,4) + (k-n+1)* 2**(n-m+1)*( nc_u(int(np.ceil(np.log2(k-n+2))),4)+nc_u(int(n-m+1),4) ))


def msb_estimate(k,n):
    return 2 * ((2**(k-n)-1)*int(np.ceil(np.log2(k-n+2)))/2 * 4 + 2**(k-n)-2)

print(nc_u(5,4))
for k in range(6,10):
    n = 5
    m = np.arange(0,n+1)
    res = [num_basic_gates(k,n,_)+msb_estimate(k,n) for _ in m]
    res_im = [improved_basic_gates(k,n,_)+msb_estimate(k,n) for _ in m]
    plt.plot(m,res,label='{}'.format(k))
    plt.plot(m,res_im,label='{} improved'.format(k))
plt.legend(loc='best')
plt.show()

plt.scatter(8,outlook_improved_basic_gates(8,5,3),label='improved')
plt.scatter(8,improved_basic_gates(8,5,3),label='normal')
plt.legend(loc='best')
plt.show()
    
