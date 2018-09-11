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