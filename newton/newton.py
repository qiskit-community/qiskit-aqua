from fixed_point import *
import numpy as np
import matplotlib.pyplot as plt

def float_newton_iteration(x0, lam, steps=1):
    for i in range(steps):
        x0 = 2*x0-x0**2*lam
    return x0

def get_x0(lam):
    return 2**-np.ceil(np.log2(lam))

def fp_newton(lam, prec=8, steps=10):
    x = FPDecimal(n=prec, val=get_x0(lam))
    l = FPInteger(n=prec, val=lam)
    t = FPInteger(n=prec, val=2)
    for i in range(steps):
        x = x*t-x*x*l
    return x.to_float()


prec = 32
lams = np.arange(2**8-1)+1
res = []
for l in lams:
    res.append(fp_newton(l, prec=prec))
plt.plot(lams, res)
plt.plot(lams, 1/lams)
plt.show()
