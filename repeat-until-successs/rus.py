import numpy as np
import matplotlib.pyplot as plt
import numpy.polynomial.chebyshev as cheb

def gb(*args):
    return np.arctan(np.tan(np.arcsin(np.prod(np.array([np.abs(np.sin(a)) for
        a in args]))))**2)

def par(*args):
    return np.arctan(np.prod(np.array([np.tan(a) for a in args])))

def func(x):
    return 1/(1-x)

def construct_order(i, c, x, s=1):
    # return c*x**i
    r = 1
    if c >= 1:
        r = 2**np.ceil(np.log2(c))
        c /= r
    r *= s**i
    if i == 0:
        return r*c
    elif i == 1:
        return r*par(x/s, c)
    elif i%2 == 0:
        return r*gb(np.arcsin(np.sqrt(c)), *[x/s for _ in range(int(i/2))])
    else:
        return r*par(x/s, construct_order(i-1, c, x, s=s))

def construct_polynomial(cheb, x):
    res = 0
    for i, c in enumerate(cheb):
        sign = c/abs(c)
        if c < 0:
            c = -c
        res += sign*construct_order(i, c, x, s=0.42)
    return res

def eval_polynomial(cheb, x):
    ret = []
    for v in x:
        ret.append(construct_polynomial(cheb, v))
    return np.array(ret)

x = np.linspace(0, 1/2, 100)
y = func(x)

coefs = cheb.cheb2poly(cheb.chebfit(x, y, 8))

plt.plot(x, y)
plt.plot(x, eval_polynomial(coefs, x))

plt.show()
