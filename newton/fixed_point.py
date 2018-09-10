import numpy as np

class FPNumber():

    def __init__(self, n=None, bits=None, bitstr=None):
        self._size = n
        self._bits = np.zeros(n, dtype=bool)
        if isinstance(bits, np.ndarray):
            self._bits = bits
            self._size = len(self._bits)
        if isinstance(bitstr, str):
            self._bits = self.from_str(bitstr)
            self._size = len(self._bits)

    def from_str(self, s):
        return np.array([0 if c=="0" else 1 for c in s], dtype=bool)

    def copy(self):
        return FPNumber(bitstr=str(self))

    def reset(self):
        self._bits = np.zeros(self._size, dtype=bool)
        return self

    def __str__(self):
        return "".join("1" if c else "0" for c in self._bits)

    def __getitem__(self, i):
        if not isinstance(i, tuple):
            i = (i,)
        ret = []
        for d in i:
            ret.append(self._bits[d])
        if len(ret) == 1:
            ret = ret[0]
        return ret

    def __setitem__(self, i, v):
        if not isinstance(i, tuple):
            i = (i,)
            v = np.array([v], dtype=bool)
        self._bits[i] = v

    def __iter__(self):
        for i in range(self._size):
            yield self._bits[i]

    def __len__(self):
        return self._size

    def __add__(self, fpnumber):
        if not isinstance(fpnumber, FPNumber):
            raise ValueError("Can only add two fixed point integers")
        if fpnumber._size > self._size:
            return fpnumber.__add__(self)
        b = fpnumber._fill(self._size)
        res = self.copy().reset()
        carry = 0
        for i in reversed(range(self._size)):
            tmp = np.logical_xor(self[i], b[i])
            res[i] = np.logical_xor(tmp, carry)
            carry = np.logical_or(np.logical_and(self[i], b[i]),
                    np.logical_and(tmp, carry))
        return res

    def __sub__(self, fpnumber):
        if not isinstance(fpnumber, FPNumber):
            raise ValueError("can only subtract two fixed point integers")
        if fpnumber._size > self._size:
            return fpnumber.__add__(self)
        b = fpnumber._fill(self._size)
        res = self.copy().reset()
        carry = 0
        for i in reversed(range(self._size)):
            tmp = np.logical_xor(self[i], b[i])
            res[i] = np.logical_xor(tmp, carry)
            carry = np.logical_or(np.logical_and(np.logical_not(self[i]), b[i]),
                    np.logical_and(np.logical_not(tmp), carry))
        return res

    def _shift(self, c):
        ret = self.copy()
        if c > 0:
            ret[:-c] = self[c:]
            ret[-c:] = np.zeros(abs(c))
        elif c < 0:
            ret[-c:] = self[:c]
            ret[:-c] = np.zeros(abs(c))
        return ret
    
    def _fill(self, c):
        if c < self._size:
            raise ValueError("New size must be larger than existing size to fill.")
        if c == self._size:
            return self.copy()
        n = self.copy()
        n._size = c
        b = np.zeros(c)
        b[c-self._size:] = n._bits
        n._bits = b
        return n


class FPInteger(FPNumber):

    def __init__(self, n=None, bits=None, bitstr=None, val=None):
        super().__init__(n, bits, bitstr)
        if val != None:
            s = np.binary_repr(val, width=self._size)
            self._bits = self.from_str(s)

    def copy(self):
        return FPInteger(bitstr=super().__str__())

    def to_int(self):
        return sum([2**(self._size-i-1) for i, b in enumerate(self._bits) if b])

    def __str__(self):
        return str(self.to_int()) + " " + super().__str__()
    
    def __mul__(self, fpnumber):
        if not isinstance(fpnumber, FPNumber):
            raise ValueError("Can only multiply fixed point numbers")
        diff = fpnumber._size - self._size
        ns = max(fpnumber._size, self._size)
        if diff < 0:
            a = self
            b = fpnumber._fill(self._size)
        elif diff > 0:
            a = self._fill(fpnumber._size)
            b = fpnumber
        else:
            a = self
            b = fpnumber
        ret = FPInteger(ns)
        if isinstance(fpnumber, FPInteger):
            for i, bit in enumerate(reversed(a)):
                if bit:
                    ret += b._shift(i)
        elif isinstance(fpnumber, FPDecimal):
            for i, bit in enumerate(b):
                if bit:
                    ret += a._shift(-(i+1))
        return ret


class FPDecimal(FPNumber):

    def __init__(self, n=None, bits=None, bitstr=None, val=None):
        super().__init__(n, bits, bitstr)
        if val != None and val < 1:
            s = np.binary_repr(int(val*2**n), width=self._size)
            self._bits = self.from_str(s)

    def copy(self):
        return FPDecimal(bitstr=super().__str__())

    def to_float(self):
        return sum([2**(self._size-i-1) for i, b in enumerate(self._bits) if
            b])/2**self._size

    def __str__(self):
        return str(self.to_float()) + " 0." + super().__str__()

    def _fill(self, c):
        if c < self._size:
            raise ValueError("New size must be larger than existing size to fill.")
        if c == self._size:
            return self.copy()
        n = self.copy()
        n._size = c
        b = np.zeros(c)
        b[:-c+self._size] = n._bits
        n._bits = b
        return n

    def __mul__(self, fpnumber):
        if not isinstance(fpnumber, FPNumber):
            raise ValueError("Can only multiply fixed point numbers")
        diff = fpnumber._size - self._size
        ns = max(fpnumber._size, self._size)
        if diff < 0:
            a = self
            b = fpnumber._fill(self._size)
        elif diff > 0:
            a = self._fill(fpnumber._size)
            b = fpnumber
        else:
            a = self
            b = fpnumber
        ret = FPDecimal(ns)
        if isinstance(fpnumber, FPDecimal):
            for i, bit in enumerate(a):
                if bit:
                    ret += b._shift(-(i+1))
        elif isinstance(fpnumber, FPInteger):
            for i, bit in enumerate(reversed(b)):
                if bit:
                    ret += a._shift(i)
        return ret


if __name__ == "__main__":
    #x = FPInteger(8, val=98)
    #y = FPInteger(8, val=73)
    x = FPDecimal(16, val=0.5)
    y = FPDecimal(8, val=0.125)
    print(x, y)
    print(x-y)



