# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
This module contains the definition of a base class for inverse quantum fourier transforms.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def bisect_max(f, a, b, steps=50, minwidth=1e-12, retval=False):
    """
    @brief Find the maximum of f in the interval [a, b] using bisection
    @param f The function
    @param a The lower limit of the interval
    @param b The upper limit of the interval
    @param steps The maximum number of steps in the bisection
    @param minwidth If the current interval is smaller than minwidth stop
                    the search
    @return The maximum of f in [a,b] according to this algorithm
    """
    it = 0
    m = (a + b) / 2
    fm = 0
    while it < steps and b - a > minwidth:
        l, r = (a + m) / 2, (m + b) / 2
        fl, fm, fr = f(l), f(m), f(r)

        # fl is the maximum
        if fl > fm and fl > fr:
            b = m
            m = l
        # fr is the maximum
        elif fr > fm and fr > fl:
            a = m
            m = r
        # fm is the maximum
        else:
            a = l
            b = r

        it += 1

    if it == steps:
        logger.warning("-- Warning, bisect_max didn't converge after {} steps".format(steps))

    if retval:
        return m, fm
    return m


class Dist:
    """
    @brief Circumferential distance and derivative function,
            Dist(x, p) = min_{z in [-1, 0, 1]} (|z + p - x|)
    """

    def __init__(self):
        pass

    @staticmethod
    def v(x, p):
        """
        Return the value of the function Dist
        """
        t = p - x
        # Since x and p \in [0,1] it suffices to check not all integers
        # but only -1, 0 and 1
        z = np.array([-1, 0, 1])

        if hasattr(t, "__len__"):
            d = np.empty_like(t)
            for idx, ti in enumerate(t):
                d[idx] = np.min(np.abs(z + ti))
            return d

        return np.min(np.abs(z + t))

    @staticmethod
    def d(x, p):
        """
        Return the derivative of the function Dist
        """
        t = p - x
        if t < -0.5 or (0 < t and t < 0.5):
            return -1
        if t > 0.5 or (-0.5 < t and t < 0):
            return 1
        return 0


class Omega:
    """
    @brief Mapping from QAE value to QAE angle and derivative,
            Omega(a) = arcsin(sqrt(a)) / pi
    """

    def __init__(self):
        pass

    @staticmethod
    def v(a):
        """
        Return the value of Omega(a)
        """
        return np.arcsin(np.sqrt(a)) / np.pi

    @staticmethod
    def d(a):
        """
        Return the value of Derivative(a)
        """
        return 1 / (2 * np.pi * np.sqrt((1 - a) * a))


class Alpha:
    """
    @brief Implementation of pi * d(w(x), w(p)) and derivative w.r.t. p
    """

    def __init__(self):
        pass

    @staticmethod
    def v(x, p):
        return np.pi * Dist.v(Omega.v(x), Omega.v(p))

    @staticmethod
    def d(x, p):
        return np.pi * Dist.d(Omega.v(x), Omega.v(p)) * Omega.d(p)


class Beta:
    """
    @brief Implementation of pi * d(1 - w(x), w(p)) and derivative w.r.t. p
    """

    def __init__(self):
        pass

    @staticmethod
    def v(x, p):
        return np.pi * Dist.v(1 - Omega.v(x), Omega.v(p))

    @staticmethod
    def d(x, p):
        return np.pi * Dist.d(1 - Omega.v(x), Omega.v(p)) * Omega.d(p)


class PdfA:
    """
    @brief Implementation of QAE PDF f(x, p) and derivative
    """

    def __init__(self):
        pass

    @staticmethod
    def numerator(x, p, m):
        M = 2**m
        return np.sin(M * Alpha.v(x, p))**2 * np.sin(Beta.v(x, p))**2 + np.sin(M * Beta.v(x, p))**2 * np.sin(Alpha.v(x, p))**2

    @staticmethod
    def single_angle(x, p, m, PiDelta):
        M = 2**m

        d = PiDelta.v(x, p)
        res = np.sin(M * d)**2 / (M * np.sin(d))**2 if d != 0 else 1

        return res

    @staticmethod
    def v(x, p, m):
        """
        Return the value of f, i.e. the probability of getting the
        estimate  x (in [0, 1]) if p (in [0, 1]) is the true value,
        given that we use m qubits
        """
        # We'll use list comprehension, so the input should be a list
        scalar = False
        if not hasattr(x, "__len__"):
            scalar = True
            x = np.asarray([x])

        # Compute the probabilities: Add up both angles that produce the given
        # value, except for the angles 0 and 0.5, which map to the unique a-values,
        # 0 and 1, respectively
        pr = np.array([PdfA.single_angle(xi, p, m, Alpha) + PdfA.single_angle(xi, p, m, Beta)
                       if (xi not in [0, 1]) else PdfA.single_angle(xi, p, m, Alpha)
                       for xi in x
                       ]).flatten()

        # If is was a scalar return scalar otherwise the array
        return (pr[0] if scalar else pr)

    @staticmethod
    def logd(x, p, m):
        """
        Return the log of the derivative of f
        """
        M = 2**m

        if x not in [0, 1]:
            def num_p1(A, B):
                return 2 * M * np.sin(M * A.v(x, p)) * np.cos(M * A.v(x, p)) * A.d(x, p) * np.sin(B.v(x, p))**2 \
                    + 2 * np.sin(M * A.v(x, p))**2 * np.sin(B.v(x, p)) * np.cos(B.v(x, p)) * B.d(x, p)

            def num_p2(A, B):
                return 2 * np.cos(A.v(x, p)) * A.d(x, p) * np.sin(B.v(x, p))

            def den_p2(A, B):
                return np.sin(A.v(x, p)) * np.sin(B.v(x, p))

            return (num_p1(Alpha, Beta) + num_p1(Beta, Alpha)) / PdfA.numerator(x, p, m) \
                - (num_p2(Alpha, Beta) + num_p2(Beta, Alpha)) / den_p2(Alpha, Beta)

        return 2 * Alpha.d(x, p) * (M / np.tan(M * Alpha.v(x, p)) - 1 / np.tan(Alpha.v(x, p)))
