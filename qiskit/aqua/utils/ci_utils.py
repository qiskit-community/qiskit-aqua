import numpy as np
from scipy.stats import norm, chi2


def normal_quantile(alpha):
    """
        @brief Quantile function, returns the value z at for which the
               cumulative distribution function reaches 1 - alpha/2.
        @param alpha Quantile
        @return See brief
        @note Check: q(0.1) = 1.64
                     q(0.01) = 2.58
              And then
                int_{-q(a)}^{q(a)} exp(-x^2 / 2) / sqrt(2 pi) dx = 1 - alpha
    """
    # equivalent:
    # return np.sqrt(2) * erfinv(1 - alpha)
    return norm.ppf(1 - alpha / 2)


def chi2_quantile(alpha, df=1):
    """
    @brief Quantile function for chi-squared distribution
    @param alpha Compute the (1 - alpha)-quantile
    @param df Degrees of freedom (dofs)
    @return (1 - alpha)-quantile for df dofs
    """
    return chi2.ppf(1 - alpha, df)


class Dist:
    """
    @brief Circumferential distance and derivative
    """

    def __init__(self):
        pass

    @staticmethod
    def v(x, p):
        t = p - x
        z = np.arange(-1, 2)
        return np.min(np.abs(z + t))

    @staticmethod
    def d(x, p):
        t = p - x
        if t < -0.5 or (0 < t and t < 0.5):
            return -1
        if t > 0.5 or (-0.5 < t and t < 0):
            return 1
        return 0


class Omega:
    """
    @brief Mapping from QAE value to QAE angle and derivative
    """

    def __init__(self):
        pass

    @staticmethod
    def v(a):
        return np.arcsin(np.sqrt(a)) / np.pi

    @staticmethod
    def d(a):
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


class f:
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
    def denominator(x, p, m):
        M = 2**m
        return (M * np.sin(Alpha.v(x, p)) * np.sin(Beta.v(x, p)))**2

    @staticmethod
    def v(x, p, m):
        return f.numerator(x, p, m) / f.denominator(x, p, m)

    @staticmethod
    def logv(x, p, m):
        return np.log(f.numerator(x, p, m) / f.denominator(x, p, m))

    @staticmethod
    def logd(x, p, m):
        M = 2**m

        if x not in [0, 1]:
            def num_p1(A, B):
                return 2 * M * np.sin(M * A.v(x, p)) * np.cos(M * A.v(x, p)) * A.d(x, p) * np.sin(B.v(x, p))**2 \
                    + 2 * np.sin(M * A.v(x, p))**2 * np.sin(B.v(x, p)) * np.cos(B.v(x, p)) * B.d(x, p)

            def num_p2(A, B):
                return 2 * np.cos(A.v(x, p)) * A.d(x, p) * np.sin(B.v(x, p))

            def den_p2(A, B):
                return np.sin(A.v(x, p)) * np.sin(B.v(x, p))

            return (num_p1(Alpha, Beta) + num_p1(Beta, Alpha)) / f.numerator(x, p, m) \
                - (num_p2(Alpha, Beta) + num_p2(Beta, Alpha)) / den_p2(Alpha, Beta)

        return 2 * Alpha.d(x, p) * (M / np.tan(M * Alpha.v(x, p)) - 1 / np.tan(Alpha.v(x, p)))


# More precise name and accepting arrays
def d_logprob(x, p, m):
    if hasattr(x, "__len__"):
        return np.array([f.logd(xv, p, m) for xv in x])
    return f.logd(x, p, m)


def fisher_information(p, m):
    def integrand(x):
        return (f.logd(x, p, m))**2 * f.v(x, p, m)

    M = 2**m
    grid = np.sin(np.pi * np.arange(M) / M)**2
    FI = np.sum([integrand(x) for x in grid])

    return FI
