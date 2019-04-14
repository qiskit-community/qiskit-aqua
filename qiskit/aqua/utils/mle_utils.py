import numpy as np
from qiskit.aqua.aqua_error import AquaError


def circ_dist(w0, w1):
    """
    @brief Circumferential distance of two angles on the unit circle,
           divided by 2 pi:
             min{|z - w0 + w1| : z is any integer}
    @param w0 First angle (in [0,1])
    @param w1 Second angle (in [0,1])
    @return (circumferential distance of w0 and w1) / (2 pi)
    @note At most one of the inputs can be an array
    """
    # Since w0 and w1 \in [0,1] it suffices to check not all integers
    # but only -1, 0 and 1
    z = np.array([-1, 0, 1])

    # Check if one of the inputs is an array and if one is an array
    # make sure it is w0, then we treat it as array and w1 not

    # Neither is an array, just do calculation
    if not hasattr(w0, "__len__") and not hasattr(w1, "__len__"):
        return np.min(np.abs(z - w0 + w1))

    # Both are an array, not allowed
    if hasattr(w0, "__len__") and hasattr(w1, "__len__"):
        raise AquaError("Only one of the inputs can be an array!")

    # w1 is an array, swap
    if hasattr(w1, "__len__"):
        w0, w1 = w1, w0

    # Calculate
    d = np.empty_like(w0)
    for idx, w in enumerate(w0):
        d[idx] = np.min(np.abs(z - w + w1))

    return d


def value_to_angle(a):
    """
    @brief Convert the value a to an angle w by applying
                w = arcsin(sqrt(a)) / pi
    @param a Value (in [0, 1])
    @result The corresponding angle w = arcsin(sqrt(a)) / pi
    """
    if hasattr(a, "__len__"):
        a = np.asarray(a)  # ensure we have a numpy array
        if not (a.all() >= 0 and a.all() <= 1):
            raise AquaError("Invalid value! Value `a` must be 0 <= a <= 1")
    else:
        if not (a >= 0 and a <= 1):
            raise AquaError("Invalid value! Value `a` must be 0 <= a <= 1")

    return np.arcsin(np.sqrt(a)) / np.pi


def angle_to_value(w):
    """
    @brief Convert the angle w to a value a by applying
                a = sin^2(pi * w)
    @param w Angle (in [0, 1])
    @result The corresponding value a = sin^2(pi * w)
    """
    if hasattr(w, "__len__"):
        w = np.asarray(w)
        if not (w.all() >= 0 and w.all() <= 1):
            raise AquaError("Invalid value! Angle `w` must be 0 <= a <= 1")
    else:
        if not (w >= 0 and w <= 1):
            raise AquaError("Invalid value! Angle `w` must be 0 <= a <= 1")

    return np.sin(np.pi * w)**2


def pdf_w(w, w_exact, m):
    """
    @brief Probability of measuring angle w if w_exact is the exact angle,
           for a QAE experiment with m qubits.
    @param w Angle w for which we calculate the probability
    @param w_exact The exact angle of the distribution
    @param m The number of qubits
    @return Pr(w | w_exact, m)
    """
    M = 2**m

    # Get the circumferential distances
    d = circ_dist(w_exact, w)

    # We'll use list comprehension, so the input should be a list
    scalar = False
    if not hasattr(d, "__len__"):
        d = [d]
        scalar = True

    # Compute probability, and if distance is 0 return 1
    pr = np.array([np.sin(M * D * np.pi)**2
                   / (M**2 * np.sin(D * np.pi)**2)
                   if D != 0 else 1 for D in d])

    # If is was a scalar return scalar otherwise the array
    return (pr[0] if scalar else pr)


def pdf_a(a, a_exact, m):
    """
    @brief Probability of measuring the value a, if a_exact would be the
           exact value for a QAE experiment with m qubits.
    @note Since we apply a mapping a = sin^2(pi w), multiple w values will
          result in the same a value.
          The qiskit probabilities are given for every possible a_i:
             {(a_i, pr_i)}_i, i = 0..(M/2)
          The PDF is stated in terms of the grid points w_i = i/M:
             {(w_i, pr(w_i))}_i, i = 0...M-1
          Hence to state the PDF in terms of the values a, we need to add those
          probabilities up, which
          result in the same value:
            def pr_for_a(a):
              w = arcsin(sqrt(a)) / pi
              w_results_in_same_a = 1 - w

              # add only if a is not 0 or 1, bc those are the unique mapping
              # points in sin^2
              if a not in [0, 1]:
                return pr(w) + pr(w_results_in_same_a)
              return pr(w)
    """
    # Transform the values to angles
    w = value_to_angle(a)
    w_exact = value_to_angle(a_exact)

    # We'll use list comprehension, so the input should be a list
    scalar = False
    if not hasattr(a, "__len__"):
        a = [a]
        w = [w]
        scalar = True

    # Compute the probabilities: Add up both angles that produce the given
    # value, except for the angles 0 and 0.5, which map to the unique a-values,
    # 0 and 1, respectively
    pr = np.array([(pdf_w(wi, w_exact, m) + pdf_w(1 - wi, w_exact, m))
                   if (ai not in [0, 1]) else pdf_w(wi, w_exact, m)
                   for ai, wi in zip(a, w)
                   ]).flatten()

    # If is was a scalar return scalar otherwise the array
    return (pr[0] if scalar else pr)


def loglik(theta, m, ai, pi=1, shots=1):
    """
    @brief Compute the likelihood of the data ai, if the exact
           value a is theta, for m qubits. If a histogram of the values
           ai (total number values is shots) has already been computed,
           the histogram (ai, pi) can also be given as argument. Then the
           original number of datapoints, shots, should also be provided.
    @param theta The parameter of the PDF, here the exact value for a
    @param m The number of qubits
    @param ai The values ai
    @param pi The empiric probabilities of ai (histogram probabilities)
    @param shots The number of original datapoints ai
    @return The loglikelihood of ai (,pi) given theta is the exact value
    """
    return np.sum(shots * pi * np.log(pdf_a(ai, theta, m)))


def bisect_max(f, a, b, steps=100, minwidth=0, retval=False):
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
        print("-- Warning, bisect_max didn't converge after {} steps".format(steps))

    if retval:
        return m, fm
    return m
