.. _random-distributions:

====================
Random Distributions
====================

A random distribution is an implementation of a circuit factory that provides a way to construct a quantum circuit
to prepare a state which corresponds to a random distribution.
More precisely, the resulting state together with an affine map can be used to sample from the considered distribution.
The qubits are measured and then mapped to the desired range using the affine map.

In the following, we discuss the currently existing implementations.

------------------------
Univariate Distributions
------------------------

.. topic:: Univariate Distribution

    asdf

.. topic:: Bernoulli Distribution

    asdf

.. topic:: Uniform Distribution

    asdf

.. topic:: Normal Distribution

    asdf

.. topic:: Log-Normal Distribution

    asdf

--------------------------
Multivariate Distributions
--------------------------

.. topic:: Multivariate Distribution

    asdf

.. topic:: Multivariate Uniform Distribution

    .. code:: python

        # specify the number of qubits that are used to represent the different dimenions of the uncertainty model
        num_qubits = [2, 3]

        # specify the lower and upper bounds for the different dimension
        low = [-1, -2]
        high = [1, 2]

        # construct random distribution
        u = MultivariateUniformDistribution(num_qubits, low, high)

.. topic:: Multivariate Normal Distribution

    .. code:: python

        # specify the number of qubits that are used to represent the different dimensions of the uncertainty model
        num_qubits = [2, 3]

        # specify the lower and upper bounds for the different dimension
        low = [-1, -2]
        high = [1, 2]
        mu = np.zeros(2)
        sigma = np.eye(2)

        # construct random distribution
        u = MultivariateNormalDistribution(num_qubits, low, high, mu, sigma)