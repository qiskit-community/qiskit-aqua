Installation and Setup
======================

Dependencies
------------

Since Qiskit Aqua is built upon `Qiskit <https://qiskit.org>`__, you are encouraged to look over the
`Qiskit
installation <https://github.com/Qiskit/qiskit-sdk-py/blob/master/README.md#installation>`__
too.

Like for Qiskit, at least `Python 3.5 or
later <https://www.python.org/downloads/>`__ is needed to use Qiskit
Aqua. In addition, `Jupyter
Notebook <https://jupyter.readthedocs.io/en/latest/install.html>`__ is
recommended for interacting with the tutorials. For this reason we
recommend installing the `Anaconda
3 <https://www.continuum.io/downloads>`__ Python distribution, as it
comes with all of these dependencies pre-installed.

.. _installation-1:

Installation
------------

Qiskit Aqua can be used as a tool to execute `quantum algorithms <./algorithms.html>`__.
With the appropriate input, a quantum algorithm will run on top of the underlying `Qiskit <https://qiskit.org>`__
platform, which will generate, compile and execute a circuit modeling the input problem.
Qiskit Aqua can also be used as the foundation for domain-specific applications, such as
`Qiskit Aqua Chemistry <https://qiskit.org/aqua/chemistry>`__,
`Qiskit Aqua Artificial Intelligence <https://qiskit.org/aqua/ai>`__ and
`Qiskit Aqua Optimization <https://qiskit.org/aqua/optimization>`__.
The best way to install Qiskit Aqua when the goal is to use as a tool or as a library
of quantum algorithms is via the `pip <https://pip.pypa.io/en/stable/>`__  package management system:

.. code:: sh

    pip install qiskit_aqua

pip will handle all dependencies automatically and you will always
install the latest (and well-tested) release version.

A different class of users --- namely, quantum researchers and developers --- might be more interested
in exploring the source code of Qiskit Aqua and `extending it <./extending.html>`__ by providing
new components, such as `quantum algorithms <./algorithms.html>`__, `optimizers <./optimizers.html>`__,
`variational forms <./variational_forms.html>`__, `initial states <./initial_states.html>`__,
`inverse Quantum Fourier Transforms (IQFTs) <./iqfts.html>`__ and `Grover search oracles <./oracles.html>`__.
The best way to install Qiskit Aqua when the goal is extending its capabilities is by cloning
the `Qiskit Aqua repository <https://github.com/Qiskit/aqua>`__.

.. note::
    We recommend using Python virtual environments to improve your experience.


