Installation and Setup
=====================

Dependencies
------------

Since QISKit ACQUA is built upon `QISKit <https://qiskit.org>`__, you are encouraged to look over the
`QISKit
installation <https://github.com/QISKit/qiskit-sdk-py/blob/master/README.md#installation>`__
too.

Like for QISKit, at least `Python 3.5 or
later <https://www.python.org/downloads/>`__ is needed to use QISKit
ACQUA. In addition, `Jupyter
Notebook <https://jupyter.readthedocs.io/en/latest/install.html>`__ is
recommended for interacting with the tutorials. For this reason we
recommend installing the `Anaconda
3 <https://www.continuum.io/downloads>`__ Python distribution, as it
comes with all of these dependencies pre-installed.

.. _installation-1:

Installation
------------

QISKit ACQUA can be used as a tool to execute `quantum algorithms <./algorithms.html>`__.
With the appropriate input, a quantum algorithm will run on top of the underlying `QISKit <https://qiskit.org>`__
platform, which will generate, compile and execute a circuit modeling the input problem.
QISKit ACQUA can also be used as the foundation for domain-specific applications, such as
`QISKit ACQUA Chemistry <https://qiskit.org/acqua/chemistry>`__,
`QISKit ACQUA Artificial Intelligence <https://qiskit.org/acqua/ai>`__ and
`QISKit ACQUA Optimization <https://qiskit.org/acqua/optimization>`__. 
The best way to install QISKit ACQUA when the goal is to use as a tool or as a library
of quantum algorithms is via the `pip <https://pip.pypa.io/en/stable/>`__  package management system:

.. code:: sh

    pip install qiskit_acqua

pip will handle all dependencies automatically and you will always
install the latest (and well-tested) release version.

A different class of users --- namely, quantum researchers and developers --- might be more interested
in exploring the source code of QISKit ACQUA and `extending it <./extending.html>`__ by providing
new components, such as `quantum algorithms <./algorithms.html>`__, `optimizers <./optimizers.html>`__,
`variational forms <./variational_forms.html>`__, `initial states <./initial_states.html>`__,
`inverse Quantum Fourier Transforms (IQFTs) <./iqfts.html>`__ and `Grover search oracles <./oracles.html>`__.
The best way to install QISKit ACQUA when the goal is extending its capabilities is by cloning
the `QISKit ACQUA repository <https://github.com/Qiskit/qiskit-acqua>`__.

.. note::
    We recommend using Python virtual environments to improve your experience.


