Extending QISKit ACQUA
======================

QISKit ACQUA has a modular and extensible architecture.
Algorithms and their supporting objects are have been designed to be
pluggable modules.  This was done in order to encourage
researchers and developers interested in contributiong to, and experimenting with,
new quantum algorithms to extend the QISKit ACQUA framework with their novel
research contributions.

Algorithms
----------
A new `algorithm <./algorithms.html>`__ may be developed according to the specific API provided by QISKit ACQUA.
By simply adding its code to the collection of existing algorithms, that new algorithm
will be immediately recognized via dynamic lookup, and made available for use within the framework of QISKit ACQUA.
To develop and deploy any new algorithm, the new algorithm class should derive from the ``QuantumAlgorithm`` class.
Along with all of its supporting modules, the new algorithm class should be installed under its own folder in the
``qiskit_acqua`` directory, just like the existing algorithms.

Optimizers
----------
New `optimizers <./optimizers.html>`__ for quantum variational algorithms
should be installed in the ``qiskit_acqua/utils/optimizers`` folder  and derive from
the ``Optimizer`` class.

Variational Forms
-----------------
`Trial wavefunctions <./variational_forms.html>`__ for quantum variational algorithms, such as
`VQE <./algorithms.html#variational-quantum-eigensolver-vqe>`__
should go under the ``qiskit_acqua/utils/variational_forms`` folder
and derive from the ``VariationalForm`` class.

Initial States
--------------
`Initial states <./initial_states.html>`__, for algorithms such as `VQE <./algorithms.html#variational-quantum-eigensolver-vqe>`__,
`QPE <./algorithms.html#quantum-phase-estimation-qpe>`__
and `IQPE <./algorithms.html#iterative-quantum-phase-estimation-iqpe>`__, should go under the ``qiskit_acqua/utils/initial_states`` folder and
derive from the ``InitialState`` class.

Inverse Quantum Fourier Transforms (IQFTs)
------------------------------------------
`IQFTs <./iqfts.html>`__, for use for example for `QPE <./algorithms.html#quantum-phase-estimation-qpe>`__, should be installed  under the
``qiskit_acqua/utils/iqfts`` folder and derive from the ``IQFT`` class.

Oracles
-------
`Oracles <./oracles.html>`__, for use with algorithms such as `Grover's search <./algorithms.html#quantum-grover-search>`__,
should go under the
``qiskit_acqua/utils/oracles`` folder  and derive from the ``Oracle`` class.

.. note::
    All the classes implementing the algorithms and their supporting components listed above
    should have a configuration dictionary including ``name``, ``description`` and ``input_schema`` properties.

