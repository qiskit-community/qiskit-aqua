Configuring and Running an Experiment
=====================================

QISKit ACQUA supports two types of users:

1. *Practitioners*, who are merely interested in executing QISKit ACQUA
   as a tool from which they can choose a
   `quantum algorithm <https://qiskit.org/documentation/acqua/algorithms.html#quantum-algorithms>`__.
   These users may not be interested in extending QISKit ACQUA
   with additional capabilities.  In fact, they may not even be interested
   in learning the details of quantum computing, such as the notions of
   circuits, gates and qubits.  What these users expect
   from quantum computing is the gains in performance and accuracy, and
   the reduction in computational complexity, compared to the use of
   a `classical
   algorithm <https://qiskit.org/documentation/acqua/algorithms.html#classical-algorithms>`__
   for generating reference values.
2. *Quantum algorithm researchers and developers*, who are interested in extending
   QISKit ACQUA with new quantum algorithms or algorithm components for more efficient
   and accurate computations.

In this section, we cover the first class of users --- the algorithm practitioners.
Specifically, this section describes how QISKit ACQUA can be accessed as a
tool for experimenting with the execution of quantum algorithms.

To see how you can extend QISKit ACQUA Chemistry with new components,
please refer to `Section "Contributing to QISKit ACQUA <./extending.html>`__.

Execution Modes
---------------

QISKit ACQUA has both `Graphical User Interface (GUI) <#gui>`__ and `command
line <#command-line>`__ tools, which may be used when experimenting with quantum algorithms.
Both can load and run an `input
file <#input-file>`__ specifying the the type of problem the experiment is about,
and the quantum
algorithm to be used for the computation, along with the algorithm configuration
and various other options to
customize the experiment.  If you are new to
QISKit ACQUA, we highly recommend getting started with the GUI.
Finally, QISKIT ACQUA can also be accessed
`programmatically <#programmable-interface>`__ by users interested
in customizing the experiments beyond what the command line and GUI can offer.

GUI
~~~

The GUI provides an easy means to create from scratch, or load
an existing, `input file <#input-file>`__, and then run that input file to experiment with a
quantum algorithm.
An input file for QISKit ACQUA is assumed to be in JSON format.  Such a file is created,
edited and saved with schema-based validation of parameter values.

When `installing <./install.html>`__
QISKit ACQUA via the ``pip install`` command,
a script is created that allows you to start the GUI from the command line,
as follows:

.. code:: sh

   qiskit_acqua_ui

If you cloned QISKit ACQUA directly from the
`GitHub repository <https://github.com/QISKit/qiskit-acqua>`__ instead of using ``pip
install``, then the script above will not be present and the launching command should be instead:

.. code:: sh

   python qiskit_acqua/ui/run

This command must be launched from the root folder of the ``qiskit-acqua`` repository clone.

Command Line
~~~~~~~~~~~~

QISKit ACQUA, if `installed <./install.html>`__ via ``pip install``,
comes with the following command-line tool:

.. code:: sh

   qiskit_acqua_cmd

If you cloned QISKit ACQUA from its remote
`GitHub repository <https://github.com/QISKit/qiskit-acqua>`__
instead of using ``pip install``, then the command-line interface can be executed as follows:

.. code:: sh

   python qiskit_acqua

from the root folder of the ``qiskit-acqua`` repository clone.

When invoking QISKit ACQUA from the command line, a JSON
`input file <#input-file>`__ is expected as a command-line
parameter.


Programmable Interface
~~~~~~~~~~~~~~~~~~~~~~

Experiments can be run programmatically too. Numerous
examples on how to program an experiment in QISKit ACQUA
can be found in the ``acqua`` folder of the
`QISKit ACQUA Tutorials GitHub repository
<https://github.com/QISKit/qiskit-acqua-tutorials>`__.

It should be noted at this point that QISKit ACQUA is
designed to be as much declarative as possible.  This is done in order
to simplify the programmatic access to QISKit ACQUA and
minimize the chances for configuration errors, and helping users
who might be experts in chemistry but not interested in writing a lot of code or
learning new Application Programming Interfaces (APIs).

There is
nothing preventing a user from accessing the QISKit ACQUA APIs and
programming an experiment step by step, but a  more direct way to access QISKit ACQUA programmatically
is by obtaining a JSON algorithm input file, such as one of those
available in the ``acqua/input_files`` subfolder of the
`QISKit ACQUA Tutorials <https://github.com/QISKit/qiskit-acqua-tutorials>`__
repository.  Such files can be constructed either manually or automatically
via QISKit ACQUA domain-specific applications.  For example,
the QISKit ACQUA Chemistry `command-line tool
<https://qiskit.org/documentation/acqua/chemistry/config_run.html#command-line>`__
and `GUI <https://qiskit.org/documentation/acqua/chemistry/config_run.html#gui>`__ 
have options to serialize the input to the quantum algorithm for future reuse.
The JSON file can then be pasted into a Python program and modified according to the
needs of the developer, before invoking the ``run_algorithm`` API in ``qiskit_acqua``.
This technique can be used, for example, to compare the results of two different algorithms.

Input File
----------

An input file is used to define an QISKit ACQUA problem,
and includes the input to the
`quantum algorithm <https://qiskit.org/documentation/acqua/algorithms.html>`__
as well as configuration information for
the underlying quantum system.
Specific configuration parameter values can be supplied to
explicitly control the processing and the quantum algorithm used for
the computation, instead of using defaulted values when none are
supplied.

The format for the input file is `JavaScript Object Notation (JSON) <https://www.json.org/>`__.
This allows for schema-based
configuration-input correctness validation.  While it is certainly possible to
generate a JSON input file manually, QISKit ACQUA allows for a simple way
to achieve the automatic generation of such a JSON input file from the execution
of a domain-specific application.

For example, the `QISKit ACQUA Chemistry `command-line tool
<https://qiskit.org/documentation/acqua/chemistry/config_run.html#command-line>`__
and `GUI <https://qiskit.org/documentation/acqua/chemistry/config_run.html#gui>`__ 
both allow for automatically serializing the input to the quantum algorithm
as a JSON file.  Serializing the input to the quantum algorithm at this point is useful in many scenarios
because the contents of one of such JSON files are domain- and problem-independent:

- Users can share JSON files among each other in order to compare and contrast
  their experimental results at the algorithm level, for example to compare
  results obtained with the same input and different algorithms, or
  different implementations of the same algorithm, regardless of the domain
  in which those inputs were generated (chemistry, artificial intelligence, optimization, etc.)
  or the problem that the user was trying to solve.
- People performing research on quantum algorithms may be interested in having
  access to a number of such JSON files in order to test and refine their algorithm
  implementations, irrespective of the domain in which those JSON files were generated
  or the problem that the user was trying to solve.
- Repeating a domain-specific experiment in which the values of the input parameters remain the same,
  and the only difference is in the configuration of the quantum algorithm and its
  supporting components becomes much more efficient because the user can choose to
  restart any new experiment directly at the algorithm level, thereby bypassing the
  input extraction from the driver, and the input translation into a qubit operator.

A number of sample JSON input files for QISKit ACQUA are available in the
``acqua/input_files``
subfolder of the `QISKit ACQUA Tutorials <https://github.com/QISKit/qiskit-acqua-tutorials>`__
repository.

An input file comprises the following main sections, although not all
are mandatory:

``problem``
~~~~~~~~~~~

In QISKit ACQUA,
a *problem* specifies the type of experiment being run.  Configuring the problem is essential
because it determines which algorithms are suitable for the specific experiment.
QISKit ACQUA comes with a set of predefined problems.
This set is extensible: new problems can be added,
just like new algorithms can be plugged in to solve existing problems in a different way,
or to solve new problems.

Currently, a problem can be configured by assigning a ``string`` value to the ``name`` parameter:

.. code:: python

    name = "energy" | "excited_states" | "ising" | "dynamics" | "search" | "svm_classification"

As shown above, ``"energy"``, ``"excited_states"``, ``"ising"``, ``"dynamics"``,
``"search"``, and ``"svm_classification"`` are currently
the only values accepted for ``name``, corresponding to the computation of
*energy*, *excited states*, *Ising models*, *dynamics of evolution*, *search* and
*Support Vector Machine (SVM) classification*, respectively.
New problems, disambiguated by their
``name`` parameter, can be programmatically
added to QISKit ACQUA via the
``algorithminput.py`` Application Programming Interface (API), and each quantum or classical
`algorithm <./algorithms.html>`__
should programmatically list the problems it is suitable for in its JSON schema, embedded into
the class implementing the ``QuantumAlgorithm`` interface.

Aspects of the computation may include use of random numbers. For instance, 
`VQE <./algorithms.html#variational-quantum-eigensolver-vqe>`__
is coded to use a random initial point if the variational form does not supply any
preference based on the initial state and if the
user does not explicitly supply an initial point. 
In this case, each run of VQE, for what would otherwise be a constant problem,
can produce a different result, causing non-determinism and the inability to replicate
the same result across different runs with
identical configurations. Even though the final value might be numerically indistinguishable,
the number of evaluations that led to the computation of that value may differ across runs.
To enable repeatable experiments, with the exact same outcome, a *random seed* can be set,
thereby forcing the same pseudo-random numbers to
be generated every time the experiment is run:

.. code:: python

    random_seed : int

The default value for this parameter is ``None``.

``input``
~~~~~~~~~

This section allows the user to specify a the input to the QISKit ACQUA algorithm.
Such input is expected to be a qubit operator, expressed as the value of the
``qubit_op`` parameter, for problems of type energy, excited states, Ising models and
dynamics of evolution.  For problems of type SVM classification, the input consists
of a *training dataset* (a map linking each label to a list of data points),
a *test dataset* (also a map linking each label to a list of data points), and
the list of data points on which to apply classification.
These are specified as the values of the parameters
``training_datasets``, ``test_datasets``, and ``datapoints``, respectively.
The ``input`` section is disabled for problems of type search; for such problems,
the input specification depends on the particular
`oracle <./oracles.html> chosen for the
`Grover <./algorithms.html#quantum-grover-search> algorithm.
Currently the satisfiability (SAT) oracle
implementation is provided, which takes as input a SAT problem in
`DIMACS CNF format <http://www.satcompetition.org/2009/format-benchmarks2009.html>`__
expressed as the value of the ``cnf`` parameter,
and constructs the corresponding quantum circuit.

``algorithm``
~~~~~~~~~~~~~

This is an optional section that allows the user to specify which
`quantum algorithm <./algorithms.html#quantum-algorithms>`__
will be used for the experiment.
To compute reference values, QISKit ACQUA also allows the use of a
`classical algorithm <./algorithms.html#classical-algorithms>`__.
In the ``algorithm`` section, algorithms are disambiguated using the
`declarative names <./algorithms.html>`__
by which QISKit ACQUA recognizes them, based on the JSON schema
each algorithm must provide according to the QISKit ACQUA ``QuantumAlgorithm`` API.
The declarative name is specified as the ``name`` parameter in the ``algorithm`` section.
The default value for the ``name`` parameter is ``VQE``, corresponding
to the `Variational Quantum Eigensolver (VQE)
<./algorithms.html#variational-quantum-eigensolver-vqe>`__
algorithm.

An algorithm typically comes with a set of configuration parameters.
For each of them, a default value is provided according to the
``QuantumAlgorithm`` API of QISKit ACQUA.

Furthermore, according to each algorithm, additional sections
may become relevant to optionally
configure that algorithm's components.  For example, variational algorithms,
such as VQE, allow the user to choose and configure an
`optimizer <./optimizers.html>`__ and a
`variational form <./variational_forms.html>`__,
whereas `Quantum Phase Estimation (QPE) <./algorithms.html#quantum-phase-estimation-qpe>`__
allows the user to configure which
`Inverse Quantum Fourier Transform (IQFT) <./iqfts.html>`__ to use.

The `QISKit ACQUA documentation <./index.html>`__
explains how to configure each algorithm and any of the pluggable entities it may use,
such as `optimizers <./optimizers.html>`__, `variational forms <./variational_forms.html>`__,
`initial states <./initial_states.html>`__, `oracles <./oracles.html>`__, and
`Inverse Quantum Fourier Transforms (IQFTs) <./iqfts.html>`__.

Here is an example in which the algorithm VQE is selected along with the
`L-BFGS-B <./optimizers.html#limited-memory-broyden-fletcher-goldfarb-shanno-bound-l-bfgs-b>`__
optimizer and the `RYRZ <./variational_forms.html#ryrz>`__ variational form:

.. code:: json

    "algorithm": {
        "initial_point": null,
        "name": "VQE",
        "operator_mode": "matrix"
    },

    "optimizer": {
        "factr": 10,
        "iprint": -1,
        "maxfun": 1000,
        "name": "L_BFGS_B"
    },

    "variational_form": {
        "depth": 3,
        "entanglement": "full",
        "entangler_map": null,
        "name": "RYRZ"
    }


``backend``
~~~~~~~~~~~

QISKit ACQUA allows for configuring the *backend*, which is the quantum machine
on which a quantum experiment will be run.
This configuration requires specifying 
the `QISKit <https://www.qiskit.org/>`__ quantum computational
backend to be used for computation, which is done by assigning a ``string`` value to
the ``name`` parameter of the ``backend`` section:

.. code:: python

    name : string

The value of the ``name`` parameter indicates either a real-hardware
quantum computer or a quantum simulator.
The underlying QISKit core used by QISKit ACQUA comes
with two predefined quantum device simulators: the *local state vector simulator* and
the *local QASM simulator*, corresponding to the following two
values for the ``name`` parameter: ``"local_statevector_simulator"`` (which
is the default value for the ``name`` parameter) and ``"local_qasm_simulator"``, respectively.
However, any suitable quantum backend can be selected, including
a real quantum hardware device. The ``QConfig.py`` file
needs to be setup for QISKit to access remote devices.  For this, it is sufficient to follow the
`QISKit installation instructions <https://qiskit.org/documentation/install.html#installation>`__.
The QISKit ACQUA `GUI <#GUI>` greatly simplifies the
configuration of ``QConfig.py`` via a user friendly interface,
accessible through the **Preferences...** menu item.

.. topic:: Backend Configuration: Quantum vs. Classical Algorithms
    Although QISKit ACQUA is mostly a library of
    `quantum algorithms <./algorithms.html#quantum-algorithms>`__,
    it also includes a number of
    `classical algorithms <./algorithms.html#classical-algorithms>`__,
    which can be selected to generate reference values
    and compare and contrast results in quantum research experimentation.
    Since a classical algorithm runs on a classical computer,
    no backend should be configured when a classical algorithm
    is selected in the ``algorithm`` section.
    Accordingly, the QISKit ACQUA `GUI <#gui>` will automatically
    disable the ``backend`` configuration section
    whenever a non-quantum algorithm is selected. 

Configuring the backend to use by a `quantum algorithm <./algorithms.html#quantum-algorithms>`__
requires setting the following parameters too:

-  The number of repetitions of each circuit to be used for sampling:

   .. code:: python

        shots : int

   This parameter applies, in particular to the local QASM simulator and any real quantum device.  The default
   value is ``1024``. 
   
-  A ``bool`` value indicating whether or not the circuit should undergo optimization:

   .. code:: python
       
        skip_transpiler : bool

   The default value is ``False``.  If ``skip_transpiler`` is set to ``True``, then
   QISKit will not perform circuit translation. If QISKit ACQUA has been configured
   to run an experiment with a quantum algorithm that uses only basis gates,
   then no translation of the circuit into basis gates is required.
   Only in such cases is it safe to skip circuit translation.
   Skipping the translation phase when only basis gates are used may improve overall performance,
   especially when many circuits are used repeatedly, as it is the case with the VQE algorithm.

   .. note::
       Use caution when setting ``skip_transpiler`` to ``True``
       as if the quantum algorithm does not restrict itself to the set of basis
       gates supported by the backend, then the circuit will fail to run.

-  An optional dictionary can be supplied to control the backend's noise model (see
   the documentation on `noise parameters
   <https://github.com/QISKit/qiskit-sdk-py/tree/master/src/qasm-simulator-cpp#noise-parameters>`__
   for more details):

   .. code:: python

       noise_params : dictionary

   This is a Python dictionary consisting of key/value pairs.  Configuring it is optional; the default
   value is ``None``.

   The following is an example of such a dictionary that can be used:

   .. code:: python

      "noise_params": {"U": {"p_depol": 0.001,
                             "p_pauli": [0, 0, 0.01],
                             "gate_time": 1,
                             "U_error": [ [[1, 0], [0, 0]]
                                        ]
                            }
                      }
