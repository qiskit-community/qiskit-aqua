Configuring and Running an Experiment
=====================================

Qiskit AQUA Chemistry supports two types of users:

1. *Chemistry practitioners*, who are merely interested in executing
   Qiskit AQUA Chemistry as a tool to compute chemistry properties.
   These users may not be interested in extending Qiskit AQUA Chemistry
   with additional capabilities.  In fact, they may not even be interested
   in learning the details of quantum computing, such as the notions of
   circuits, gates and qubits.  What these users expect
   from quantum computing is the gains in performance and accuracy, and
   the reduction in computational complexity.
2. *Chemistry and quantum researchers*, who are interested in extending
   Qiskit AQUA Chemistry with new computational chemistry software drivers,
   new operators for classical-to-quantum
   input translation, and/or new quantum algorithms for more efficient
   and accurate computations.

In this section, we cover the first class of users --- the chemistry practitioners.
Specifically, this section describes how Qiskit AQUA Chemistry can be accessed as a
tool for quantum-based chemistry computations.

To see how you can extend Qiskit AQUA Chemistry with new components,
please refer to `Section "Contributing to Qiskit AQUA Chemistry <./extending.html>`__.

Execution Modes
---------------

Qiskit AQUA Chemistry has both `Graphical User Interface (GUI) <#gui>`__ and `command
line <#command-line>`__ tools, which may be used when solving chemistry
problems. Both can load and run an `input
file <#input-file>`__ specifying a molecule configuration and the quantum
algorithm to be used for the computation, along with the algorithm configuration
and various other options to
customize the experiment.  If you are new to
Qiskit AQUA Chemistry, we highly recommend getting started with the GUI.
Finally, QISKIT AQUA Chemistry can also be accessed
`programmatically <#programmable-interface>`__ by users interested
in customizing the experiments beyond what the command line and GUI can offer.

GUI
~~~

The GUI provides an easy means to create an input file from scratch, or to load
an existing input file, and then run that input file to experiment with a
chemistry problem on a quantum machine.
An input file is created,
edited and saved with validation of parameter values.

When `installing <./install.html>`__
Qiskit AQUA Chemistry via the ``pip install`` command,
a script is created that allows you to start the GUI from the command line,
as follows:

.. code:: sh

   qiskit_aqua_chemistry_ui

If you cloned Qiskit AQUA Chemistry directly from the
`GitHub repository <https://github.com/Qiskit/aqua-chemistry>`__ instead of using ``pip
install``, then the script above will not be present and the launching command should be instead:

.. code:: sh

   python qiskit_aqua_chemistry/ui

This command must be launched from the root folder of the ``qiskit-aqua-chemistry`` repository
clone.

When executing an Qiskit AQUA Chemistry problem using the GUI, the user can choose
to specify a `JavaScript Object Notation (JSON) <http://json.org>`__
output file name by selecting the **Generate Algorithm Input**
checkbox.  When this is done,
Qiskit AQUA Chemistry will not attempt to bring the chemistry experiment to completion; rather,
it will stop the execution of the experiment right after forming the input for the
quantum algorithm, before invoking that algorithm, and
will serialize the input to the quantum algorithm in a
`JSON file for direct algorithm invocation <#input-file-for-direct-algorithm-invocation>`__
later on.

Command Line
~~~~~~~~~~~~

Qiskit AQUA Chemistry, if `installed <./install.html>`__ via ``pip install``,
comes with the following command-line tool:

.. code:: sh

   qiskit_aqua_chemistry_cmd

If you cloned Qiskit AQUA Chemistry from its remote
`GitHub repository <https://github.com/Qiskit/aqua-chemistry>`__
instead of using ``pip install``, then the command-line interface can be executed as follows:

.. code:: sh

   python qiskit_aqua_chemistry

from the root folder of the ``aqua-chemistry`` repository clone.

Here is a summary of the command-line options:

.. code:: sh

   usage: qiskit_aqua_chemistry_cmd [-h] [-o output | -jo json output] input

   Quantum Chemistry Program.

   positional arguments:
     input            Qiskit AQUA Chemistry input file

   optional arguments:
     -h, --help       Show this help message and exit
     -o output        Output file name
     -jo json output  JSON output file name

As shown above, in addition to the mandatory input file name parameter, the user can
specify an output file name where the output of the chemistry problem
will be saved (otherwise it will just be printed
on the command screen) or, alternatively, a JSON output file name.  When the latter is specified,
Qiskit AQUA Chemistry will not attempt to bring the chemistry experiment to completion; rather,
it will stop its execution right after forming the input for the
quantum algorithm specified in the input file, before invoking that algorithm, and
will serialize the input to the quantum algorithm `JSON file for direct algorithm
invocation <#input-file-for-direct-algorithm-invocation>`__
later on.

Programmable Interface
~~~~~~~~~~~~~~~~~~~~~~

Qiskit AQUA Chemistry also offers Application Programming Interfaces (APIs)
to execute experiments programmatically. Numerous
examples on how to do so
can be found in the
`chemistry folder of the Qiskit AQUA Tutorials GitHub repository
<https://github.com/Qiskit/aqua-tutorials/tree/master/chemistry>`__.

Programming an Experiment Step by Step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It is very well possible to program an experiment step by step by invoking
all the necessary APIs one by one to construct the flow that executes a
classical computation software with a given molecular configuration,
extracts from that execution the molecular structural data necessary to form
the input to one of the Qiskit AQUA quantum algorithms, and finally invokes that algorithm
to build, compile and execute a circuit modeling the experiment on top of a quantum
machine.  An example of this is available in the `PySCF_end2end tutorial
<https://github.com/Qiskit/aqua-tutorials/blob/master/chemistry/PySCF_end2end.ipynb>`__.

Declarative Programming Interface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

It should be noted, however, that Qiskit AQUA Chemistry is
designed to be programmed in a declarative way as well.  This was done in order
to simplify the programmatic access to Qiskit AQUA Chemistry,
minimizing the chances for configuration errors, and addressing the needs of users
who might be experts in chemistry but not interested in writing a lot of code or
learning new Application Programming Interfaces (APIs).  Even though there is
nothing preventing a user from accessing the Qiskit AQUA Chemistry APIs and
programming an experiment step by step, Qiskit AQUA Chemistry lets you
build a Python dictionary from an `input file <#input-file>`__.  This can be achieved via the
`GUI <#gui>`__
by loading (or creating from scratch) the input file representing the 
configuration of the desired experiment, and by then selecting **Export Dictionary**
from the **File** menu.  Assuming that the programmer assigns the
exported dictionary to variable ``aqua_chemistry_dict``, then the
experiment can be executed with the following two lines of code:

.. code:: python

   solver = AQUAChemistry()
   result = solver.run(aqua_chemistry_dict)

Executing the Python dictionary extracted from the `input file <#input-file>`__
via a call to the ``run`` method of an ``AQUAChemistry`` solver
is essentially what the `command line <#command-line>`__ and `GUI <#gui>`__
do too in order to execute an experiment.

The advantage of this approach is that users can now programmatically customize the
Python dictionary extracted from the GUI according to their needs.
Since a Python dictionary can be updated programmatically, the programmable
interface of Qiskit AQUA Chemistry makes it
possible to carry out experiments that are more complicated than those
that can be executed via the command line or the GUI.

The following example shows a simple programmatic use of two Python dictionaries extracted from
the Qiskit AQUA Chemistry `GUI <#gui>`__ in order to compute the ground-state molecular
energy of a hydrogen molecule computed via the
`Quantum Phase Estimation (QPE)
<https://qiskit.org/documentation/aqua/algorithms.html#quantum-phase-estimation-qpe>`__
algorithm and compare that result against the reference value computed via the
`Exact Eigensolver <https://qiskit.org/documentation/aqua/algorithms.html#exact-eigensolver>`__
classical algorithm.  A comparison with the Hartree-Fock energy is also offered.

.. code:: python

    distance = 0.735
    molecule = 'H .0 .0 0; H .0 .0 {}'.format(distance)

    # Input dictionaries to configure Qiskit AQUA Chemistry using QPE and Exact Eigensolver
    aqua_chemistry_qpe_dict = {
        'driver': {'name': 'PYSCF'},
        'PYSCF': {
            'atom': molecule, 
            'basis': 'sto3g'
        },
        'operator': {'name': 'hamiltonian', 'transformation': 'full', 'qubit_mapping': 'parity'},
        'algorithm': {
            'name': 'QPE',
            'num_ancillae': 9,
            'num_time_slices': 50,
            'expansion_mode': 'suzuki',
            'expansion_order': 2,
        },
        'initial_state': {'name': 'HartreeFock'},
        'backend': {
            'name': 'local_qasm_simulator',
            'shots': 100,
        }
    }

    aqua_chemistry_ees_dict = {
        'driver': {'name': 'PYSCF'},
        'PYSCF': {'atom': molecule, 'basis': 'sto3g'},
        'operator': {'name': 'hamiltonian', 'transformation': 'full', 'qubit_mapping': 'parity'},
        'algorithm': {
            'name': 'ExactEigensolver',
        },
    }

    # Execute the experiments
    result_qpe = AQUAChemistry().run(aqua_chemistry_qpe_dict)
    result_ees = AQUAChemistry().run(aqua_chemistry_ees_dict)

    # Extract the energy values
    print('The ground-truth ground-state energy is       {}.'.format(result_ees['energy']))
    print('The ground-state energy as computed by QPE is {}.'.format(result_qpe['energy']))
    print('The Hartree-Fock ground-state energy is       {}.'.format(result_ees['hf_energy']))

More complex examples include
`plotting the dissociation curve 
<https://github.com/Qiskit/aqua-tutorials/blob/master/chemistry/lih_dissoc.ipynb>`__
or `comparing results obtained via different algorithms 
<https://github.com/Qiskit/aqua-tutorials/blob/master/chemistry/lih_uccsd.ipynb>`__.

Result dictionary
^^^^^^^^^^^^^^^^^

As can be seen in the programmable-interface example above, the
``AQUAChemistry`` ``run`` method returns a result dictionary.
The unit of measure for the energy values is
Hartree, while for the dipole-moment values it is atomic units (a.u.).

The dictionary contains the following fields of note:

-  ``energy``: the ground state energy

-  ``energies``: an array of energies comprising the ground-state molecular energy and any
   excited states if they were computed

-  ``nuclear_repulsion_energy``: the nuclear repulsion energy

-  ``hf_energy``: the Hartree-Fock ground-state molecular energy as computed by the driver

-  ``nuclear_dipole_moment``, ``electronic_dipole_moment``, ``dipole_moment``:
   nuclear, electronic, and combined dipole moments for ``x``, ``y`` and ``z``

-  ``total_dipole_moment``: total dipole moment

-  ``algorithm_retvals``:  The result dictionary of the
   `algorithm <https://qiskit.org/documentation/aqua/algorithms.html>`__
   that produced the values in the 

Input File
----------

An input file is used to define a chemistry problem,
and includes both chemistry and quantum configuration information. It contains at a
minimum the definition of a molecule and its associated configuration, such
as a basis set, in order to compute the electronic structure using an
external ab-initio `chemistry driver <./drivers.html>`__. Further configuration can also be supplied to
explicitly control the processing and the quantum algorithm, used for
the computation, instead of using defaulted values when none are
supplied.

Several sample input files can be found in the `chemistry folder of
the aqua-tutorials repository
<https://github.com/Qiskit/aqua-tutorials/tree/master/chemistry/input_files>`__.

An input file comprises the following main sections, although not all
are mandatory:

``name``
~~~~~~~~

This is an optional free-format text section. Here you can name and
describe the problem solved by the input file. For example:

.. code:: python

   &name
      H2 molecule experiment
      Ground state energy computed via Variational Quantum Eigensolver
   &end

``driver``
~~~~~~~~~~

This is a mandatory section, which defines the molecule and
associated configuration for the electronic-structure computation by the
chosen driver via its external computational chemistry program. The exact
form of the configuration depends on the specific driver being used since
Qiskit AQUA Chemistry allows external drivers to be the system's front-ends,
without interposing any new programming language or API
on top of existing drivers.

Here are a couple of examples.
Note that the ``driver`` section names which specific chemistry driver will
be used, and a subsequent section in the input file, having the name of the driver, then
supplies the driver specific configuration.  For example, if you
choose ``PSI4`` as the driver, then a section called ``psi4`` must
be defined, containing the molecular configuration written as a PSI4
input file.  Users who have already collected input files for existing drivers
can simply paste those files' contents into this section.

The following is an example showing how to use the `PySCF
driver <./drivers.html#pyscf>`__ for the configuration of a Lithium Hydride (LiH) molecule.  The
``driver`` section names ``PYSCF`` as the driver and then a ``pyscf`` section,
corresponding to the name of the chosen driver, must be provided in order to define,
at a minimum, the geometrical coordinates of the molecule's atoms
and basis set (or sets) that will
be used by PySCF library to compute the
electronic structure.

.. code:: python

   &driver
      name=PYSCF
   &end

   &pyscf
      atom=Li 0.0 0.0 -0.8; H 0.0 0.0 0.8
      unit=Angstrom
      basis=sto3g
   &end

Here is another example showing again how to configure the same LiH molecule as above,
this time using the `PSI4 driver <./drivers.html#psi4>`__. Here, ``PSI4``
is named as the driver to be used and the ``psi4`` section contains the
molecule and basis set (or sets) directly in a form that PSI4 understands. The
language in which the molecular configuration is input is 
the input-file language for PSI4, and thus should be familiar to
existing users of PSI4, who may have already collected such an input file
from previous experiments and whose only job at this point would be to copy and paste
its contents into the ``psi4`` section of the input file.

.. code:: python

       &psi4
          molecule LiH {
             0 1
             Li 0.0 0.0 -0.8
             H  0.0 0.0  0.8
          }

          set {
             basis sto-3g
             scf_type pk
          }
       &end

The Qiskit AQUA Chemistry `documentation on drivers <./drivers.html>`__
explains how to install and configure the drivers currently interfaced by
Qiskit AQUA Chemistry.

As shown above, Qiskit AQUA Chemistry allows input files from the classical driver
libraries to be used directly, without any modification and without interposing
any new programming language or API.  This has a clear advantage, not only in terms
of usability, but also in terms of functionality, because any capability
of any chemistry library chosen by the user is automatically integrated into
Qiskit AQUA Chemistry, which would not have been possible if a new language or
API had been interposed between the library and the user.

``operator``
~~~~~~~~~~~~

This is an optional section. This section can be configured to
control the operator that converts the electronic structure information, obtained from the
driver, to qubit-operator form, in order to be processed by
the algorithm. The following parameters may be set:

- The name of the operator:

  .. code:: python

      name = hamiltonian

  This parameter accepts a ``string`` value.  However, currently,
  ``hamiltonian`` is the only value allowed for ``name`` since there is only
  one operator entity at present. The translation layer of Qiskit AQUA Chemistry
  is extensible and new translation operators can be plugged in.  Therefore,
  in the future, more operators may be supported.

-  The transformation type of the operator:

   .. code:: python

       transformation = full | particle_hole

   The ``transformation`` parameter takes a ``string`` value.  The only
   two allowed values, currently, are ``full`` and ``particle_hole``,
   with ``full``, the default one, corresponding to the standard second
   quantized hamiltonian.  Setting the ``transformation`` parameter
   to ``particle_hole`` yields a transformation of the electronic structure
   Hamiltonian in the second quantization framework into the
   particle-hole (p/h) picture, which offers
   a better starting point for the expansion of the trial wave function
   from the Hartree Fock reference state.
   For trial wave functions in Qiskit AQUA, such as
   `Unitary Coupled Cluster Singles and Doubles (UCCSD)
   <https://qiskit.org/documentation/aqua/variational_forms.html#uccsd>`__, the
   p/h Hamiltonian can improve the speed of convergence of the
   `Variational Quantum Eigensolver (VQE) algorithm
   <https://qiskit.org/documentation/aqua/algorithms.html#variational-quantum-eigensolver-vqe>`__
   in the calculation of the electronic ground state properties. 
   More information on the p/h formalism can be found in
   `arXiv:1805.04340 <https://arxiv.org/abs/1805.04340>`__.

-  The desired mapping from fermion to qubit:

   .. code:: python

       qubit_mapping = jordan_wigner | parity | bravyi_kitaev

   This parameter takes a value of type ``string``.  Currently, only the three values
   above are supported, but new qubit mappings can easily be plugged in.
   Specifically:

   -  ``jordan_wigner`` corresponds to the
      `Jordan-Wigner transformation <https://rd.springer.com/article/10.1007%2FBF01331938>`__,
      which maps spin operators onto fermionic creation and annihilation operators.
      It was proposed by Ernst Pascual Jordan and Eugene Paul Wigner
      for one-dimensional lattice models,
      but now two-dimensional analogues of the transformation have also been created.
      The Jordan–Wigner transformation is often used to exactly solve 1D spin-chains
      by transforming the spin operators to fermionic operators and then diagonalizing
      in the fermionic basis.
   -  ``parity``, the default value for the ``qubit_mapping`` parameter, corresponds to the
      `parity-mapping transformation <https://arxiv.org/abs/1701.08213>`__.
      This mapping optimizes encodings of fermionic many-body systems by qubits
      in the presence of symmetries.
      Such encodings eliminate redundant degrees of freedom in a way that preserves
      a simple structure of the system Hamiltonian enabling quantum simulations with fewer qubits. 
   -  ``bravyi_kitaev`` corresponds to the
      `binary-tree-based qubit mapping
      <https://www.sciencedirect.com/science/article/pii/S0003491602962548>`__,
      which was proposed by Sergey B. Bravyi and Alexei Yu. Kitaev.
      The Bravyi–Kitaev transformation is a method of mapping the occupation state of a
      fermionic system onto qubits. This transformation maps the Hamiltonian of :math:`n`
      interacting fermions to an O(log :math:`n`)‐local Hamiltonian of :math:`n` qubits.
      This is an improvement in locality over the Jordan–Wigner transformation, which results
      in an O(:math:`n`)‐local qubit Hamiltonian.
     

-  A Boolean flag specifying whether or not to apply the precision-preserving two-qubit reduction
   optimization:

   .. code:: python

       two_qubit_reduction : bool

   When the parity mapping is selected, the operator can be reduced by two qubits without loss
   of precision.  The default value for this parameter is ``False``.

-  The maximum number of workers used when forming the input to the Qiskit AQUA quantum algorithm:

   .. code:: python   

       max_workers = 1 | 2 | ...

   Processing of the hamiltonian from fermionic to qubit can take
   advantage of multiple CPU cores to run parallel processes to carry
   out the transformation. The number of such worker processes used will
   not exceed the actual number of CPU cores or this ``max_workers`` positive integer,
   whichever is the smaller.  The default value for ``max_worker`` is ``4``.

-  A Boolean value indicating whether or not to freeze the core orbitals in the computation:

   .. code:: python   

       freeze_core : bool

   To reduce the number of qubits required to compute the molecular energy values,
   and improve computation efficiency, frozen
   core orbitals corresponding to the nearest noble gas can be removed
   from the subsequent computation performed by the
   Qiskit AQUA algorithm, and a corresponding offset from this removal is added back
   into the final computed result. This approximation may be combined with
   ``orbital_reduction`` setting below.  The default value for this parameter is ``False``.

-  A list of molecular orbital to remove from the computation:

   .. code:: python

       orbital_reduction : [int]

   The orbitals from the electronic structure can be simplified for the
   subsequent computation.  With this parameter, you can specify a list of orbitals as
   a list of ``int`` values, the default
   being an empty list.  Each value in the list corresponds to an orbital
   to be removed from the subsequent computation.
   The list should be indices of the orbitals from ``0`` to :math:`n-1`, where the
   electronic structure has :math:`n` orbitals.
   
   For ease of referring to
   the higher orbitals, the list also supports negative values with ``-1``
   being the highest unoccupied orbital, ``-2`` the next one down, and so on.
   Also note that, while orbitals may be listed to reduce the overall
   size of the problem, the final computation can be less accurate as a result of
   using this approximation.

   The following should be taken into account when assigning a value to the ``orbital_reduction``
   parameter:

   -  Any orbitals in the list that are *occupied orbitals* are frozen and an offset
      is computed from their removal. This is the same procedure as that one that takes place
      when ``freeze_core`` is set to ``True``, except that with ``orbital_reduction``
      you can specify exactly the
      orbitals you want.

   -  Any orbitals in the list that are *unoccupied virtual orbitals* are
      simply eliminated entirely from the subsequent computation.

   When a list is specified along with ``freeze_core`` set to ``True``, the effective
   orbitals being removed from the computation are those in the frozen core combined with
   those specified in the ``orbital_reduction`` list.

   Below is an example where, in addition to freezing the core orbitals,
   a couple of other orbitals are listed for removal. We assume that there
   are a total of ten orbitals, so the highest two unoccupied virtual orbitals will
   be eliminated from the subsequent computation, in addition to the frozen-core
   orbitals:

   .. code:: python

       &operator
          name=hamiltonian
          qubit_mapping=jordan_wigner
          freeze_core=true
          orbital_reduction=[8, 9]
       &end

   Alternatively, the above code could be specified via the following,
   eqivalent way,
   which simplifies
   expressing the higher orbitals using the fact that the numbering is relative to the
   highest orbital:

   .. code:: python

       &operator
          name=hamiltonian
          qubit_mapping=jordan_wigner
          freeze_core=true
          orbital_reduction=[-2, -1]
       &end

``algorithm``
~~~~~~~~~~~~~

This is an optional section that allows you to specify which
algorithm will be used by the computation.
`Quantum algorithms
<https://qiskit.org/documentation/aqua/algorithms.html#quantum-algorithms>`__ are provided by
`QISKIt
AQUA <https://qiskit.org/documentation/aqua/qiskit-aqua.html>`__.
To compute reference values, Qiskit AQUA also allows the use of
`classical algorithms
<https://qiskit.org/documentation/aqua/algorithms.html#classical-algorithms>`__.
In the ``algorithm`` section, algorithms are disambiguated using the
`declarative names <https://qiskit.org/documentation/aqua/algorithms.html>`__
by which Qiskit AQUA recognizes them, based on the JSON schema
each algorithm must provide according to the Qiskit AQUA ``QuantumAlgorithm`` API.
The declarative name is specified as the ``name`` parameter in the ``algorithm`` section.
The default value for the ``name`` parameter is ``VQE``, corresponding
to the `Variational Quantum Eigensolver (VQE)
<https://qiskit.org/documentation/aqua/algorithms.html#variational-quantum-eigensolver-vqe>`__
algorithm.

An algorithm typically comes with a set of configuration parameters.
For each of them, a default value is provided according to the
``QuantumAlgorithm`` API of Qiskit AQUA.

Furthermore, according to each algorithm, additional sections
may become relevant to optionally
configure that algorithm's components.  For example, variational algorithms,
such as `VQE
<https://qiskit.org/documentation/aqua/algorithms.html#variational-quantum-eigensolver-vqe>`__,
allow the user to choose and configure an
`optimizer <https://qiskit.org/documentation/aqua/optimizers.html>`__ and a
`variational form <https://qiskit.org/documentation/aqua/variational_forms.html>`__,
whereas `Quantum Phase Estimation (QPE)
<https://qiskit.org/documentation/aqua/algorithms.html#quantum-phase-estimation-qpe>`__
allows the user to configure which `Inverse Quantum Fourier Transform (IQFT)
<https://qiskit.org/documentation/aqua/iqfts.html>`__ to use.

The `Qiskit AQUA documentation <https://qiskit.org/documentation/aqua/index.html>`__
explains how to configure each algorithm and any of the pluggable entities it may use,
such as `optimizers <https://qiskit.org/documentation/aqua/optimizers.html>`__,
`variational forms <https://qiskit.org/documentation/aqua/variational_forms.html>`__,
`initial states <https://qiskit.org/documentation/aqua/initial_states.html>`__,
`oracles <https://qiskit.org/documentation/aqua/oracles.html>`__, and
`Inverse Quantum Fourier Transforms (IQFTs)
<https://qiskit.org/documentation/aqua/iqfts.html>`__.

Here is an example in which the algorithm `VQE
<https://qiskit.org/documentation/aqua/algorithms.html#variational-quantum-eigensolver-vqe>`__
is selected along with the `Limited-memory Broyden-Fletcher-Goldfarb-Shanno Bound (L-BFGS-B)
<https://qiskit.org/documentation/aqua/optimizers.html#limited-memory-broyden-fletcher-goldfarb-shanno-bound-l-bfgs-b>`__
optimizer and the
`RYRZ <https://qiskit.org/documentation/aqua/variational_forms.html#ryrz>`__ variational form:

.. code:: python

   &algorithm
      name=VQE
      shots=1
      operator_mode=matrix
   &end

   &optimizer
      name=L_BFGS_B
      factr=10
   &end

   &variational_form
      name=RYRZ
      entangler_map={0: [1]}
   &end

``backend``
~~~~~~~~~~~

Qiskit AQUA allows for configuring the *backend*, which is the quantum machine
on which a quantum experiment will be run.
This configuration requires specifying 
the `Qiskit <https://www.qiskit.org/>`__ quantum computational
backend to be used for computation, which is done by assigning a ``string`` value to
the ``name`` parameter of the ``backend`` section:

.. code:: python

    name : string

The value of the ``name`` parameter indicates either a real-hardware
quantum computer or a quantum simulator.
The underlying Qiskit core used by Qiskit AQUA comes
with two predefined quantum device simulators: the *local state vector simulator* and
the *local QASM simulator*, corresponding to the following two
values for the ``name`` parameter: ``"local_statevector_simulator"`` (which
is the default value for the ``name`` parameter) and ``"local_qasm_simulator"``, respectively.
However, any suitable quantum backend can be selected, including
a real quantum hardware device. The ``QConfig.py`` file
needs to be setup for Qiskit to access remote devices.  For this, it is sufficient to follow the
`Qiskit installation instructions <https://qiskit.org/documentation/install.html#installation>`__.
The Qiskit AQUA Chemistry `GUI <#gui>` greatly simplifies the
configuration of ``QConfig.py`` via a user friendly interface,
accessible through the **Preferences...** menu item.

.. topic:: Backend Configuration: Quantum vs. Classical Algorithms
    Although Qiskit AQUA is mostly a library of `quantum algorithms
    <https://qiskit.org/documentation/aqua/algorithms.html#quantum-algorithms>`__,
    it also includes a number of `classical algorithms
    <https://qiskit.org/documentation/aqua/algorithms.html#classical-algorithms>`__,
    which can be selected to generate reference values
    and compare and contrast results in quantum research experimentation.
    Since a classical algorithm runs on a classical computer,
    no backend should be configured when a classical algorithm
    is selected in the ``algorithm`` section.
    Accordingly, the Qiskit AQUA Chemistry `GUI <#gui>` will automatically
    disable the ``backend`` configuration section
    whenever a non-quantum algorithm is selected. 

Configuring the backend to use by a `quantum algorithm
<https://qiskit.org/documentation/aqua/algorithms.html#quantum-algorithms>`__
requires setting the following parameters too:

-  The number of repetitions of each circuit to be used for sampling:

   .. code:: python

        shots : int

   This parameter applies, in particular to the local QASM simulator and any real quantum device.
   The default value is ``1024``. 
   
-  A ``bool`` value indicating whether or not the circuit should undergo optimization:

   .. code:: python
       
        skip_transpiler : bool

   The default value is ``False``.  If ``skip_transpiler`` is set to ``True``, then
   Qiskit will not perform circuit translation. If Qiskit AQUA Chemistry has been configured
   to run an experiment with a quantum algorithm that uses only basis gates,
   then no translation of the circuit into basis gates is required.
   Only in such cases is it safe to skip circuit translation.
   Skipping the translation phase when only basis gates are used may improve overall performance,
   especially when many circuits are used repeatedly, as it is the case with the `VQE 
   <https://qiskit.org/documentation/aqua/algorithms.html#variational-quantum-eigensolver-vqe>`__
   algorithm.

   .. note::
       Use caution when setting ``skip_transpiler`` to ``True``
       as if the quantum algorithm does not restrict itself to the set of basis
       gates supported by the backend, then the circuit will fail to run.

-  An optional dictionary can be supplied to control the backend's noise model (see
   the documentation on `noise parameters
   <https://github.com/Qiskit/qiskit-sdk-py/tree/master/src/qasm-simulator-cpp#noise-parameters>`__
   for more details):

   .. code:: python

       noise_params : dictionary

   This is a Python dictionary consisting of key/value pairs.  Configuring it is optional;
   the default value is ``None``.

   The following is an example of such a dictionary that can be used:

   .. code:: python

      noise_params: {"U": {"p_depol": 0.001,
                             "p_pauli": [0, 0, 0.01],
                             "gate_time": 1,
                             "U_error": [ [[1, 0], [0, 0]]
                                        ]
                          }
                    }

``problem``
~~~~~~~~~~~

In Qiskit AQUA,
a *problem* specifies the type of experiment being run.  Configuring the problem is essential
because it determines which algorithms are suitable for the specific experiment.

Problem Categories
^^^^^^^^^^^^^^^^^^
Qiskit AQUA comes with a set of predefined problems.
This set is extensible: new problems can be added,
just like new algorithms can be plugged in to solve existing problems in a different way,
or to solve new problems.
Currently, a problem can be configured by assigning a ``string`` value to the ``name`` parameter
of the ``problem`` section of the input file:

.. code:: python

    name = energy | excited_states | ising | dynamics | search | svm_classification

As shown above, ``energy``, ``excited_states``, ``ising``, ``dynamics``,
``search``, and ``svm_classification`` are currently
the only values accepted for ``name`` in Qiskit AQUA, corresponding to the computation of
*energy*, *excited states*, *Ising models*, *dynamics of evolution*, *search* and
*Support Vector Machine (SVM) classification*, respectively.
New problems, disambiguated by their
``name`` parameter, can be programmatically
added to Qiskit AQUA via the
``AlgorithmInput`` Application Programming Interface (API), and each quantum or classical
`algorithm <./algorithms.html>`__
should programmatically list the problems it is suitable for in its JSON schema, embedded into
the class implementing the ``QuantumAlgorithm`` API.  Typical choices of problems
in chemistry include energy and excited states.

Generating Repeatable Experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Aspects of the computation may include use of random numbers. For instance, 
`VQE <https://qiskit.org/documentation/aqua/algorithms.html#variational-quantum-eigensolver-vqe>`__
is coded to use a random initial point if the
`variational form <https://qiskit.org/documentation/aqua/variational_form.html>`__
does not supply any
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

Reconciling Chemistry and Quantum Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The configuration of a chemistry problem directly affects the configuration
of the underlying quantum system.  For example, the number of particles and
orbitals in a molecular system depends on the molecule being modeled and the
basis set chosen by the user, and that, in turn, directly affects the number of qubits
necessary to model the molecular system on a quantum machine.  The number of
qubits directly derived from the molecular configuration can then be reduced
as indicated in the ``operator`` section of the input file
via optimizations, such as the precision-preserving
two-qubit reduction based on the parity qubit mapping, or via approximations, obtained
by freezing the core or by virtually removing unoccupied orbitals.  This is just an example
of how the chemistry
configuration can affect the quantum configuration.  Letting the user set
the number of qubits would force the user to have to know the numbers of particles
and orbitals of the molecular system, and then precompute the number of
qubits based on the numbers of particles and
orbitals, as well as the qubit-reduction optimization
and approximation techniques.  Any mistake in this manual computation
may lead to misconfiguring the whole experiment.  For this reason,
Qiskit AQUA Chemistry automatically computes the numbers of particles and orbitals,
infers the total number of qubits necessary to model the molecular system under analysis,
and subtracts from that total number of qubits the number of qubits that are
redundant based on the optimization and approximation techniques that the user
may have chosen to apply.  In essence, Qiskit AQUA Chemistry automatically
configures the quantum system.

Things become more subtle when configuring the
`initial state <https://qiskit.org/documentation/aqua/initial_states.html>`__ and
`variational form <https://qiskit.org/documentation/aqua/variational_forms.html>`__
used by a quantum algorithm.  These components are
configured in sections ``initial_state`` and ``variational_form``, respectively,
which only become enabled when the
`algorithm <https://qiskit.org/documentation/aqua/algorithms.html>`__
selected by the user supports them.
For example, the ``variational_form`` section is enabled only
if the user has chosen to execute the experiment using a variational algorithm, such as
`VQE <https://qiskit.org/documentation/aqua/algorithms.html#variational-quantum-eigensolver-vqe>`__.
The Qiskit AQUA Chemistry `GUI <#gui>`__ disables the ``variational_form``
section for non-variational algorithms.
The problem with the configuration of an initial state and a variational form is that
the values of parameters ``qubit_mapping`` and ``two_qubit_reduction`` may require matching
their settings across these two sections, as well as the settings applied to the
identically named parameters in the ``operator``
section.  This is the case, for example, for the `Unitary Coupled Cluster Singles and Doubles (UCCSD)
<https://qiskit.org/documentation/aqua/variational_forms.html#uccsd>`__ variational form
and the `Hartree-Fock <https://qiskit.org/documentation/aqua/initial_states.html#hartree-fock>`__
initial state.  Furthermore, some variational forms and initial states may require setting
the numbers of particles (``num_particles``) and orbitals (``num_orbitals``), which,
as discussed above, can be complicated to compute, especially for large and complex molecules.

Qiskit AQUA Chemistry inherits the problem configuration from Qiskit AQUA.
However, *exclusive to Qiskit AQUA Chemistry*
is a Boolean field inside the ``problem`` section which assists users with these
complicated settings:

.. code:: python

    auto_substitutions : bool

When this parameter is set to ``True``, which is the default, the values of parameters 
``num_particles`` and ``num_orbitals`` in sections ``initial_state`` and
``variational_form`` are automatically computed by Qiskit AQUA Chemistry.  As such,
their configuration is disabled; the user will not be required to assign values to
these two parameters.  This is also reflected in the `GUI <#gui>`__, where
these parameters are grayed out.  Furthermore, Qiskit AQUA Chemistry automatically sets
parameters ``qubit_mapping`` and ``two_qubit_reduction`` in sections ``initial_state`` and
``variational_form`` to the values the user assigned to them in the ``operator`` section
of the input file in order to enforce parameter-value matching across these three different
sections.  As a result, the user will only have to configure ``qubit_mapping``
and ``two_qubit_reduction`` in the ``operator`` section; the configuration of these two
parameters in sections ``initial_state`` and ``variational_form`` is disabled,
as reflected also in the `GUI <#gui>`__, where the values of these two parameters are only
editable in the ``operator`` section, while the parameters themselves are grayed out in the
``initial_state`` and ``variational_form`` sections.

On the other hand, if ``auto_substitutions`` is set to ``False``,
then the end user has the full responsibility for the entire
configuration.  Setting ``auto_substitutions`` to ``False``, while
made possible for experimental purposes, should only
be done with extreme care, since it could easily lead to misconfiguring
the entire experiment and producing imprecise results.

Input File for Direct Algorithm Invocation
------------------------------------------

Qiskit AQUA allows for its
`algorithms <https://qiskit.org/documentation/aqua/algorithms.html>`__,
whether they are
`quantum <https://qiskit.org/documentation/aqua/algorithms.html#quantum-algorithms>`__
or `classical <https://qiskit.org/documentation/aqua/algorithms.html#classical-algorithms>`__
to be invoked directly, without necessarily
having to go through the execution of a domain-specific application.  Qiskit AQUA
Chemistry supports accessing the Qiskit AQUA algorithm-level entry point in the following way:
after the translation process terminates with the creation of the input to a quantum
algorithm, in the form of a qubit operator, Qiskit AQUA Chemistry allows for that
input to be serialized as a `JavaScript Object Notation (JSON) <http://json.org/>`__
file.

Serializing the input to the quantum algorithm at this point is useful in many scenarios
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
- Repeating an experiment in which the domain-specific parameters remain the same,
  and the only difference is in the configuration of the quantum algorithm and its
  supporting components becomes much more efficient because the user can choose to
  restart any new experiment directly at the algorithm level, thereby bypassing the
  input extraction from the driver, and the input translation into a qubit operator.
