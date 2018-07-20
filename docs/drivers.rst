Drivers
=======

QISKit ACQUA Chemistry requires a computational chemistry program or library, known as *driver*, to be installed on the
system for the electronic-structure computation.  When launched via the
`command line <./config_run.html#command-line>`__,
`Graphical User Interface (GUI) <./config_run.html#gui>`__, or
its `programmable interface <./config_run.html##programmable-interface>`__,
QISKit ACQUA Chemistry expects a driver to be specified, and a
molecular configuration to be passed in the format compatible with that driver.
QISKit ACQUA Chemistry uses the driver not only as a frontend input language, to allow the user to configure
a chemistry problem in a language that an experienced chemist is already familiar with, but also
to compute some intermediate data, which will be later on used to form the input to the
`quantum algorithm <https://qiskit.org/documentation/acqua/algorithms.html>`__.  Such intermediate date
includes the following:

1. One- and two-body integrals in Molecular Orbital (MO) basis
2. Dipole integrals
3. Molecular orbital coefficients
4. Hartree-Fock energy
5. Nuclear repulsion energy

Once extracted, the structure of this intermediate data is independent of the
driver that was used to compute it.  The only thing that could still depend on the driver
is the level of accuracy of such data; most likely,
a more elaborate driver will produce more accurate data.
QISKit ACQUA Chemistry offers the option to serialize this data in a binary format known as
Hierarchical Data Format 5 (HDF5).  This is done to allow chemists to reuse the same input data in the future
and to enable researchers to exchange
input data with each other --- which is especially useful to researchers who may not have particular
computational chemistry drivers installed on their computers.

In order for a driver to be usable by QISKit ACQUA Chemistry, an interface to that driver
must be built in QISKit ACQUA Chemistry.  QISKit ACQUA Chemistry offers the ``BaseDriver``
Application Programming Interface (API) to support interfacing new drivers.

Currently, QISKit ACQUA Chemistry comes with interfaces prebuilt
for the following four computational chemistry software drivers:

1. `Gaussian™ 16 <http://gaussian.com/gaussian16/>`__, a commercial chemistry program
2. `PSI4 <http://www.psicode.org/>`__, an open-source chemistry program built on Python
3. `PySCF <https://github.com/sunqm/pyscf>`__, an open-source Python chemistry program
4. `PyQuante <http://pyquante.sourceforge.net/>`__, a pure cross-platform open-source Python chemistry program

.. topic:: The HDF5 Driver

    A fifth driver, called HDF5, comes prebuilt in QISKit ACQUA Chemistry.  This is, in fact, the only driver
    that does not require the installation or configuration of any external computational chemistry software,
    since it is already part of QISKit ACQUA Chemistry.
    The HDF5 driver allows for chemistry input, in the form of an HDF5 file as specified above,
    to be passed into the computation.

.. topic:: Extending QISKit ACQUA Chemistry with Support for New Drivers

    The driver support in QISKit ACQUA Chemistry was designed to make the drivers pluggable and discoverable.
    In order for QISKit ACQUA Chemistry to
    be able to interface a driver library, the ``BaseDriver`` base class must be implemented in order
    to provide the interfacing code, or *wrapper*.  As part of this process, the required
    `JavaScript Object Notation (JSON) <http://json.org>`__ schema for the driver interface must
    be provided in a file named ``configuration.json``.  The interfacing code in the driver wrapper
    is responsible for constructing and populating a ``QMolecule`` instance with the electronic
    structure data listed above.  Driver wrappers implementing the ``BaseDriver`` class and the
    associated ``configuration.json`` schema file are organized in subfolders of the ``drivers`` folder
    for automatic discovery and dynamic lookup.  Consulting the existing driver interface
    implementations may be helpful in accomplishing the task of extending .

The remainder of this section describes how to install and configure the drivers currently supported
by QISKit ACQUA Chemistry.

Gaussian™ 16
------------

`Gaussian™ 16 <http://gaussian.com/gaussian16/>`__ is a commercial program for computational chemistry.
The corresponding driver wrapper in QISKit ACQUA Chemistry accesses electronic structure information from Gaussian™ 16
via the Gaussian-supplied open-source `interfacing code <http://www.gaussian.com/interfacing/>`__.

In the ``qiskit_acqua_chemistry/drivers/gaussiand/gauopen`` folder of the QISKit ACQUA Chemistry installation package,
the Python part of the above interfacing code, as needed by QISKit ACQUA Chemistry,
has been made available. It is licensed under a `Gaussian Open-Source Public License(./gauopen/LICENSE.txt) which can
also be found in the ``gauopen`` folder.

Part of this interfacing code --- specifically, the Fortran file ``qcmatrixio.F`` --- requires compilation to a Python native extension. However,
QISKit ACQUA Chemistry comes with pre-built binaries for most common platforms. If there is no pre-built binary
matching your platform, then it will be necessary to compile this file as per the instructions below.  

Compiling the Fortran Interfacing Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If no prebuilt native extension binary, as supplied with QISKit ACQUA Chemistry, works for your platform, then
to use the Gaussian™ 16 driver on your machine, the Fortran file ``qcmatrixio.F`` must be compiled into object code that can
be used by Python. This is accomplished using the
`Fortran to Python Interface Generator (F2PY) <https://docs.scipy.org/doc/numpy/f2py/>`__,
which is part of the `NumPy <http://www.numpy.org/>`__ Python library.
Specifically, on your command prompt window, change directory to the ``qiskit_acqua_chemistry/drivers/gaussiand/gauopen``
directory inside the QISKit ACQUA Chemistry installation directory, and while in the Python environment
created for QISKit ACQUA and QISKit ACQUA Chemistry, invoke ``f2py`` on ``qcmatrixio.F`` as follows:


Apple macOS and Linux
^^^^^^^^^^^^^^^^^^^^^

The full syntax of the ``f2py`` command on macOS and Linux is as follows:

.. code:: sh

    f2py -c -m qcmatrixio qcmatrixio.F

This command will generate a file with name prefix ``qcmatrixio`` and extension ``so``, for example
``qcmatrixio.cpython-36m-x86_64-linux-gnu.so``.
In order for the command above to work and such file to be generated, you will need a supported Fortran compiler installed.
On macOS, you may have to download the `GNU Compiler Collection (GCC) <https://gcc.gnu.org/>__
and, in particular, the `GFortran Compiler <https://gcc.gnu.org/fortran/>`__ source and compile it first
if you do not a suitable Fortran compiler installed
On Linux you may be able to download and install a supported Fortran compiler via your distribution's installer.

Microsoft Windows
^^^^^^^^^^^^^^^^^

The following steps can be used with the Intel Fortran compiler on the Microsoft Windows platform:

1. Set up the environment by adding the following line to ``ifortvars.bat``:

   .. code:: sh

       ifortvars -arch intel64

2. Issue the following command from within the ``gauopen`` directory:

   .. code:: sh

       f2py -c --fcompiler=intelvem -m qcmatrixio qcmatrixio.F

   Upon successful execution, the ``f2py`` command above will generate a file with name prefix ``qcmatrixio`` and
   extension ``so``, for example ``qcmatrixio.cp36-win_amd64.pyd``.  However, in order for the ``f2py`` command above
   to work, ``#ifdef`` may need to be manually edited if it is not recognized or supported during the processing of the ``f2py`` command
   above.  For example, with ``f2py`` from Intel Visual Fortran Compiler with Microsoft Visual Studio, the following code snippet
   originally shows two occurrences of the line ``Parameter (Len12D=8,Len4D=8)``, as shown next:

   .. code::

       #ifdef USE_I8
           Parameter (Len12D=8,Len4D=8)
       #else
           Parameter (Len12D=4,Len4D=4)
       #endif

   This may need to be simplified by deleting the first three lines and the last line, leaving just the fourth line, as follows:

   .. code::

       Parameter (Len12D=4,Len4D=4)

Verifying Path and Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

You should also make sure the Gaussian™ 16 ``g16`` executable can be run from a command line.
This requires verifying that the ``g16`` executable is reachable via the system environment path, and appropriate
exports, such as ``GAUSS_EXEDIR``, have been configured as per
`Gaussian installation instructions <http://gaussian.com/techsupport/#install]>__.

Special Notes for macOS X
~~~~~~~~~~~~~~~~~~~~~~~~~

If your account is using the bash shell on a macOS X machine, you can edit the ``.bash_profile`` file
in your account's home directory and add the following lines:


.. code:: sh

    export GAUSS_SCRDIR=~/.gaussian
    export g16root=/Applications
    alias enable_gaussian='. $g16root/g16/bsd/g16.profile'

The above assumes that the application Gaussian™ 16 was placed in the ``/Applications`` folder and that
``~/.gaussian`` is the full path to
the selected scratch folder, where Gaussian™ 16 stores its temporary files. 
 
Now, before QISKit ACQUA Chemistry can properly interface Gaussian™ 16, you will have to run the ``enable_gaussian`` command
defined above.  This, however, may generate the following error:

.. code:: sh

    bash: ulimit: open files: cannot modify limit: Invalid argument

While this error is not harmful, you might want to suppress it, which can be done by entering the following sequence
of commands on the command line:

.. code:: sh

    echo kern.maxfiles=65536 | sudo tee -a /etc/sysctl.conf
    echo kern.maxfilesperproc=65536 | sudo tee -a /etc/sysctl.conf
    sudo sysctl -w kern.maxfiles=65536
    sudo sysctl -w kern.maxfilesperproc=65536
    ulimit -n 65536 65536 

as well as finally adding the following line to the ``.bash_profile`` file in your account's home directory:

.. code:: sh

    ulimit -n 65536 65536

At the end of this configuration, the ``.bash_profile`` in your account's home directory should have a section in it
like in the following script snippet:

.. code:: sh

    # Gaussian 16
    export GAUSS_SCRDIR=~/.gaussian
    export g16root=/Applications
    alias enable_gaussian='. $g16root/g16/bsd/g16.profile'
    ulimit -n 65536 65536

Input File Example
~~~~~~~~~~~~~~~~~~

To use Gaussian™ 16 to configure a molecule on which to do a chemistry experiment with QISKit ACQUA Chemistry,
set the ``name`` field in the ``driver`` section of the `input file <./config_run.html#input-file>`__ to ``GAUSSIAN`` and
then create a ``gaussian``section in the input file as per the example below, which shows the configuration of a molecule of
hydrogen.  Here, the molecule, basis set and other options are specified according
to the Gaussian™ 16 control file, so the syntax specified by Gaussian™ 16 should be followed:

.. code::

    &gaussian
       # rhf/sto-3g scf(conventional)

       h2 molecule

       0 1
       H   0.0  0.0    0.0
       H   0.0  0.0    0.74
    &end

Experienced chemists who already have existing Gaussian™ 16 control files can simply paste the contents of those files
into the ``gaussian`` section of the input file.  This configuration can also be easily achieved using the
QISKit ACQUA Chemistry `Graphical User Interface (GUI) <./config_run.html#gui>`__.

PSI4
----
`PSI4 <http://www.psicode.org/>`__ is an open-source program for computational chemistry.
In order for QISKit ACQUA Chemistry to interface PSI4, accept PSI4 input files and execute PSI4 to extract
the electronic structure information necessary for the computation of the input to the quantum algorithm,
PSI4 must be installed and discoverable on the system where QISKit ACQUA Chemistry is installed.
Therefore, once PSI4 has been installed, the ``psi4`` executable must be reachable via the system environment path.
For example, on macOS, this can be achieved by adding the following section to the ``.bash_profile`` file in the
user's home directory:

.. code:: sh

    # PSI4
    alias enable_psi4='export PATH=/Users/marcopistoia/psi4conda/bin:$PATH'

In order for QISKit ACQUA Chemistry to discover PSI4 at run time, it is then necessary to execute the ``enable_psi4`` command
before launching QISKit ACQUA Chemistry.

To use PSI4 to configure a molecule on which to do a chemistry experiment with QISKit ACQUA Chemistry,
set the ``name`` field in the ``driver`` section of the `input file <./config_run.html#input-file>`__ to ``PSI4`` and
then create a ``psi4``section in the input file as per the example below, which shows the configuration of a molecule of
hydrogen.  Here, the molecule, basis set and other options are specified according
to the PSI4 control file, so the syntax specified by PSI4 should be followed:

.. code:: python

    &psi4
       molecule h2 {
          0 1
          H 0.0 0.0 0.0
          H 0.0 0.0 0.74
       }

       set {
          basis sto-3g
          scf_type pk
       }
    &end

Experienced chemists who already have existing PSI4 control files can simply paste the contents of those files
into the ``psi4`` section of the input file.  This configuration can also be easily achieved using the
QISKit ACQUA Chemistry `Graphical User Interface (GUI) <./config_run.html#gui>`__.