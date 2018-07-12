Installation and Setup
======================

Dependencies
------------

As QISKit ACQUA Chemistry is built upon QISKit ACQUA.
Like QISKit ACQUA, at least `Python 3.5 or
later <https://www.python.org/downloads/>`__ is needed to use QISKit
ACQUA Chemistry. In addition, `Jupyter
Notebook <https://jupyter.readthedocs.io/en/latest/install.html>`__ is
recommended for interacting with the tutorials. For this reason we
recommend installing the `Anaconda
3 <https://www.continuum.io/downloads>`__ Python distribution, as it
comes with all of these dependencies pre-installed.


Installation
------------

We encourage you to install QISKit ACQUA Chemistry via the pip tool (a
Python package manager):

.. code:: sh

   pip install qiskit-acqua-chemistry

pip will handle all dependencies automatically and you will always
install the latest (and well-tested) release version.

We recommend using Python virtual environments to improve your
experience.

Running a Chemistry Experiment
------------------------------

Now that you have installed QISKit ACQUA Chemistry, you can run an
experiment, for example to compute the ground state energy of a
molecule.

QISKit ACQUA Chemistry has both `GUI <#gui>`__ and `command
line <#command-line>`__ tools, which may be used when solving chemistry
problems. Both can load and run an `input
file <qiskit_acqua_chemistry#input-file>`__ specifying the molecule, an
algorithm to be used and its configuration, and various other options to
tailor the experiment.  If you are new to the
library we highly recommend getting started with the GUI.
Finally, QISKIT ACQUA Chemistry can also be accessed `programmatically <#programming>`__.

GUI
~~~

The GUI allows provides an easy means to load and run an input file
specifying your chemistry problem. An input file is created,
edited and saved with validation of parameter values to provide ease of
configuring the chemistry problem using the input file. The ``pip install``
creates a script that allows you to start the GUI from the command line,
as follows:

.. code:: sh

   qiskit_acqua_chemistry_ui

If you clone and run directly from the repository, instead of using ``pip
install``, then it can be run using:

.. code:: sh

   python qiskit_acqua_chemistry/ui``

from the root folder of the ``qiskit-acqua-chemistry`` repository clone.

Command Line
~~~~~~~~~~~~

Here is a summary of the ``qiskit_acqua_chemistry`` command-line options:

.. code:: sh

   usage: qiskit_acqua_chemistry [-h] [-o output | -jo json output] input

   Quantum Chemistry Program.

   positional arguments:
     input            Chemistry Driver input or Algorithm JSON input file

   optional arguments:
     -h, --help       show this help message and exit
     -o output        Algorithm Results Output file name
     -jo json output  Algorithm JSON Output file name

QISKit ACQUA Chemistry, if installed via ``pip install``, comes with the following command-line tool:

.. code:: sh

   qiskit_acqua_chemistry_cmd

If you cloned QISKit ACQUA Chemistry from the repository instead of using ``pip
install``, then the command-line interface can be executed as follows:

.. code:: sh

   python qiskit_acqua_chemistry

from the root folder of the ``qiskit-acqua-chemistry`` repository clone.

Programming
~~~~~~~~~~~

Chemistry experiments can be run programmatically too. Please refer to
the tutorials for a number of examples. Here you
will see different ways of programming an experiment. The simplest,
which matches closely to the input file, is used in many examples. Here,
a Python dictionary is passed as an input to an ``ACQUAChemistry`` instance to
run the experiment and return the result.

.. code:: python

   solver = ACQUAChemistry()
   result = solver.run(acqua_chemistry_dict)

The
`acqua_chemistry_howto <https://github.com/QISKit/qiskit-acqua-tutorials/blob/master/chemistry/acqua_chemistry_howto.ipynb>`__
notebook details this simple example.

Creating the Python dictionary for a programmatic experiment without the risk
of typos, mismatching parameters, or parameter values out of range or of the wrong type
can be a challenge.  QISKit ACQUA Chemistry dramatically simplifies the
generation of a correct Python dictionary of a chemistry problem.  Users can first
configure a chemistry problem by using the `GUI <#gui>`__, extract the corresponding
Python dictionary through the GUI utilities, embed that dictionary in
a Python program, and programmatically customize the dictionary according to their needs.

Since a Python dictionary can be updated programmatically, it is
possible to carry out more complicated experiments, such as plotting a
`dissociation curve 
<https://github.com/QISKit/qiskit-acqua-tutorials/blob/master/chemistry/lih_dissoc.ipynb>`__
or `comparing results obtained with different algorithms 
<https://github.com/QISKit/qiskit-acqua-tutorials/blob/master/chemistry/lih_uccsd.ipynb>`__.