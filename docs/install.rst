Installation and Setup
=====================

Dependencies
------------

As QISKit ACQUA is built upon `QISKit <https://qiskit.org>`__, you are encouraged to look over the
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

We encourage you to install QISKit ACQUA via the `pip <https://pip.pypa.io/en/stable/>`__  tool (a Python
package manager):

.. code:: sh

    pip install qiskit_acqua

pip will handle all dependencies automatically and you will always
install the latest (and well-tested) release version.

We recommend using Python virtual environments to improve your
experience.

Running an Algorithm
--------------------

Now that you have installed QISKit ACQUA, you can run an algorithm by invoking it with the appropriate input.
The input to a QISKit ACQUA algorithm is expected to be in `JSON <http://json.org>`__ format.
This can be done programmaticall, via the Graphical User Interface (GUI) or from the command line.  In addition to the input itself,
the JSON file encodes the algorithm that QISKit ACQUA will invoke on that input.
One way to generate the JSON input is by
serializing the input to QISKit ACQUA when executing one of the applications running on top of QISKit ACQUA,
such as QISKit ACQUA Chemistry, QISKit ACQUA Artificial Intelligence, and QISKit ACQUA Optimization.

GUI
~~~
The QISKit ACQUA GUI allows you to load and save a JSON file to run an algorithm
as well as create a new one or edit an existing one. So, for example,
using the UI, you can alter the parameters of an  algorithm and/or its dependent
objects to see how the changes affect the outcome. pip installs a
small script that allows you to start the GUI from the command line, as follows:

.. code:: sh

    qiskit_acqua_ui

If you clone and run directly from the repository instead of using
the ``pip install`` recommended way, then it can be run using

.. code::
 
   python qiskit_acqua/ui/run

from the root folder of the ``qiskit-acqua`` repository clone.

Configuring an experiment that involves both quantum-computing and domain-specific parameters
may look like a challenging activity, which requires specialized knowledge on both the specific
domain in which the experiment runs and quantum computing itslf.  QISKit ACQUA simplifies the
configuration of any run in two ways:

1. Defaults are provided for each parameter.  Such defaults have been validated to be the best choices in most cases.

2. A robust configuration correctness enforcement mechanism is in place.  Any configuration is validated by QISKit ACQUA upon startup, and if the user has chosen to use the GUI to configure an experiment, the GUI itself prevents incompatible parameters from being selected, making the configuration error resilient.

Command Line
~~~~~~~~~~~~

The command line tool will run an algorithm from the supplied JSON file.
Run without any arguments, it will print help ibformation.  pip installs a
small script, which can be invoked with a JSON algorithm input file from the command line as follows:

.. code:: sh

    qiskit_acqua_cmd examples/H2-0.735.json

If you clone and run direct from the repository instead of using
the ``pip install`` recommended way then it can be run using

.. code:: sh

    python qiskit_acqua

from the root folder of the ``qiskit-acqua``
repository clone.

Browser
~~~~~~~

As QISKit ACQUA is extensible with pluggable components, we have provided
a documentation GUI that shows all the pluggable components along with the schema for
their parameters. ``pip`` installsa small script to invoke the
browser GUI as follows:

.. code:: sh

    qiskit_acqua_browser

Note: if you clone the repository and want to start the documentation GUI
directly from your local repository instead of using
the ``pip install`` recommended way, then the documentation GUI can be run using the following command:

.. code:: sh

    python qiskit_acqua/ui/browser

from the root folder of the
``qiskit-acqua`` repository clone.

Programming
~~~~~~~~~~~

Any algoirithm in QISJit ACQUA can be run programmatically too. The
``examples`` folder contains numerous cases that explain  how to do this. Here you will
see there is a ``run_algorithm`` method used, which takes either the JSON algorithm input
or an equivalent Python dictionary and optional ``AlgorithmInput`` object
for the algorithm. There is also a ``run_algorithm_to_json`` that simply
takes the input and saves it to JSON in a self-contained form, which  can
later be used by the command line or GUI.


