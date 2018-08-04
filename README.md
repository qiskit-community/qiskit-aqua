# Qiskit Aqua Chemistry

Qiskit Aqua Chemistry is a set of tools, algorithms and software for use with quantum computers
to carry out research and investigate how to take advantage of the quantum computational power to solve chemistry
problems. Qiskit Aqua Chemistry translates chemistry-specific problem inputs into inputs for a quantum algorithm
supplied by [Qiskit Aqua](https://github.com/Qiskit/aqua), which then in turn uses
[Qiskit Terra](https://www.qiskit.org/terra) for the actual quantum computation.

Qiskit Aqua Chemistry allows users with different levels of experience to execute chemistry experiments and
contribute to the software stack.  Users with pure chemistry background can continue to configure chemistry
problems according to their favorite software packages, called *drivers*.  These users do not need to learn the
details of quantum computing; Qiskit Aqua Chemistry translates any chemistry program configuration entered by
any end user in their favorite driver into quantum-specific input.

You can follow the [installation](#installation) instructions to install this software and its dependencies.

Once you have it installed, you can experiment with Aqua Chemistry using either the supplied [GUI](#gui) or
[command line](#command-line) tools.

More advanced users and developers may wish to develop and add their own
algorithms or other code. Algorithms and supporting components may be added to
[Qiskit Aqua](https://github.com/Qiskit/aqua) which was designed with an extensible, pluggable
framework. Qiskit Aqua Chemistry utilizes a similar framework for drivers and the core computation.

**If you'd like to contribute to Qiskit Aqua Chemistry, please take a look at our**
[contribution guidelines](.github/CONTRIBUTING.rst).

Links to Sections:

* [Installation](#installation)
* [Running a chemistry experiment](#running-a-chemistry-experiment)
* [Authors](#authors-alphabetical)
* [License](#license)

## Installation

### Dependencies

As Qiskit Aqua Chemistry is built upon Qiskit Aqua you are encouraged to look over the
[Qiskit Aqua installation](https://github.com/Qiskit/aqua/blob/master/README.md#installation) too.

Like Qiskit Aqua at least [Python 3.5 or later](https://www.python.org/downloads/) is needed to use
Qiskit Aqua Chemistry.
In addition, [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) is recommended
for interacting with the tutorials.
For this reason we recommend installing the [Anaconda 3](https://www.continuum.io/downloads)
Python distribution, as it comes with all of these dependencies pre-installed.

### Installation

We encourage you to install Qiskit Aqua Chemistry via pip, a Python package manager:

```
pip install qiskit-aqua-chemistry
```

pip will handle all dependencies automatically and you will always install the latest (and well-tested)
release version.

We recommend using Python virtual environments to cleanly separate the installation of Terra, Aqua and Aqua Chemistry
from other programs and improve your experience.

#### Chemistry drivers

To run chemistry experiments on molecules, you will also need to install a supported chemistry program or library. 
Several so-called chemistry drivers are supported and while logic to
interface these external libraries and programs is supplied, by the above pip install, the dependent chemistry library
or program needs to be installed separately. The following chemistry drivers are supported:

1. [Gaussian 16](http://gaussian.com/gaussian16/), a commercial chemistry program
2. [PSI4](http://www.psicode.org/), an open-source chemistry program built on Python
3. [PySCF](https://github.com/sunqm/pyscf), an open-source Python chemistry program
4. [PyQuante](https://github.com/rpmuller/pyquante2), a pure cross-platform open-source Python chemistry program

Please refer to the Aqua Chemistry drivers installation instructions in the [Aqua documentation](https://qiskit.org/documentation/aqua/).

Even without installing one of the above drivers, it is still possible to run some chemistry experiments if
you have an Aqua Chemistry HDF5 file that has been previously created when using one of the above drivers.
The HDF5 driver takes such an input.  
A few sample hdf5 files have been provided and these can be found in the 
Qiskit Aqua Tutorials GitHub repository's [chemistry folder](https://github.com/Qiskit/aqua-tutorials/tree/master/chemistry).

## Running a chemistry experiment

Now that you have installed Aqua Chemistry you can run an experiment, for example to compute the ground
state energy of a molecule.

Aqua Chemistry has both [GUI](#gui) and [command line](#command-line) tools, which may be used when conducting
chemistry simulation experiments on a quantum machine. Both can load and run an [input file](qiskit_aqua_chemistry#input-file) specifying the molecule,
an algorithm to be used and its configuration, and various other options to tailor the experiment. You can find several
input files to experiment with in the Aqua Chemistry Git Hub repository's [chemistry folder]
(https://github.com/Qiskit/aqua-tutorials/tree/master/chemistry/input_files).
If you are new to the library we highly recommend getting started with the GUI.

### GUI

The GUI provides an easy means to load and run an input file specifying your chemistry problem. An input file
can also be created, edited and saved with validation of values to provide ease of configuring the chemistry problem
using the input file. The pip installation creates a script that allows you to start the GUI from the
command line, as follows:

`qiskit_aqua_chemistry_ui`

If you clone and run directly from the repository, instead of using
pip install, then it can be run using:

`python qiskit_aqua_chemistry/ui`

from the root folder of the aqua-chemistry repository clone.

### Command line

Summary of qiskit_aqua_chemistry command line options:

`qiskit_aqua_chemistry_cmd`:
```
usage: qiskit_aqua_chemistry [-h] [-o output | -jo json output] input

Quantum Chemistry Program.

positional arguments:
  input            Chemistry Driver input or Algorithm JSON input file

optional arguments:
  -h, --help       show this help message and exit
  -o output        Algorithm Results Output file name
  -jo json output  Algorithm JSON Output file name
```

If you clone and run directly from the repository, instead of using
pip install, then it can be run using

`python qiskit_aqua_chemistry`

from the root folder of the aqua-chemistry repository clone.

### Programming

Chemistry experiments can be run programmatically too. Please refer to the
Aqua Chemistry's [chemistry folder](https://github.com/Qiskit/aqua-tutorials/tree/master/chemistry)
for a number of examples. Here you will see different ways of programming an experiment. The simplest, which
matches closely to the input file, is used in many examples. Here a similar Python dictionary, which can
be automatically generated from the GUI, is used and an
`AquaChemistry` instance is used to run the experiment and return the result.
```
solver = AquaChemistry()
result = solver.run(aqua_chemistry_dict)
```
The [aqua_chemistry_howto](https://github.com/Qiskit/aqua-tutorials/blob/master/chemistry/aqua_chemistry_howto.ipynb)
notebook details this simple example.

Since the Python dictionary can be updated programmatically it is possible to carry out more complicated experiments
such as plotting a
[dissociation curve](https://github.com/Qiskit/aqua-tutorials/blob/master/chemistry/lih_uccsd.ipynb).


## Authors

Qiskit Aqua Chemistry was inspired, authored and brought about by the collective
work of a team of researchers.

Qiskit Aqua Chemistry continues now to grow with the help and work of [many people](CONTRIBUTORS.rst) who contribute
to the project at different levels.

## License

This project uses the [Apache License Version 2.0 software license](https://www.apache.org/licenses/LICENSE-2.0).

Some code supplied here for [drivers](qiskit_aqua_chemistry/drivers/README.md), for interfacing to external chemistry
programs/libraries, has additional licensing.

* The [Gaussian 16 driver](qiskit_aqua_chemistry/drivers/gaussiand/README.md) contains work licensed under the
[Gaussian Open-Source Public License](qiskit_aqua_chemistry/drivers/gaussiand/gauopen/LICENSE.txt).

* The [Pyquante driver](qiskit_aqua_chemistry/drivers/pyquanted/README.md) contains work licensed under the
[modified BSD license](qiskit_aqua_chemistry/drivers/pyquanted/LICENSE.txt).
