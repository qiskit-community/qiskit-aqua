# Qiskit AQUA Chemistry

Qiskit AQUA Chemistry is a set of tools, algorithms and software for use with quantum computers
to carry out research and investigate how to take advantage of quantum computing power to solve chemistry
problems. Qiskit AQUA Chemistry translates chemistry-specific problem inputs into inputs for a quantum algorithm
supplied by [Qiskit AQUA](https://github.com/Qiskit/aqua), which then in turn uses
[Qiskit](https://www.qiskit.org/) for the actual quantum computation.

Qiskit AQUA Chemistry allows users with different levels of experience to execute chemistry experiments and
contribute to the software stack.  Users with pure chemistry background can continue to configure chemistry
problems according to their favorite software packages, called *drivers*.  These users do not need to learn the
details of quantum computing; Qiskit AQUA Chemistry translates any chemistry program configuration entered by
any end user in their favorite driver into quantum-specific input.

You can follow the [installation](#installation) instructions to install this software and its dependencies.

Once you have it installed you can experiment with the library using either the supplied [GUI](#gui) or
[command line](#command-line) tools.

More advanced users and [developers](qiskit_aqua_chemistry#developers) may wish to develop and add their own
algorithms or other code. Algorithms and supporting components may be added to
[Qiskit AQUA](https://github.com/Qiskit/aqua) which was designed with an extensible, pluggable
framework. Qiskit AQUA Chemistry utilizes a similar framework for drivers and the core computation.

**If you'd like to contribute to Qiskit AQUA Chemistry, please take a look at our**
[contribution guidelines](.github/CONTRIBUTING.rst).

Links to Sections:

* [Installation](#installation)
* [Running a chemistry experiment](#running-a-chemistry-experiment)
* [Authors](#authors-alphabetical)
* [License](#license)

## Installation

### Dependencies

As Qiskit AQUA Chemistry is built upon Qiskit AQUA you are encouraged to look over the
[Qiskit AQUA installation](https://github.com/Qiskit/aqua/blob/master/README.md#installation) too.

Like Qiskit AQUA at least [Python 3.5 or later](https://www.python.org/downloads/) is needed to use
Qiskit AQUA Chemistry.
In addition, [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) is recommended
for interacting with the tutorials.
For this reason we recommend installing the [Anaconda 3](https://www.continuum.io/downloads)
Python distribution, as it comes with all of these dependencies pre-installed.

### Installation

We encourage you to install Qiskit AQUA Chemistry via the PIP tool (a Python package manager):

```
pip install qiskit-aqua-chemistry
```

PIP will handle all dependencies automatically and you will always install the latest (and well-tested)
release version.

We recommend using Python virtual environments to improve your experience.

#### Chemistry drivers

To run chemistry experiments on molecules you will also need to install a supported chemistry program or library. 
Several so-called chemistry [drivers](qiskit_aqua_chemistry/drivers/README.md) are supported and while logic to
interface these external libraries and programs is supplied, by the above pip install, the dependent chemistry library
or program needs to be installed separately. The following chemistry drivers are supported, please refer to these for
more instructions on their specific installation: 

* [Gaussian](qiskit_aqua_chemistry/drivers/gaussiand/README.md): A commercial chemistry program  
* [PyQuante](qiskit_aqua_chemistry/drivers/pyquanted/README.md): An open-source Python library, with a pure Python version usable cross-platform
* [PySCF](qiskit_aqua_chemistry/drivers/pyscfd/README.md): An open-source Python library 
* [PSI4](qiskit_aqua_chemistry/drivers/psi4d/README.md): An open-source chemistry program built on Python

However, even without installing one of the above drivers, it is still possible to run some chemistry experiments if
you have a Qiskit AQUA Chemistry HDF5 file that has been previously created when using one of the above drivers.
The HDF5 driver takes such an input. 

* [HDF5](qiskit_aqua_chemistry/drivers/hdf5d/README.md): Driver for Qiskit AQUA Chemistry hdf5 files    
 
A few sample hdf5 files have been provided and these can be found in the 
Qiskit AQUA Tutorial's [chemistry folder](https://github.com/Qiskit/aqua-tutorials/tree/master/chemistry).  

## Running a chemistry experiment

Now that you have installed Qiskit AQUA Chemistry you can run an experiment, for example to compute the ground
state energy of a molecule.

Qiskit AQUA Chemistry has both [GUI](#gui) and [command line](#command-line) tools which may be used when solving
chemistry problems. Both can load and run an [input file](qiskit_aqua_chemistry#input-file) specifying the molecule,
an algorithm to be used and its configuration, and various other options to tailor the experiment. You can find several
input files in the chemistry folder of
[aqua-tutorials](https://github.com/Qiskit/aqua-tutorials/tree/master/chemistry/input_files)
to experiment with. If you are new to the library we highly recommend getting started with the GUI.

### GUI

The GUI allows provides an easy means to load and run an input file specifying your chemistry problem. An input file
can also be created, edited and saved with validation of values to provide ease of configuring the chemistry problem
using the input file. The pip install creates a script that allows you to start the GUI from the
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

Chemistry experiments can be run programmatically too. Please refer to the chemistry folder of
[aqua-tutorials](https://github.com/Qiskit/aqua-tutorials/tree/master/chemistry)
for a number of examples. Here you will see different ways of programming an experiment. The simplest, which
matches closely to the input file, is used in many examples. Here a similar Python dictionary is used and an
AQUAChemistry instance is used to run the experiment and return the result.
```
solver = AQUAChemistry()
result = solver.run(aqua_chemistry_dict)
```
The [aqua_chemistry_howto](https://github.com/Qiskit/aqua-tutorials/blob/master/chemistry/aqua_chemistry_howto.ipynb)
notebook details this simple example.

Since the Python dictionary can be updated programmatically it is possible to carry out more complicated experiments
such as plotting a
[disocciation curve](https://github.com/Qiskit/aqua-tutorials/blob/master/chemistry/lih_uccsd.ipynb)


## Authors

Qiskit AQUA Chemistry was inspired, authored and brought about by the collective
work of a team of researchers.

Qiskit AQUA Chemistry continues now to grow with the help and work of [many people](CONTRIBUTORS.md) who contribute
to the project at different levels.

## License

This project uses the [Apache License Version 2.0 software license](https://www.apache.org/licenses/LICENSE-2.0).

Some code supplied here for [drivers](qiskit_aqua_chemistry/drivers/README.md), for interfacing to external chemistry
programs/libraries, has additional licensing.

* The [Gaussian 16 driver](qiskit_aqua_chemistry/drivers/gaussiand/README.md) contains work licensed under the
[Gaussian Open-Source Public License](qiskit_aqua_chemistry/drivers/gaussiand/gauopen/LICENSE.txt).

* The [Pyquante driver](qiskit_aqua_chemistry/drivers/pyquanted/README.md) contains work licensed under the
[modified BSD license](qiskit_aqua_chemistry/drivers/pyquanted/LICENSE.txt).
