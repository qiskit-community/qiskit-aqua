# QISKit AQUA

QISKit Algorithms and Circuits for QUantum Applications (QISKit AQUA) is a library of algorithms for quantum computing
that uses [QISKit](https://qiskit.org/) to build out and run quantum circuits.

QISKit AQUA provides a library of cross-domain algorithms upon which domain-specific applications and stacks can be
built. At the time of writing, [QISKit AQUA Chemistry](https://github.com/QISKit/aqua-chemistry) has
been created to utilize QISKit AQUA for quantum chemistry computations. QISKIt AQUA is also showcased for other
domains with both code and notebook examples. Please see
[QISKit AQUA Optimization](https://github.com/QISKit/aqua-tutorials/tree/master/optimization) and
[QISKit AQUA Artificial Intelligence](https://github.com/QISKit/aqua-tutorials/tree/master/artificial_intelligence).

QISKit AQUA was designed to be extensible, and uses a pluggable framework where algorithms and support objects used
by algorithms, e.g optimizers, variational forms, oracles etc., are derived from a defined base class for the type and
discovered dynamically at runtime.

**If you'd like to contribute to QISKit AQUA, please take a look at our**
[contribution guidelines](.github/CONTRIBUTING.rst).

Links to Sections:

* [Installation](#installation)
* [Running an Algorithm](#running-an-algorithm)
* [Authors](#authors)
* [License](#license)

## Installation

### Dependencies

As QISKit AQUA is built upon [QISKit](https://qiskit.org), you are encouraged to look over the
[QISKit installation](https://github.com/QISKit/qiskit-core/blob/master/README.md#installation)
too.

Like for QISKit, at least [Python 3.5 or later](https://www.python.org/downloads/) is needed to use QISKit AQUA.
In addition, [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) is recommended for interacting
with the tutorials. For this reason we recommend installing the [Anaconda 3](https://www.continuum.io/downloads)
Python distribution, as it comes with all of these dependencies pre-installed.

### Getting the Code

We encourage you to install QISKit AQUA via the [pip](https://pip.pypa.io/en/stable/) tool (a Python package manager):

```
pip install qiskit-aqua
```

pip will handle all dependencies automatically and you will always install the latest (and well-tested) release version.

We recommend using Python virtual environments to improve your experience.

### Running an Algorithm

Now that you have installed QISKit AQUA you can run an algorithm. This can be done [programmatically](#programming)
or can be done using JSON as an input. Whether via dictionary or via JSON the input is validated for correctness against
schemas. 
 
JSON is convenient when the algorithm input has been saved in this form from a prior run. A file containing a saved
JSON input can be given to either the [GUI](#gui) or the [command line](#command-line) tool in order to run
the algorithm.
 
One simple way to generate such JSON input is by serializing the input to QISKit AQUA when executing one of the
applications running on top of QISKit AQUA, such as QISKit AQUA Chemistry, QISKit AQUA Artificial Intelligence
or QISKit AQUA Optimization. The GUI also saves any entered configuration in JSON 

The [algorithms](aqua/README.md) readme contains detailed information on the various parameters for each
algorithm along with links to the respective components they use.
 

### GUI

The QISKit AQUA GUI allows you to load and save a JSON file to run an algorithm, as well as create a new one or edit
an existing one. So, for example, using the UI, you can alter the parameters of an algorithm and/or its dependent
objects to see how the changes affect the outcome. The pip install creates a script that allows you to start the GUI
from the command line, as follows:

```
qiskit_aqua_ui
```

If you clone and run directly from the repository instead of using the `pip install` recommended way, then the GUI can
be run, from the root folder of the `qiskit-aqua` repository clone, using

```
python qiskit_aqua/ui/run
```

Configuring an experiment that involves both quantum-computing and domain-specific parameters may look like a 
challenging activity, which requires specialized knowledge on both the specific domain in which the experiment runs and
quantum computing itself. QISKit AQUA simplifies the configuration of any run in two ways:

1.  Defaults are provided for each parameter. Such defaults have been validated to be the best choices in most cases.
2.  Robust configuration correctness enforcement mechanisms are in place. The input parameters are always schema
    validated by QISKit AQUA when attempting to run an algorithm. When using the GUI to configure an experiment,
    the GUI itself prevents incompatible parameters from being selected.

### Command Line

The command line tool will run an algorithm from the supplied JSON file. Run without any arguments, it will print help
information. The pip install creates a script, which can be invoked with a JSON algorithm input file from the command
line, for example as follows:

```
qiskit_aqua_cmd examples/H2-0.735.json
```

If you clone and run direct from the repository instead of using the `pip install` recommended way then it can be
run, from the root folder of the `qiskit-aqua` repository clone, using

```
python qiskit_aqua
```

### Browser

As QISKit AQUA is extensible with pluggable components, we have provided a documentation GUI that shows all the
pluggable components along with the schema for their parameters. The pip install creates a script to invoke the
browser GUI as follows:

```
qiskit_aqua_browser
```

Note: if you clone the repository and want to start the documentation GUI directly from your local repository instead
of using the `pip install` recommended way, then the documentation GUI can be run, from the root folder of the
`qiskit-aqua` repository clone, using the following command:

```
python qiskit_aqua/ui/browser
```

### Programming

Any algorithm in QISKit AQUA can be run programmatically too. The aqua folder in the [aqua-tutorials](https://github.com/QISKit/aqua-tutorials/tree/master/aqua) contains numerous
samples that demonstrate how to do this. Here you will see there is a `run_algorithm` method used, which takes either
the JSON algorithm input or an equivalent Python dictionary and optional `AlgorithmInput` object for the algorithm.
There is also a `run_algorithm_to_json` that simply takes the input and saves it to JSON in a self-contained form,
which can later be used by the command line or GUI.

## Authors

QISKit AQUA was inspired, authored and brought about by the collective work of a team of researchers.

QISKit AQUA continues now to grow with the help and work of [many people](./CONTRIBUTORS.md), who contribute
to the project at different levels.

## License

This project uses the [Apache License Version 2.0 software license](https://www.apache.org/licenses/LICENSE-2.0).
