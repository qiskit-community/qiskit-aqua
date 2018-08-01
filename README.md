<<<<<<< HEAD
<<<<<<< HEAD
# Qiskit Aqua

Qiskit Algorithms for QUantum Applications (Qiskit Aqua) is a library of algorithms for quantum computing
that uses [Qiskit](https://qiskit.org/) to build out and run quantum circuits.

Qiskit Aqua provides a library of cross-domain algorithms upon which domain-specific applications and stacks can be
built. At the time of writing, [Qiskit Aqua Chemistry](https://github.com/Qiskit/aqua-chemistry) has
been created to utilize Qiskit Aqua for quantum chemistry computations. QISKIt Aqua is also showcased for other
domains with both code and notebook examples. Please see
[Qiskit Aqua Optimization](https://github.com/Qiskit/aqua-tutorials/tree/master/optimization) and
[Qiskit Aqua Artificial Intelligence](https://github.com/Qiskit/aqua-tutorials/tree/master/artificial_intelligence).

Qiskit Aqua was designed to be extensible, and uses a pluggable framework where algorithms and support objects used
by algorithms, e.g optimizers, variational forms, oracles etc., are derived from a defined base class for the type and
discovered dynamically at runtime.

**If you'd like to contribute to Qiskit Aqua, please take a look at our**
=======
# QISKit ACQUA
=======
# Qiskit Aqua
>>>>>>> Update README.md

Qiskit Algorithms for Quantum Applications (Qiskit Aqua) is a library of algorithms for quantum computing
that uses [Qiskit Terra](https://qiskit.org/terra) to build out, compile and run quantum circuits.

Aqua provides a library of cross-domain algorithms upon which domain-specific applications can be
built. At the time of writing, [Aqua Chemistry](https://github.com/Qiskit/aqua-chemistry) has
been created to utilize Aqua for quantum chemistry computations. Aqua is also showcased for other
domains with both code and notebook examples, such as
[Aqua Optimization](https://github.com/Qiskit/aqua-tutorials/tree/master/optimization) and
[Aqua Artificial Intelligence](https://github.com/Qiskit/aqua-tutorials/tree/master/artificial_intelligence).

QISKit ACQUA was designed to be extensible, and uses a pluggable framework where algorithms and support objects used
by algorithms, such as optimizers, variational forms, and oracles, are derived from a defined base class for the type and
discovered dynamically at run time.

<<<<<<< HEAD
**If you'd like to contribute to QISKit ACQUA, please take a look at our**
>>>>>>> Add files via upload
=======
**If you'd like to contribute to Aqua, please take a look at our**
>>>>>>> Update README.md
[contribution guidelines](.github/CONTRIBUTING.rst).

Links to Sections:

* [Installation](#installation)
* [Running an Algorithm](#running-an-algorithm)
* [Authors](#authors)
<<<<<<< HEAD
=======
* [License](#license)
>>>>>>> Add files via upload

## Installation

### Dependencies

<<<<<<< HEAD
<<<<<<< HEAD
As Qiskit Aqua is built upon [Qiskit](https://qiskit.org), you are encouraged to look over the
[Qiskit installation](https://github.com/Qiskit/qiskit-core/blob/master/README.md#installation)
too.

Like for Qiskit, at least [Python 3.5 or later](https://www.python.org/downloads/) is needed to use Qiskit Aqua.
=======
As QISKit ACQUA is built upon [QISKit](https://qiskit.org), you are encouraged to look over the
[QISKit installation](https://github.com/QISKit/qiskit-core/blob/master/README.md#installation)
too.

Like for QISKit, at least [Python 3.5 or later](https://www.python.org/downloads/) is needed to use QISKit ACQUA.
>>>>>>> Add files via upload
=======
Aqua is built upon [Qiskit Terra](https://qiskit.org/terra).  Therefore, you are encouraged to look over the
[Qiskit Terra installation instructions](https://github.com/Qiskit/qiskit-terra/blob/master/README.md#installation)
too.

Just like for Terra, at least [Python 3.5 or later](https://www.python.org/downloads/) is needed to use Aqua.
>>>>>>> Update README.md
In addition, [Jupyter Notebook](https://jupyter.readthedocs.io/en/latest/install.html) is recommended for interacting
with the tutorials. For this reason we recommend installing the [Anaconda 3](https://www.continuum.io/downloads)
Python distribution, as it comes with all of these dependencies pre-installed.

### Getting the Code

<<<<<<< HEAD
<<<<<<< HEAD
We encourage you to install Qiskit Aqua via the [pip](https://pip.pypa.io/en/stable/) tool (a Python package manager):

```
pip install qiskit-aqua
=======
We encourage you to install QISKit ACQUA via the [pip](https://pip.pypa.io/en/stable/) tool (a Python package manager):

```
pip install qiskit-acqua
>>>>>>> Add files via upload
=======
We encourage you to install Aqua via the [pip](https://pip.pypa.io/en/stable/) Python package management tool:

```
pip install qiskit-aqua
>>>>>>> Update README.md
```

pip will handle all dependencies automatically and you will always install the latest (and well-tested) release version.

If, however, your goal is not to use Aqua has a tool, but rather to contribute new components to Aqua, then we recommend 
cloning this repository.  This will give you a more direct access to the code.  In any case, we recommend using Python virtual 
environments to improve your experience.

### Running an Algorithm

<<<<<<< HEAD
<<<<<<< HEAD
Now that you have installed Qiskit Aqua you can run an algorithm. This can be done [programmatically](#programming)
or can be done using JSON as an input. Whether via dictionary or via JSON the input is validated for correctness against
schemas.

JSON is convenient when the algorithm input has been saved in this form from a prior run. A file containing a saved
JSON input can be given to either the [GUI](#gui) or the [command line](#command-line) tool in order to run
the algorithm.

One simple way to generate such JSON input is by serializing the input to Qiskit Aqua when executing one of the
applications running on top of Qiskit Aqua, such as Qiskit Aqua Chemistry, Qiskit Aqua Artificial Intelligence
or Qiskit Aqua Optimization. The GUI also saves any entered configuration in JSON

The [algorithms](aqua/README.md) readme contains detailed information on the various parameters for each
algorithm along with links to the respective components they use.


### GUI

The Qiskit Aqua GUI allows you to load and save a JSON file to run an algorithm, as well as create a new one or edit
=======
Now that you have installed QISKit ACQUA you can run an algorithm. This can be done [programmatically](#programming)
or can be done using JSON as an input. Whether via dictionary or via JSON the input is validated for correctness against
=======
Now that you have installed Aqua, you can execute an algorithm. This can be done [programmatically](#programming) or using a 
[JSON](http://json.org/) file as an input. Whether via dictionary or via JSON, the input is validated for correctness against
>>>>>>> Update README.md
schemas. 
 
JSON is convenient when the algorithm input has been saved in this form from a prior run. A file containing a saved
JSON input can be given to either the [GUI](#gui) or the [command line](#command-line) tool in order to run
the algorithm.
 
One simple way to generate such JSON input is by serializing the input to Aqua when executing one of the
applications running on top of Aqua, such as Aqua Chemistry, Aqua AI or Aqua Optimization. The GUI also saves any entered 
configuration in JSON.

The [documentation](https://qiskit.org/documentation/aqua/) contains detailed information on the various parameters for each
algorithm along with links to the respective components they use.
 

### GUI

<<<<<<< HEAD
The QISKit ACQUA GUI allows you to load and save a JSON file to run an algorithm, as well as create a new one or edit
>>>>>>> Add files via upload
an existing one. So, for example, using the UI, you can alter the parameters of an algorithm and/or its dependent
objects to see how the changes affect the outcome. The pip install creates a script that allows you to start the GUI
from the command line, as follows:

```
<<<<<<< HEAD
qiskit_aqua_ui
```

If you clone and run directly from the repository instead of using the `pip install` recommended way, then the GUI can
be run, from the root folder of the `qiskit-aqua` repository clone, using

```
python qiskit_aqua/ui/run
```

Configuring an experiment that involves both quantum-computing and domain-specific parameters may look like a
challenging activity, which requires specialized knowledge on both the specific domain in which the experiment runs and
quantum computing itself. Qiskit Aqua simplifies the configuration of any run in two ways:

1.  Defaults are provided for each parameter. Such defaults have been validated to be the best choices in most cases.
2.  Robust configuration correctness enforcement mechanisms are in place. The input parameters are always schema
    validated by Qiskit Aqua when attempting to run an algorithm. When using the GUI to configure an experiment,
=======
qiskit_acqua_ui
=======
The Aqua GUI allows you to load and save a JSON file to run an algorithm, as well as create a new one or edit
an existing one. Using the GUI, you can alter the parameters of an algorithm and/or its dependent
objects to see how the changes affect the outcome. If you installed Aqua via the pip tool, a script will be present on
your system allowing you to start the GUI from the command line, as follows:

```
qiskit_aqua_ui
>>>>>>> Update README.md
```

If you installed Aqua by cloning this repository directly, instead of using the pip tool, then the GUI can
be run from the root folder of the qiskit-aqua repository clone, using the following command:

```
python qiskit_aqua/ui/run
```

Configuring an experiment that involves both quantum-computing and domain-specific parameters may look like a 
challenging activity, which requires specialized knowledge on both the specific domain in which the experiment runs and
quantum computing itself. Aqua simplifies the configuration of any run in two ways:

1.  Defaults are provided for each parameter. Such defaults have been validated to be the best choices in most cases.
2.  Robust configuration correctness enforcement mechanisms are in place. The input parameters are always schema
<<<<<<< HEAD
    validated by QISKit ACQUA when attempting to run an algorithm. When using the GUI to configure an experiment,
>>>>>>> Add files via upload
=======
    validated by Aqua when attempting to run an algorithm. When using the GUI to configure an experiment,
>>>>>>> Update README.md
    the GUI itself prevents incompatible parameters from being selected.

### Command Line

The command line tool will run an algorithm from the supplied JSON file. Run without any arguments, it will print help
information. The pip installation creates a script, `qiskit_aqua_cmd`, which can be invoked with a JSON algorithm input file from the command line, for example as follows:

```
<<<<<<< HEAD
<<<<<<< HEAD
qiskit_aqua_cmd examples/H2-0.735.json
```

If you clone and run direct from the repository instead of using the `pip install` recommended way then it can be
run, from the root folder of the `qiskit-aqua` repository clone, using

```
python qiskit_aqua
=======
qiskit_acqua_cmd examples/H2-0.735.json
=======
qiskit_aqua_cmd examples/H2-0.735.json
>>>>>>> Update README.md
```

If you installed Aqua by cloning this repository directly, instead of using the pip tool, then the command line tool can be
run from the root folder of the qiskit-acqua repository clone using the following command:

```
<<<<<<< HEAD
python qiskit_acqua
>>>>>>> Add files via upload
=======
python qiskit_aqua
>>>>>>> Update README.md
```

### Browser

<<<<<<< HEAD
<<<<<<< HEAD
As Qiskit Aqua is extensible with pluggable components, we have provided a documentation GUI that shows all the
=======
As QISKit ACQUA is extensible with pluggable components, we have provided a documentation GUI that shows all the
>>>>>>> Add files via upload
pluggable components along with the schema for their parameters. The pip install creates a script to invoke the
browser GUI as follows:

```
<<<<<<< HEAD
qiskit_aqua_browser
=======
qiskit_acqua_browser
>>>>>>> Add files via upload
```

Note: if you clone the repository and want to start the documentation GUI directly from your local repository instead
of using the `pip install` recommended way, then the documentation GUI can be run, from the root folder of the
<<<<<<< HEAD
`qiskit-aqua` repository clone, using the following command:

```
python qiskit_aqua/ui/browser
=======
`qiskit-acqua` repository clone, using the following command:

```
python qiskit_acqua/ui/browser
>>>>>>> Add files via upload
=======
Since Aqua is extensible with pluggable components, we have provided a documentation GUI that shows all the
pluggable components along with the schema for their parameters. The pip installation creates a script to invoke the
browser GUI as follows:

```
qiskit_aqua_browser
```

If you installed Aqua by cloning this repository directly, instead of using the pip tool, then the documentation GUI can be 
run from the root folder of the qiskit-acqua repository clone using the following command:

```
python qiskit_aqua/ui/browser
>>>>>>> Update README.md
```

### Programming

<<<<<<< HEAD
<<<<<<< HEAD
Any algorithm in Qiskit Aqua can be run programmatically too. The aqua folder in the [aqua-tutorials](https://github.com/Qiskit/aqua-tutorials/tree/master/aqua) contains numerous
=======
Any algorithm in QISKit ACQUA can be run programmatically too. The acqua folder in the [qiskit-acqua-tutorials](https://github.com/QISKit/qiskit-acqua-tutorials/tree/master/acqua) contains numerous
>>>>>>> Add files via upload
samples that demonstrate how to do this. Here you will see there is a `run_algorithm` method used, which takes either
=======
Any algorithm in Aqua can be run programmatically too. The aqua folder in the
[aqua-tutorials GitHub repository](https://github.com/Qiskit/aqua-tutorials/tree/master/acqua) contains numerous
examples that demonstrate how to do this. As you can see, Aqua exposes a `run_algorithm` method, which takes either
>>>>>>> Update README.md
the JSON algorithm input or an equivalent Python dictionary and optional `AlgorithmInput` object for the algorithm.
There is also a `run_algorithm_to_json` method that simply takes the input and saves it to JSON in a self-contained form,
which can later be used by the command line or GUI.

## Authors

<<<<<<< HEAD
Qiskit Aqua was inspired, authored and brought about by the collective work of a team of researchers.

Qiskit Aqua continues now to grow with the help and work of [many people](./CONTRIBUTORS.md), who contribute
to the project at different levels.
=======
QISKit ACQUA was inspired, authored and brought about by the collective work of a team of researchers.

QISKit ACQUA continues now to grow with the help and work of [many people](./docs/CONTRIBUTORS.rst), who contribute
to the project at different levels.
<<<<<<< HEAD

## License

This project uses the [Apache License Version 2.0 software license](https://www.apache.org/licenses/LICENSE-2.0).
>>>>>>> Add files via upload
=======
>>>>>>> Update README.md
