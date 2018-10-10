# Modules for testing the HHL algorithm

## General

### HHL Test suite overview

The `hhl_test_suite.py` module enables the user to test the HHL algorithm. The
result of each test run with specific parameters will be stored in `data/raw/`
and the corresponding parameters are stored in `data/info/` with the same
filename. The filename is composed through hashing the parameters, which
generates an unique key for each parameter set. When a test run is started and
the parameter hash is already saved, the results are automatically loaded.
Furthermore the module enables to run over a grid of parameters. [See
more](#hhl-test-suite)

### Generate Test Objects overview

The module `generate_test_object.py` allows the user to specify a few parameters for
generating a random matrix and vector. The matrices and vectors are
automatically stored in `test_objects/` to reuse them when the same parameters
are specified for generation. However, a specific `test_set` can be defined to
generate different random matrices with same parameters for a different
usecase. [See more](#generate-test-objects)

## HHL Test suite

### Test single case

The method `run_test` takes one required parameter, the parameters for the HHL
algorithm. Additionally `force` can be set to `True`, which overrides existing
results of this parameter set. The parameters differ in the input section
a little bit from the ones required for the HHL algorithm. Here are three
different input types valid:
  * Matrix or vector can be directly defined by a numpy array.
  * Matrix or vector can be defined by an index in an pickeled list of
      matrices/vectors located in `test_objects/mat_{test_set}.pkl` and
      `test_object/vec_{test_set}.pkl`.
  * Matrix and vector can be generated dynamically using `generate_test_object`


## Generate Test Objects
