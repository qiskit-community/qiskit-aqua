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
use case. [See more](#generate-test-objects)

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
  * Matrix and vector can be generated dynamically using
      `generate_test_object`. Preferred way!
So parameters are for instance
```Python
params = <Normal HHL params>
# numpy array
params["input"] = {
  "vector": np.array([1, 1])
  "matrix": np.array([[1, 0], [0, 2]])
}

# index, test_objects/mat_default.pkl and test_objects/vec_default.pkl exists
params["input"] = {
  "test_set": "default",
  "matrix": 0,
  "vector": 1
}

# generate_test_object
params["input"] = {
  "type": "generate",
  "test_set": "generated_set",
  "n": 2,
  "condition": 5
}
```
Executing the same parameters twice will load the first result at the second
execution.

### Run tests for a parameter grid

The method `run_tests` takes in a special params dict that specifies a range of
parameters that should be tested. `run_tests` has following optional arguments:
  * `force` boolean value that determines whether existing results will be
      overridden.
  * `status` function which takes in the result of an execution an returns
      a string, which then will be printed
  * `interest` filters the result for the interest and only keeps the
      'interesting' things in memory. Can be string (key for result dict) or
      list of strings (for more than one interest). Can also be callable or
      list of callables.

#### The input parameters
The input parameters have basically the same form as the ones for `run_test`,
however, since it is now possible to run over a range of parameters, one can
also input a list (or tuple or any iterable) of parameters to one parameter.
For instance
```Python
"num_ancillae": (6, 7, 8, 9)
```
runs the rest of the parameters four times with 6, 7, 8 and 9 ancillary
qubits.

If more than one parameter is tested, all possible combinations of parameters
occur in the `run_tests` method, so the number of executions multiplies.

Here is an full example to test `num_ancillae`, `num_time_slices` and matrix
`condition`:
```Python
params = {
    "algorithm": {
        "name": "HHL",
        "mode": "state_tomography"
    },
    "eigs": {
        "name": "QPE",
        "num_time_slices": [20, 50, 80],
        "expansion_mode": "suzuki",
        "expansion_order": 2,
        "negative_evals": (True, False),
        "num_ancillae": (5, 6, 7, 8, 9)
    },
    "reciprocal": {
        "name": "LOOKUP",
        "lambda_min": 0.9,
        "pat_length": 4
    },
    "backend": {
        "name": "local_qasm_simulator",
        "shots": 1
    },
    "input": {
        "type": "generate",
        "n": 2,
        "test_set": "general_condition",
        "condition": range(10, 110, 20)
    }
}
```
#### Output Format

The output format is a list of ('interesting') results if only one parameter is
tested. Otherwise it is a dict of following format:
```Python
{
  "keys": [ 'list of tested parameter (format: "module key")' ],
  "vals": [ 'list of all possible parameter combinations' ],
  "data": [ 'list of all results corresponding to the vals' ]
}
```

In the case of the example from input parameters it will be
```Python
{
  "keys": [
    "eigs num_time_slices",
    "eigs num_ancillae",
    "input condition"
  ],
  "vals": [
    [20, 5, 10],
    [20, 5, 30], 
    [20, 5, 50], 
    ...
    [80, 9, 90]
  ]
  "data": [
    result([20, 5, 10]),
    result([20, 5, 30]), 
    ...
    result([80, 9, 90])
  ]
}
```

### Conditioned parameter runs
Sometimes the values one parameter can be in are determined by the values of
the other parameters, therefore one can conditionalize the parameter run
values by using functions or dictionarys as parameter values.

For instance the `eigs expansion_order` only makes sense if `eigs
expansion_mode` = 'trotter'. Such a definition for testing different expansion
orders and modes would look like
```Python
{
  ...
  "eigs": {
    ...
    "expansion_mode": ["trotter": "suzuki"],
    "expansion_order": lambda x: 1 if x["eigs"]["expansion_mode"] == "trotter" else  (1, 2, 3),
    or
    "expansion_order": {
      "keys": ["eigs expansion_mode"],
      "vals": ["suzuki", "trotter"],
      "data": [(1, 2, 3), 1]
    }
    ...
  }
  ...
}
```
Conditioned parameters can also depend on other conditioned parameters, but
only if they are defined above in the dictionary.

The returned object has the same form as the normal parameter grid run.

### Retrieving data from returned object

The method `get_for` retrieves data of specific interest from the return
object. It has the following arguments
  * `params` a dictionary for which specifies the interest
  * `data` the result object

For instance, the user has results for a grid over `num_ancillae` and
`num_time_slices` but is now only interested in`num_ancillae` = 6, then the
syntax for doing that would look like this
```Python
result = run_tests(params)
time_slice_result = get_for({"eigs num_ancillae": 6}, result)
```
If only one free parameter is left, a list of results is returned, otherwise
another result object is returned.

## Generate Test Objects
Generate test objects is a module to generate random test objects with few
control parameters and reuse the generated ones in further queries. The method
`generate_input` generates an input dict with numpy matrix und vector. The method
is triggered in `run_test` if `"type" = "generate"` specified. The default
additional parameters for the random matrix are
```Python
default_params = {
    "type": "generate",
    "test_set": "specified",
    "condition": 2,
    "n": 2,
    "lambda_min": 1,
    "repetition": 0,
    "hermitian": True,
    "negative_evals": True
}
```
See `qiskit_aqua/utils/random_matrix_generator.py` for more details of the
matrix generation. The attributes `negative_evals` and `hermitian` will be
fetched from the `eigs` parameters (if set). The repetition index is just an
integer which allows the generation of a new matrix with same parameters stored
separately. 

#### Generating the vector

The default input vector is the equal sum of all eigenvalues of the corresponding
matrix. The weight of each eigenvector can be defined by
```Python
"vector": np.array([u_1, ..., u_n]),
```
needs to be `np.array`.

