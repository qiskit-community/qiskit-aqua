# Contributing

**We appreciate all kinds of help, so thank you!**

First please read the overall project contributing guidelines. These are
included in the Qiskit documentation here:

https://qiskit.org/documentation/contributing_to_qiskit.html

## Contributing to Qiskit Aqua

In addition to the general guidelines above there are specific details for
contributing to Aqua, these are documented below.

### Project Code Style.

Code in Aqua should conform to PEP8 and style/lint checks are run to validate
this.  Line length must be limited to no more 100 characters. Docstrings
should be written using the Google docstring format.

### Pull request checklist

When submitting a pull request and you feel it is ready for review,
please ensure that:

1. The code follows the _code style_ of this project and successfully
   passes the _unit tests_. Aqua uses [Pylint](https://www.pylint.org) and
   [PEP8](https://www.python.org/dev/peps/pep-0008) style guidelines.
   
   You can run
   ```shell script
   make lint
   make style 
   ```
   from the root of the Aqua repository clone for lint and style conformance checks.
   
   For unit testing please see [Testing](#testing) section below.
   
2. The documentation has been updated accordingly. In particular, if a
   function or class has been modified during the PR, please update the
   *docstring* accordingly.
   
   The documentation will be built/tested using Sphinx and should be free
   from errors and warnings.
   
   You can run
   ```shell script
    make html
   ```
   in the 'docs' folder. You might also like to check the html output
   to see the changes formatted output is as expected. You will find an index.html
   file in docs\_build\html and you can navigate from there.
   
   Please note that a spell check is run in CI, on the docstrings, since the text
   becomes part of the online [API Documentation](https://qiskit.org/documentation/).
   
   You can run `make spell` locally to check spelling though you would need to
   [install pyenchant](https://pyenchant.github.io/pyenchant/install.html) and be using
   hunspell-en-us as is used by the CI. 
   
   For some words, such as names, technical terms, referring to parameters of the method etc., 
   that are not in the en-us dictionary and get flagged as being misspelled, despite being correct,
   there is a [.pylintdict](./.pylintdict) custom word list file, in the root of the Aqua repo,
   where such words can be added, in alphabetic order, as needed.
   
3. If it makes sense for your change that you have added new tests that
   cover the changes and any new function.
   
4. Update the CHANGELOG.md to include added, changed, fixed, removed and
   deprecated entries as appropriate. The PR number should be added too. 

5. Ensure all code, including unit tests, has the copyright header. The copyright
   date will be checked by CI build. The format of the date(s) is _year of creation,
   last year changed_. So for example:
   
   > \# (C) Copyright IBM 2018, 2020.

   If the _year of creation_ is the same as _last year changed_ then only
   one date is needed, for example:

   > \# (C) Copyright IBM 2020.
                                                                                                                                                                                                 
   If code is changed in a file make sure the copyright includes the current year.
   If there is just one date and it's a prior year then add the current year as the 2nd date, 
   otherwise simply change the 2nd date to the current year. The _year of creation_ date is
   never changed.
 
## Installing Qiskit Aqua from source

Please see the [Installing Qiskit Aqua from
Source](https://qiskit.org/documentation/contributing_to_qiskit.html#installing-aqua-from-source)
section of the Qiskit documentation.

Note: Aqua depends on Ignis and Terra, and has optional dependence on Aer and IBM Q Provider, so
these should be installed too. The master branch of Aqua is kept working with those other element
master branches so these should be installed from source too following the the instructions at 
the same location

Aqua also has some other optional dependents see 
[Aqua optional installs](https://github.com/Qiskit/qiskit-aqua#optional-installs) and
[Chemistry optional installs](https://github.com/Qiskit/qiskit-aqua#optional-installs-1) for
further information. Unit tests that require any of the optional dependents will check
and skip the test if not installed.

### Testing

Once you've made a code change, it is important to verify that your change
does not break any existing tests and that any new tests that you've added
also run successfully. Before you open a new pull request for your change,
you'll want to run the test suite locally.

The test suite can be run from a command line or via your IDE. You can run `make test` which will
run all unit tests. Another way to run the test suite is to use
[**tox**](https://tox.readthedocs.io/en/latest/#). For more information about using tox please
refer to
[Terra CONTRIBUTING](https://github.com/Qiskit/qiskit-terra/blob/master/CONTRIBUTING.md#test)
Test section. However please note Aqua does not have any
[online tests](https://github.com/Qiskit/qiskit-terra/blob/master/CONTRIBUTING.md#online-tests)
nor does it have
[test skip
 options](https://github.com/Qiskit/qiskit-terra/blob/master/CONTRIBUTING.md#test-skip-options).    

### Development Cycle

The development cycle for qiskit-aqua is informed by release plans in the 
[Qiskit rfcs repository](https://github.com/Qiskit/rfcs)
 
### Branches

* `master`

The master branch is used for development of the next version of qiskit-aqua.
It will be updated frequently and should not be considered stable. The API
can and will change on master as we introduce and refine new features.

* `stable`

The stable branch is used to maintain released versions of qiskit-aqua.
It is tagged for each version of the code corresponding to the release of
that version on PyPI. The only changes that will be merged to stable
are for bug fixes. For further information please refer to the Qiskit
[stable branch
 policy](https://qiskit.org/documentation/contributing_to_qiskit.html#stable-branch-policy)

### Release Cycle

From time to time, we will release brand new versions of Qiskit Aqua.
These are well-tested versions of the software.

When the time for a new release has come, we will:

1.  Merge the `master` branch with the `stable` branch.
2.  Create a new tag with the version number in the `stable` branch.
3.  Create and distribute the pip package.
4.  Change the `master` version to the next release version.
5.  Announce the new version to the world!

The `stable` branch will only receive changes in the form of critical bug
fixes, so the third number, the patch/maintenance number of the version,
of its form _major.minor.maintenance_, will increase when a fixed version is
released from the stable branch.
