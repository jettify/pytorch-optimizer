Contributing
============

Running Tests
-------------

.. _GitHub: https://github.com/jettify/pytorch-optimizer
.. _PyTorch: https://github.com/pytorch/pytorch

Thanks for your interest in contributing to ``pytorch-optimizer``, there are multiple
ways and places you can contribute.

First of all just clone repository::

    $ git clone git@github.com:jettify/pytorch-optimizer.git

Create virtualenv with python3.5 (older version are not supported). For example
using *virtualenvwrapper* commands could look like::

   $ cd pytorch-optimizer
   $ mkvirtualenv --python=`which python3.7` pytorch-optimizer


After that please install libraries required for development::

    $ pip install -r requirements-dev.txt
    $ pip install -e .

Congratulations, you are ready to run the test suite::

    $ make cov

To run individual use following command::

    $ py.test -sv tests/test_basic.py -k test_name


Reporting an Issue
------------------
If you have found issue with `pytorch-optimizer` please do
not hesitate to file an issue on the GitHub_ project. When filing your
issue please make sure you can express the issue with a reproducible test
case.

When reporting an issue we also need as much information about your environment
that you can include. We never know what information will be pertinent when
trying narrow down the issue. Please include at least the following
information:

* Version of `pytorch-optimizer`, `python`.
* Version PyTorch_ if installed.
* Version or CUDA if installed.
* Platform you're running on (OS X, Linux).
