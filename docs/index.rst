.. pytorch-optimizer documentation master file, created by
   sphinx-quickstart on Thu Feb 13 21:14:16 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pytorch-optimizer's documentation!
=============================================

**torch-optimizer** -- collection of optimizers for PyTorch_.

Simple example
--------------

.. code:: python

    import torch_optimizer as optim

    # model = ...
    optimizer = optim.DiffGrad(model.parameters(), lr=0.001)
    optimizer.step()


Installation
------------
Installation process is simple, just::

    $ pip install torch_optimizer


Supported Optimizers
====================

+-------------+-------------------------------------------------------------------------------+
|             |                                                                               |
|  AccSGD     | https://arxiv.org/abs/1803.05591                                              |
+-------------+-------------------------------------------------------------------------------+
|             |                                                                               |
|  AdaBound   | https://arxiv.org/abs/1902.09843                                              |
+-------------+-------------------------------------------------------------------------------+
|             |                                                                               |
|  AdaMod     | https://arxiv.org/abs/1910.12249                                              |
+-------------+-------------------------------------------------------------------------------+
|             |                                                                               |
|  DiffGrad   | https://arxiv.org/abs/1909.11015                                              |
+-------------+-------------------------------------------------------------------------------+
|             |                                                                               |
|  Lamb       | https://arxiv.org/abs/1904.00962                                              |
+-------------+-------------------------------------------------------------------------------+
|             |                                                                               |
|  RAdam      | https://arxiv.org/abs/1908.03265                                              |
+-------------+-------------------------------------------------------------------------------+
|             |                                                                               |
|  SGDW       | https://arxiv.org/abs/1608.03983                                              |
+-------------+-------------------------------------------------------------------------------+
|             |                                                                               |
|  Yogi       | https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization |
+-------------+-------------------------------------------------------------------------------+


.. toctree::
   :maxdepth: 2
   :caption: Contents:

Contents
--------

.. toctree::
   :maxdepth: 2

   api
   examples
   contributing


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _Python: https://www.python.org
.. _PyTorch: https://github.com/pytorch/pytorch
