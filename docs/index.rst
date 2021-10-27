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


Citation
--------
Please cite original authors of optimization algorithms. If you like this
package::

    @software{Novik_torchoptimizers,
    	title        = {{torch-optimizer -- collection of optimization algorithms for PyTorch.}},
    	author       = {Novik, Mykola},
    	year         = 2020,
    	month        = 1,
    	version      = {1.0.1}
    }

Or use github feature: "cite this repository" button.


Supported Optimizers
====================

+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`AccSGD`   | https://arxiv.org/abs/1803.05591                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`AdaBound` | https://arxiv.org/abs/1902.09843                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`AdaMod`   | https://arxiv.org/abs/1910.12249                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`Adafactor`| https://arxiv.org/abs/1804.04235                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`AdamD`    | https://arxiv.org/abs/2110.10828                                             |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`AdamP`    | https://arxiv.org/abs/1804.00325                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`AggMo`    | https://arxiv.org/abs/2006.08217                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`DiffGrad` | https://arxiv.org/abs/1909.11015                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`Lamb`     | https://arxiv.org/abs/1904.00962                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`NovoGrad` | https://arxiv.org/abs/1905.11286                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`PID`      | https://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR18_PID.pdf                 |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`QHAdam`   | https://arxiv.org/abs/1810.06801                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`QHM`      | https://arxiv.org/abs/1810.06801                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`Ranger`   | https://arxiv.org/abs/1908.00700v2                                            |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`RangerQH` | https://arxiv.org/abs/1908.00700v2                                            |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`RangerVA` | https://arxiv.org/abs/1908.00700v2                                            |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`SGDP`     | https://arxiv.org/abs/2006.08217                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`SGDW`     | https://arxiv.org/abs/1608.03983                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`Shampoo`  | https://arxiv.org/abs/1802.09568                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`SWATS`    | https://arxiv.org/abs/1712.07628                                              |
+-----------------+-------------------------------------------------------------------------------+
|                 |                                                                               |
| :ref:`Yogi`     | https://papers.nips.cc/paper/8186-adaptive-methods-for-nonconvex-optimization |
+-----------------+-------------------------------------------------------------------------------+

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
