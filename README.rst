What is captest?
================

Captest is intended to facilitate capacity testing following ASTM E2848.
The captest module contains a single class, CapData, which provides
methods for loading, visualizing, filtering, and regressing capacity
testing data. The module also includes functions that take CapData
objects as arguments and provide summary data and capacity test results.

Please see the Jupyter notebooks in the examples directory, which
include examples of the core features.

Installation
============

The recommended method to install captest is to create an environment
using conda and then pip installing captest within your new environment.

There are a few ways to go about this as listed below.

If you do not have conda installed:
-----------------------------------

Downloading and installing the `anaconda
distribution <https://www.anaconda.com/distribution/#download-section>`__
will install python and all packages required to use captest except
pvlib.

Then you can simply use: ``pip install captest``

To install pvlib also use: ``pip install captest[csky]``

If you have conda installed:
----------------------------

Install into a new conda environment:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you already have conda installed and are familiar with its use, then
the recommended method to install captest is to create a new environment
using the provided `environment yml
file <https://github.com/bt-/pvcaptest/blob/master/environment.yml>`__.
Download this file and then run:

``conda env create -f environment.yml``

Activate the new environment:

``conda activate captest_env_05``

Then pip install captest:

``pip install captest``

The environment created includes all captest dependenices including
Holoviews and PVLIB.

Install into an existing conda environment:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you prefer to use an existing environment, you can pip install
captest.

``pip install captest``

This will not load the optional Holoviews and PVLIB dependencies, which
captest relies on for advanced plotting and clear sky modelling. You can
load either or both of these with the following:

Load both: ``pip install captest[all]``

Load PVLIB: ``pip install captest[csky]``

Load Holoviews: ``pip install captest[viz]``
