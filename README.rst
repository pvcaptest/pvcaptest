pvcaptest
========

What is pvcaptest?
==================

pvcaptest is an open source python package created to facilitate
capacity testing following the ASTM E2848 standard. The captest module
contains a single class, CapData, which provides methods for loading,
visualizing, filtering, and regressing capacity testing data. The module
also includes functions that take CapData objects as arguments and
provide summary data and capacity test results.

Documentation and examples are available on
`readthedocs <https://pvcaptest.readthedocs.io/en/latest/>`__ including
full examples in jupyter notebooks that can be run in the browser
without installing anything.

Installation
============

With Conda
----------

These instructions assume that you are new to python and managing
virtual environments to isolate pvcaptest and its dependencies from the
rest of your operating system. If you are not sure what that means,
these instructions (using miniforge) are for you!

Install Miniforge
~~~~~~~~~~~~~~~~~

The first step to using pvcaptest is to install pvcaptest and Jupyter
into an isolated python environment. These directions guide you through
the steps to use conda to do that.

Miniforge will install conda, but with the default source for packages
set to conda-forge.

1. Go to the `Miniforge github
   page <https://github.com/conda-forge/miniforge>`__ to download
   miniforge. Pick the correct installation option for your operating
   system and follow the directions.

Windows:

- Use the default options and the .exe installer.

OSX / Linux:

- Open a command line (Type command + space to open Spotlight and type
  terminal and open your default terminal app) and copy and paste the
  commands and hit enter.

Installing pvcaptest
~~~~~~~~~~~~~~~~~~~~

Open a command line, which we will use to run the conda commands
required to install pvcaptest.

- Windows: Type miniforge into your search bar and select the Miniforge
  Prompt app, which should be the first option.

- OSX: Type command + space to open Spotlight and type terminal and open
  your default terminal app.

Now we can use conda to create a new conda environment and install
pvcaptest, its dependencies, and Jupyter. Copy and paste the following
command into the command line and hit enter:

``conda create --name pvcaptest python=3.12 pvcaptest notebook``

See the `conda
documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`__
for managing environments for more information on what that command is
doing.

Once the installation finishes, you should see a message similar to the
one below. The installation process should run for 5-10 minutes, but it
will depend on your computer specs and internet speed.

\```Preparing transaction: done Verifying transaction: done Executing
transaction: done # # To activate this environment, use # # $ conda
activate pvcaptest # # To deactivate an active environment, use # # $
conda deactivate

Retrieving notices: …working… done \``\`

**Activate the environment using the command provded.**

Try running this command in your terminal as a quick check that
pvcaptest did install correctly:

``python -c 'import captest; print(captest.__version__)``

You may see some warning messages, but at the bottom you should see a
version number.You should have the latest version installed; ``0.13.3``
or later. You can check what the most recent release is on
`pypi <https://pypi.org/project/captest/#history>`__.

Congratulations! You now have pvcaptest installed in an isolated conda
environment.

Run ``jupyter lab`` which will open jupyter lab in your default browser.
You can now use pvcaptest in jupyter lab.

You may want to either navigate to the folder containing your project
files in the terminal before running ``jupyter lab`` or store your
project files in a location you can easily navigate to in jupyter lab’s
file browser.

With uv - Contributors
----------------------

Using uv is the preferred method for contributors setting up a
development installation as of v0.13.3. See
`MAINTAINER.md <MAINTAINER.md>`__ for additional details, specifically
the section on just.

Install
`uv <https://docs.astral.sh/uv/getting-started/installation/#__tabbed_1_1>`__
and clone your fork of the repository.

The just recipes (commands) use ``uv run ...`` to run development
workflow commands (e.g., ``uv run --python 3.12 pytest tests/``) and
``uv run`` will ensure a venv with the necessary dependencies is used.

With pip
--------

``pip install captest``

or with optional dependencies:

``pip install captest[optional]``

**Note: The conda package is named pvcaptest and the pip package is
named captest. The project is moving to consistent use of the pvcaptest
name, but the package name on pypi will remain as captest.**
