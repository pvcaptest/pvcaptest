pvcaptest
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

These instructions assume that you are new to using conda and python, if
that is not the case skip to the last section for users familiar with
conda and pip.

The recommended method to install pvcaptest is to create a conda
environment for pvcaptest. Installing Anaconda or miniconda will install
both python and conda. There is no need to install python separately.

**Easiest Option:**

1. Download and install the `anaconda distribution <https://www.anaconda.com/products/individual>`__. Follow the default installation settings.
2. On Windows go to the start menu and open the Anaconda prompt under the newly installed Anaconda program. On OSX or Linux open a terminal window.
3. Install pvcaptest by typing the command ``conda install -c conda-forge pvcaptest`` and pressing enter. The ``-c conda-forge`` option tells conda to install pvcaptest from the `conda forge channel <https://conda-forge.org/#about>`__.

This will install the pvcaptest package in the base environment created when Anaconda is installed. This should work and provide you with jupyter notebook and jupyer lab to run pvcaptest in. If you think you will use your Anaconda installation to create and maintain additional environments, the following process for creating a stand alone option is likely a better option.

**Better long term option:**

1. If you do not already have it installed, download and install the `anaconda distribution <https://www.anaconda.com/products/individual>`__ or miniconda.
2. Go to the `project github page <https://github.com/pvcaptest/pvcaptest>`__ and download the project source to obtain a copy of the ``environment.yml`` file. Click the green code button and click ‘Download ZIP’.
3. On Windows go to the start menu and open the Anaconda prompt under the newly installed Anaconda program. On OSX or Linux open a terminal window. Note the path in the prompt for the next step. On Windows this should be something like ``C:\Users\username\``.
4. Unzip and move the ``environment.yml`` file to the folder identified by the path from the previous step.
5. In your Anaconda prompt or terminal type ``conda env create -f environment.yml`` and hit enter. Wait for a few seconds while conda works to solve the environment. It should ask you if you want to proceed to install new packages including pvcaptest. Type ``y`` enter to proceed and wait for conda to finish installing pvcaptest and the other packages.
6. Once the installation is complete conda will print out a command for activating the new environment. Run that command, which should be like ``conda activate captest_env``.


The environment created will include jupyter lab and notebook for you to use pvcaptest in. You can start these using the commands ``jupyter lab`` or ``jupyter notebook``.

See the `conda
documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`__
for more details on using conda to create and manage environments.

Install for users familiar with conda and pip:
----------------------------------------------

Conda install into an existing environment:

``conda install -c conda-forge pvcaptest``

If you prefer, you can pip install pvcaptest, but the recommended
approach is to use the conda package.

**Note: The conda package is named pvcaptest and the pip package is
named captest. The project is moving to consistent use of the pvcaptest
name, but the package name on pypi will remain as captest.**
