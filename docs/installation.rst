.. _installation:

Installation
============

Quick Start
------------

If you are using conda:

``conda install -c conda-forge pvcaptest``

If you are using pip:

``pip install captest[optional]``

See below, for additional pip installation options.


Recommended Installation
-------------------------

The recommended method to install pvcaptest is to create a conda
environment for pvcaptest. Installing Anaconda or miniconda will install
both python and conda. There is no need to install python separately i.e. it is NOT recommended to install python using the python installer from python.org or to rely on the python installation that comes with your operating system.

If you are new to Python and virtual environments, the `conda documentation <https://docs.conda.io/projects/conda/en/stable/user-guide/concepts/environments.html#virtual-environments>`__ provides a succinct summary of what a virtual environment is and why you might want to use one.

Another helpful resource is the `installation section <https://pandas.pydata.org/docs/getting_started/install.html#installing-pandas>`__ of the pandas documentation.

**Easiest Option:**

1. Download and install the `anaconda distribution <https://www.anaconda.com/download/>`__. Follow the default installation settings.
2. On Windows go to the start menu and open the Anaconda prompt under the newly installed Anaconda program. On OSX or Linux open a terminal window.
3. Install pvcaptest by typing the command ``conda install -c conda-forge pvcaptest`` and pressing enter. The ``-c conda-forge`` option tells conda to install pvcaptest from the `conda forge channel <https://anaconda.org/conda-forge/pvcaptest>`__.

This will install the pvcaptest package in the base environment created when Anaconda is installed. This should work and provide you with jupyter notebook and jupyter lab to run pvcaptest in. If you think you will use your Anaconda installation to create and maintain additional environments, the following process for creating a stand alone option is likely a better option.

**Better long term option:**

1. If you do not already have it installed, download and install the `anaconda distribution <https://www.anaconda.com/download/>`__ or `miniconda <https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links>`__.
2. Go to the `environment.yaml <https://raw.githubusercontent.com/pvcaptest/pvcaptest/master/environment.yml>`__ file, right click, and select "save page as" to download the ``environment.yml`` file.
3. On Windows go to the start menu and open the Anaconda prompt under the newly installed Anaconda program. On OSX or Linux open a terminal window. Note the path in the prompt for the next step. On Windows this should be something like ``C:\Users\username\``. If you don't see the path, type ``cd`` and hit enter (Windows) or ``pwd`` (OSX or Linux) to display the path of the current directory.
4. Move the ``environment.yml`` file to the directory identified by the path from the previous step.
5. In your Anaconda prompt or terminal type ``conda env create -f environment.yml`` and hit enter. Wait while conda works to solve the environment and install the packages.
6. Once the installation is complete conda will print out a command for activating the new environment. Run that command, which should be like ``conda activate pvcaptest`` to activate the new environment.


The environment created will include jupyter lab and notebook for you to use pvcaptest in. You can start these using the commands ``jupyter lab`` or ``jupyter notebook``.

See the `conda
documentation <https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file>`__
for more details on using conda to create and manage environments.

pip Install Options
--------------------
If you prefer, you can pip install pvcaptest, but the recommended
approach is to use the conda package.

**Note: The conda package is named pvcaptest and the pip package is
named captest. The project is moving to consistent use of the pvcaptest
name, but the package name on pypi will remain as captest.**

Pip installation provides a ways to install optional dependencies:

``pip install captest[optional]``

Will install the optional dependencies: holoviews, panel, pvlib, and openpyxl. For users who want full functionality, but do not want to run the tests or build the documentation this is the recommended method.

``pip install captest[test]``

Will install the dependencies needed to run the tests.

``pip install captest[test, build]``

Will install the dependencies needed to run the tests and build the package.

``pip install captest[docs]``

Will install the dependencies needed to build the documentation. Note that the examples require nbsphinx, which requires pandoc, which pip will not install automatically. You will need to install pandoc separately. Using conda to install pandoc is recommended per the `nbshpinx documentation <https://nbsphinx.readthedocs.io/en/0.9.1/installation.html#pandoc>`__. Pip installing pandoc will install the python wrapper, but not pandoc itself, which is written in Haskell. Using conda to install should install both the python wrapper and pandoc itself.

``pip install captest[all]``

Will install all of the optional dependencies.