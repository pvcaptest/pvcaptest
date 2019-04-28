# captest

<table>

<tr>
  <td>Latest Release</td>
  <td><img src="https://badge.fury.io/py/captest.svg"
           alt="Latest release version" /></td>
</tr>

<tr>
  <td>Live Example</td>
  <td>
    <a href="https://mybinder.org/v2/gh/bt-/pvcaptest/master?filepath=examples%2Fcaptest_example.ipynb">
    <img src="https://mybinder.org/badge.svg"
         alt="Live captest example on MyBinder" />
    </a>
  </td>
</tr>
</table>

# What is captest?
Captest is intended to facilitate capacity testing following ASTM E2848.  The module contains a single class, CapData, which provides methods for loading, visualizing, filtering, and regressing capacity testing data.  The module also includes functions that take CapData objects as arguments and provide summary data and capacity test results.

# Installation
The recommended method to install captest is to create an environment using conda and then pip installing captest within your new environment.

There are a few ways to go about this as listed below.  

## If you do not have conda installed:
Downloading and installing the [anaconda distribution](https://www.anaconda.com/distribution/#download-section) will install python and all packages required to use captest except pvlib.

Then you can simply use:
`pip install  captest`

To install pvlib also use:
`pip install captest[csky]`


## If you have conda installed:
If you already have conda installed and are familiar with its use.

`conda create -n new_env python=3.6 notebook pip`

Use the above to create a new environment, where `new_env` is whatever name you would like to use.  Python 3.5 and above should all work, thorough testing is not complete yet agains all python versions.

Activate the new env and then pip install captest:
`pip install captest`

Pip will install the captest dependencies. captest relies on the package Holoviews for advanced plotting and the package pvlib for clear sky modelling.  You can load either or both of these with the following:

`pip install captest[all]`
`pip install captest[csky]`
`pip install captest[viz]`
