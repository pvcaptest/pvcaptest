.. currentmodule:: captest

Test Setup Documents
====================

``captest.setup`` holds the document model for a capacity-test setup: the
regression formula, the measured and modeled regression-column trees, the
reporting-condition options, the scatter-plot name and any required test-level
parameters. A setup is plain data (yaml, json or a dict of strings, numbers,
booleans, lists and tagged mappings); :py:class:`~captest.setup.TestSetup` is
its validated, immutable view. The built-in presets in
:data:`~captest.captest.TEST_SETUPS` are ``TestSetup`` documents, and
:py:attr:`~captest.CapTest.resolved_setup` holds the one a test used. See
:ref:`custom_test_setups` in the user guide for the node grammar and examples.

Validation runs in tiers. Tier 1 (structure, registry names, formula variables,
output-column collisions) runs whenever a document is loaded and raises a
``pydantic.ValidationError`` whose locations are document paths. Tier 2
(:py:func:`~captest.setup.check_project_fit`) checks one side against a
project's column groups, columns and parameters without reading data.

Documents
---------

.. autosummary::
   :toctree: generated/

   setup.TestSetup
   setup.PrSetup
   setup.Side
   setup.RepConditions

A document with ``kind: pr`` is a performance-ratio setup
(:py:class:`~captest.setup.PrSetup`); a document without ``kind`` is a regression
setup. :py:func:`~captest.setup.validate_setup` and
:py:func:`~captest.setup.load_setup` dispatch on ``kind``. A
:py:class:`~captest.CapTest` raises ``NotImplementedError`` for a PR setup until
PR support lands.

.. autosummary::
   :toctree: generated/

   setup.validate_setup
   setup.load_setup

Nodes
-----

The three node kinds a ``reg_cols`` tree is built from. Any other value inside a
calculation's ``args`` is a literal.

.. autosummary::
   :toctree: generated/

   setup.Group
   setup.Column
   setup.Calc

Deriving and Checking
---------------------

.. autosummary::
   :toctree: generated/

   setup.derive
   setup.derive_pr
   setup.check_project_fit
   setup.FitError
   setup.SetupFitError

Helpers
-------

.. autosummary::
   :toctree: generated/

   setup.canonical_json
   setup.parse_regression_formula
