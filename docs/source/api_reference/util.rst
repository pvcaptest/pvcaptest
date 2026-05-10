.. currentmodule:: captest

Utilities
=========

Miscellaneous helper functions used internally and available for
advanced users.

I/O Helpers
-----------

.. autosummary::
   :toctree: generated/

   util.read_json
   util.read_yaml

Time Series
-----------

.. autosummary::
   :toctree: generated/

   util.get_common_timestep
   util.reindex_datetime
   util.detect_solar_noon

Irradiance
----------

.. autosummary::
   :toctree: generated/

   util.generate_irr_distribution

Column and Tag Operations
-------------------------

.. autosummary::
   :toctree: generated/

   util.tags_by_regex
   util.append_tags
   util.get_agg_column_name

Regression
----------

.. autosummary::
   :toctree: generated/

   util.parse_regression_formula
   util.process_reg_cols
   util.transform_calc_params

Configuration
-------------

.. autosummary::
   :toctree: generated/

   util.update_by_path
