.. currentmodule:: captest

CapData
=======

The :py:class:`~captest.capdata.CapData` class is the primary interface for
capacity testing. It holds raw and filtered data, column group mappings, filter
history, and regression results.

.. autosummary::
   :toctree: generated/

   capdata.CapData

Data Management
---------------

Methods for inspecting, renaming, copying, and exporting data.

.. autosummary::
   :toctree: generated/

   capdata.CapData.set_regression_cols
   capdata.CapData.get_reg_cols
   capdata.CapData.review_column_groups
   capdata.CapData.copy
   capdata.CapData.empty
   capdata.CapData.drop_cols
   capdata.CapData.rename_cols
   capdata.CapData.process_regression_columns
   capdata.CapData.custom_param

Aggregation
-----------

Methods for aggregating sensor readings into single representative columns.

.. autosummary::
   :toctree: generated/

   capdata.CapData.agg_sensors
   capdata.CapData.agg_group
   capdata.CapData.expand_agg_map
   capdata.CapData.reset_agg

Filtering
---------

Methods that apply filters to ``data_filtered``. Each method records the
rows kept and removed in the filter history.

.. autosummary::
   :toctree: generated/

   capdata.CapData.filter_irr
   capdata.CapData.filter_pvsyst
   capdata.CapData.filter_shade
   capdata.CapData.filter_time
   capdata.CapData.filter_days
   capdata.CapData.filter_outliers
   capdata.CapData.filter_pf
   capdata.CapData.filter_power
   capdata.CapData.filter_custom
   capdata.CapData.filter_sensors
   capdata.CapData.filter_clearsky
   capdata.CapData.filter_missing
   capdata.CapData.filter_op_state
   capdata.CapData.reset_filter

Reporting Conditions
--------------------

Methods for computing ASTM E2848 reporting conditions.

.. autosummary::
   :toctree: generated/

   capdata.CapData.rep_cond
   capdata.CapData.rep_cond_freq

Regression
----------

Methods for fitting the ASTM E2848 regression and predicting capacities.

.. autosummary::
   :toctree: generated/

   capdata.CapData.fit_regression
   capdata.CapData.predict_capacities

Results and Uncertainty
-----------------------

Methods for quantifying test results, uncertainty, and completeness.

.. autosummary::
   :toctree: generated/

   capdata.CapData.uncertainty
   capdata.CapData.spatial_uncert
   capdata.CapData.expanded_uncert
   capdata.CapData.get_filtering_table
   capdata.CapData.get_summary
   capdata.CapData.print_points_summary
   capdata.CapData.get_length_test_period
   capdata.CapData.get_pts_required
   capdata.CapData.set_test_complete

Visualization
-------------

Methods for scatter plots, filter inspection, and interactive dashboards.

.. autosummary::
   :toctree: generated/

   capdata.CapData.scatter
   capdata.CapData.scatter_hv
   capdata.CapData.plot
   capdata.CapData.reg_scatter_matrix
   capdata.CapData.scatter_filters
   capdata.CapData.timeseries_filters

Export
------

Methods for writing data and column groups to Excel.

.. autosummary::
   :toctree: generated/

   capdata.CapData.data_columns_to_excel
   capdata.CapData.column_groups_to_excel
