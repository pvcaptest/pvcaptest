.. currentmodule:: captest

CapData
=======

The :py:class:`~captest.capdata.CapData` class is the core interface for
capacity testing. It holds raw and filtered data, column group mappings, filter
history, and regression results.

.. autosummary::
   :toctree: generated/

   capdata.CapData

.. _capdata-api-attributes:

Attributes
----------

The state a :py:class:`~captest.capdata.CapData` carries.
:py:attr:`~captest.capdata.CapData.data` is the only frame that is stored:
:py:attr:`~captest.capdata.CapData.data_filtered` is read-only, derived from
``data`` and the applied :py:attr:`~captest.capdata.CapData.filters` chain.

.. autosummary::
   :toctree: generated/

   capdata.CapData.data_filtered
   capdata.CapData.filters
   capdata.CapData.prep
   capdata.CapData.name

The instance attributes — :py:attr:`~captest.capdata.CapData.data`,
``column_groups``, ``regression_cols``, ``regression_formula``,
``regression_results``, ``rc``, ``loc``, ``floc``, ``data_loader``, and
``tolerance`` — are documented in the
:py:class:`~captest.capdata.CapData` class reference above.

Setup
-----

Methods that must be called after loading data and before filtering or
fitting a regression. Use :meth:`~captest.capdata.CapData.set_regression_cols`
to map regression terms to column names or column group ids, then call
:meth:`~captest.capdata.CapData.process_regression_columns` to validate and
prepare those mappings. :meth:`~captest.capdata.CapData.custom_param` adds a
column of derived values (e.g. temperature-corrected power or spectral
corrections) directly to the :py:class:`~captest.capdata.CapData` instance;
see :doc:`calcparams` for the available calculation functions.

.. autosummary::
   :toctree: generated/

   capdata.CapData.set_regression_cols
   capdata.CapData.process_regression_columns
   capdata.CapData.custom_param

Data Management
---------------

Methods for inspecting, renaming, copying, and exporting data.
:meth:`~captest.capdata.CapData.drop_cols` and
:meth:`~captest.capdata.CapData.rename_cols` record a prep step by default;
see :ref:`data_prep`.

.. autosummary::
   :toctree: generated/

   capdata.CapData.get_reg_cols
   capdata.CapData.review_column_groups
   capdata.CapData.create_column_group_attributes
   capdata.CapData.copy
   capdata.CapData.empty
   capdata.CapData.drop_cols
   capdata.CapData.rename_cols

.. _capdata-api-prep:

Data Preparation
----------------

Thin wrappers that build a step class from :doc:`prep` and
``run()`` it, appending it to the ``CapData.prep`` chain. Prep steps rewrite
``data`` in place between loading and ``setup()``. ``describe_prep`` returns a
written summary of what was applied, while ``prep_to_config`` / ``run_prep``
serialize and replay the chain. See :doc:`prep` for the underlying step
classes and :ref:`data_prep` for the workflow.

.. autosummary::
   :toctree: generated/

   capdata.CapData.prep_convert_units
   capdata.CapData.prep_scale
   capdata.CapData.prep_astype
   capdata.CapData.prep_custom
   capdata.CapData.describe_prep
   capdata.CapData.prep_to_config
   capdata.CapData.run_prep

Aggregation
-----------

Methods for aggregating sensor readings into single representative columns.

.. autosummary::
   :toctree: generated/

   capdata.CapData.agg_sensors
   capdata.CapData.agg_group
   capdata.CapData.expand_agg_map
   capdata.CapData.reset_agg

.. _capdata-api-filtering:

Filtering
---------

Thin wrappers that build a step class from :doc:`filters` and
``run()`` it, appending it to the ``CapData.filters`` chain (the single source
of truth from which ``data_filtered`` is derived). Each accepts an optional
``custom_name`` label. ``describe_filters`` returns a written summary of the
run, while ``filters_to_config`` / ``run_pipeline`` serialize and replay the
chain and ``rerun_filters_from`` re-runs the chain from a given step with the steps'
current parameter values. See :doc:`filters` for the underlying step classes.

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
   capdata.CapData.filter_rolling_std
   capdata.CapData.filter_abs_diff_prev
   capdata.CapData.filter_flag
   capdata.CapData.filter_threshold
   capdata.CapData.filter_custom
   capdata.CapData.filter_sensors
   capdata.CapData.filter_sensors_abs_diff
   capdata.CapData.filter_clearsky
   capdata.CapData.filter_backtracking
   capdata.CapData.filter_missing
   capdata.CapData.filter_op_state
   capdata.CapData.reset_filter
   capdata.CapData.describe_filters
   capdata.CapData.filters_to_config
   capdata.CapData.run_pipeline
   capdata.CapData.rerun_filters_from

Reporting Conditions
--------------------

Methods for computing ASTM E2848 reporting conditions. ``rep_irr`` is the
reporting POA irradiance used to anchor ``filter_irr(ref_val='rep_irr')``; within
a :py:class:`~captest.CapTest` it resolves from the single test RC. See
:ref:`reporting_conditions` in the user guide.

.. autosummary::
   :toctree: generated/

   capdata.CapData.rep_cond
   capdata.CapData.rep_cond_freq
   capdata.CapData.rep_irr

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
