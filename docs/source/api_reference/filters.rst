.. currentmodule:: captest

Filters
=======

The :py:mod:`captest.filters` module holds all the filters. Each
filter is a first-class object: a :py:class:`~captest.filters.BaseSummaryStep`
subclass that declares its configuration as typed ``param`` parameters and
implements ``_execute``. The filter methods on :py:class:`~captest.capdata.CapData` 
are thin wrappers that build the matching step and ``run()`` it. Steps may also be 
constructed and run directly:

.. code-block:: Python

   from captest.filters import Irradiance, RepCond

   Irradiance(low=200, high=800, custom_name="Irradiance bounds").run(cd)
   RepCond(percent_filter=20).run(cd)

Base Classes
------------

:py:class:`~captest.filters.BaseSummaryStep` is the common ancestor for
everything that appears in the summary table; it owns the ``run()`` lifecycle and
the ``custom_name`` label. :py:class:`~captest.filters.BaseFilter` adds the
contract that ``_execute`` returns the pandas ``Index`` of rows to keep.

.. autosummary::
   :toctree: generated/

   filters.BaseSummaryStep
   filters.BaseFilter

Filter Steps
------------

Each step removes rows from ``CapData.data_filtered``. The corresponding
``CapData.filter_*`` wrapper builds and runs the step.

.. autosummary::
   :toctree: generated/

   filters.Irradiance
   filters.Pvsyst
   filters.Shade
   filters.Time
   filters.Days
   filters.Outliers
   filters.PowerFactor
   filters.Power
   filters.Custom
   filters.Sensors
   filters.Clearsky
   filters.Backtracking
   filters.Missing
   filters.RollingStd
   filters.AbsDiffPrev
   filters.BooleanFlag
   filters.Regression

Reporting Conditions
--------------------

:py:class:`~captest.filters.RepCond` is a zero-removal step: it computes
``CapData.rc`` and returns the unchanged index, so it appears in the summary at
its position in the filter chain without removing any data.

.. autosummary::
   :toctree: generated/

   filters.RepCond


Row-filter Helpers
------------------

Standalone row-filtering functions that the step classes (and a few non-filter
:py:class:`~captest.capdata.CapData` methods) build on. These moved here from
``capdata.py`` during the filter-class refactor.

.. autosummary::
   :toctree: generated/

   filters.filter_irr
   filters.sensor_filter
   filters.check_all_perc_diff_comb
   filters.perc_difference
   filters.abs_diff_from_average
   filters.backtracking_active
