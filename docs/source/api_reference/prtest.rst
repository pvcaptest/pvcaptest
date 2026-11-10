.. currentmodule:: captest

Performance Ratio
=================

Pure functions for computing performance ratio (PR) test quantities: expected
energy, the complete-row set, references (fixed, derived, and monthly),
measured PR, and the pass/fail judgement. These are separate from the ASTM
E2848 capacity test workflow. Most users call
:py:meth:`~captest.capdata.CapData.performance_ratio` or
:py:func:`~captest.prtest.performance_ratio`, which compose the pieces below
into a :py:class:`~captest.prtest.PrTestResults`.

Constants
---------

.. data:: captest.prtest.G_STC

   Irradiance at standard test conditions, 1000 W/m2.

.. data:: captest.prtest.POWER_UNITS

   Power unit names (``"W"``, ``"kW"``, ``"MW"``) and their factors to W.

.. data:: captest.prtest.ENERGY_UNITS

   Energy unit names (``"Wh"``, ``"kWh"``, ``"MWh"``) and their factors to Wh.

Entry point
-----------

.. autosummary::
   :toctree: generated/

   prtest.performance_ratio

Units and time step
-------------------

.. autosummary::
   :toctree: generated/

   prtest.dt_hours
   prtest.to_energy_wh
   prtest.power_from_energy
   prtest.resample_interval_energy

Expected energy and temperature
-------------------------------

.. autosummary::
   :toctree: generated/

   prtest.validate_power_temp_coeff
   prtest.validate_availability
   prtest.expected_energy_wh
   prtest.weighted_cell_temp

Complete rows and ratio
-----------------------

.. autosummary::
   :toctree: generated/

   prtest.complete_rows
   prtest.pr_ratio

Degradation
-----------

.. autosummary::
   :toctree: generated/

   prtest.degradation_factor
   prtest.degradation_years

References
----------

.. autosummary::
   :toctree: generated/

   prtest.fixed_reference
   prtest.derive_reference
   prtest.weighted_expected_pr

Measured PR and judgement
-------------------------

.. autosummary::
   :toctree: generated/

   prtest.measured_pr
   prtest.parse_pr_tolerance
   prtest.judge

Result classes
--------------

.. autosummary::
   :toctree: generated/

   prtest.PrTestResults
   prtest.Reference
   prtest.RowSet
   prtest.MeasuredPr
   prtest.Judgement
