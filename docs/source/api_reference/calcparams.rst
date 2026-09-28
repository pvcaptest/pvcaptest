.. currentmodule:: captest

Calculation Parameters
======================

Functions for computing custom regression parameters (e.g., temperature
corrections, spectral corrections, effective irradiance) to be used as
additional columns in the :py:class:`~captest.capdata.CapData` regression.
Each function is registered in :data:`~captest.calcparams.CALC_REGISTRY` under
its own name, which is how a test setup's ``calc`` node refers to it (see
:ref:`custom_test_setups`).

Registry
--------

.. autosummary::
   :toctree: generated/

   calcparams.register_calc
   calcparams.CalcEntry

.. data:: captest.calcparams.CALC_REGISTRY

   Registry of calculations a setup document may name under ``calc``. Keys are
   registry names; values are :py:class:`~captest.calcparams.CalcEntry`
   records. Add to it with :py:func:`~captest.calcparams.register_calc`.

Temperature Corrections
-----------------------

.. autosummary::
   :toctree: generated/

   calcparams.power_temp_correct
   calcparams.bom_temp
   calcparams.cell_temp
   calcparams.avg_typ_cell_temp

Irradiance and Atmosphere
-------------------------

.. autosummary::
   :toctree: generated/

   calcparams.rpoa_pvsyst
   calcparams.e_total
   calcparams.apparent_zenith
   calcparams.apparent_zenith_pvsyst
   calcparams.absolute_airmass
   calcparams.precipitable_water_gueymard
   calcparams.poa_spec_corrected

Spectral Corrections
--------------------

.. autosummary::
   :toctree: generated/

   calcparams.spectral_factor_firstsolar

Utilities
---------

.. autosummary::
   :toctree: generated/

   calcparams.scale
   calcparams.multiply
