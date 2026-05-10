.. currentmodule:: captest

Calculation Parameters
======================

Functions for computing custom regression parameters (e.g., temperature
corrections, spectral corrections, effective irradiance) to be used as
additional columns in the :py:class:`~captest.capdata.CapData` regression.

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
