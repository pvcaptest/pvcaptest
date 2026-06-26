.. currentmodule:: captest

Clear-sky Modeling
==================

The :py:mod:`captest.clearsky` module provides the pvlib-based clear-sky GHI/POA
modeling used by :py:func:`~captest.io.load_data` when site metadata is supplied.
This is distinct from clear-sky *filtering*, which is the
:py:class:`~captest.filters.Clearsky` step (it calls
``pvlib.clearsky.detect_clearsky`` directly and does not depend on this module).

.. autosummary::
   :toctree: generated/

   clearsky.csky
   clearsky.pvlib_location
   clearsky.pvlib_system
   clearsky.get_tz_index
