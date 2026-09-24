
.. _bifacial:

Bifacial Tests
==============


.. note::
   The manual approach described in this section can still be implemented, but the preferred approach to bifacial capacity testing as of v0.15.0 is to use one of the default test setups (`bifi_e2848_etotal_rear_shade_sim`, `bifi_e2848_etotal_rear_shade_meas`, or `bifi_power_tc_calc_tbom`) or write a `regression_columns` dictionary that will provide the calculated regressors when processed by `process_regression_columns`.

This section discusses how pvcaptest can be used to conduct a capacity test for a project with bifacial modules.

NREL Modified Bifacial Capacity Test
------------------------------------
Pvcaptest can be used to conduct a bifacial capacity test following the `NREL Suggested Modifications for Bifacial Capacity Testing <https://www.nrel.gov/docs/fy20osti/73982.pdf>`_. 

The suggested approach uses the standard ASTM regression equation:

.. math::
    P = E_{POA}\left(a_{1} + a_{2} * E_{POA} + a_{3} * T_{a} + a_{4} * v\right)

but, replaces the :math:`E_{POA}` term with :math:`E_{Total}`:

.. math::
    E_{Total} = E_{POA} + E_{Rear} * \varphi

where,

| :math:`E_{Rear}` is the rear POA irradiance and
| :math:`\varphi` is the bifaciality factor.

To conduct a bifacial capacity test you should make the following adjustments.

The regression equation default does not need to be changed.

You will need an :math:`E_{Total}` term in the `CapData.data` dataframe (`CapData.data_filtered` is derived from it).

.. code-block:: Python
    
        CapData.data['E_Total'] = CapData.data['E_POA'] + CapData.data['E_Rear'] * bifaciality
        # data_filtered is derived from data, so clear any filtering to pick up
        # the new column
        CapData.reset_filter()

You will then also need to adjust the `CapData.regression_columns` to map the `poa` term of the regression equation to the new `E_Total` column in the dataframe.

.. code-block:: Python

        CapData.set_regression_cols(
            power='real_power_column',
            poa='E_Total',
            t_amb='temp_col_or_group',
            w_vel='wind_speed_col_or_group'
        )

Alternatively, let pvcaptest calculate :math:`E_{Total}` for you by mapping the ``poa`` term to a calculation node. The registered :py:func:`~captest.calcparams.e_total` calculation reads the front and rear irradiance from the nodes in ``args`` and takes ``bifaciality`` from the ``CapData`` attribute of that name, so set it first. :py:meth:`~captest.capdata.CapData.process_regression_columns` then adds an ``e_total`` column to :py:attr:`~captest.capdata.CapData.data`:

.. code-block:: Python

        CapData.bifaciality = 0.7
        CapData.regression_cols = {
            'power': {'group': 'real_pwr_mtr', 'agg': 'sum'},
            'poa': {
                'calc': 'e_total',
                'args': {
                    'poa': {'group': 'irr_poa', 'agg': 'mean'},
                    'rpoa': {'group': 'irr_rpoa', 'agg': 'mean'},
                },
            },
            't_amb': {'group': 'temp_amb', 'agg': 'mean'},
            'w_vel': {'group': 'wind_speed', 'agg': 'mean'},
        }
        CapData.process_regression_columns()

This is the measured side of the built-in ``bifi_e2848_etotal_rear_shade_sim`` test setup; see :ref:`choosing-test-setup`.


Other Bifacial Capacity Test Approaches
---------------------------------------
The regression equation can be easily modified by simply assigning an new regression formula. For example, to conduct a regression of temperature corrected power against front side POA irradiance and rear side POA irradiance, you could use the following:

.. code-block:: Python

        CapData.regression_formula = 'power_temp_adj ~ poa_front + poa_rear'

The regression columns would also need to be updated to map the regression terms to the correct columns or groups of columns. In this case a dictionary should be assigned to the `regression_cols` attribute directly rather than using the `set_regression_cols` method.

.. code-block:: Python

        CapData.regression_cols = {
            'power_temp_adj': {'column': 'Power_Temp_Adj'},
            'poa_front': {'column': 'E_POA'},
            'poa_rear': {'column': 'E_Rear'},
        }

Each value is a node: ``{'column': ...}`` names one column of the ``data`` dataframe, ``{'group': ..., 'agg': ...}`` aggregates a column group, and ``{'calc': ..., 'args': {...}}`` runs a registered calculation such as :py:func:`~captest.calcparams.power_temp_correct`. The built-in ``bifi_power_tc_meas_tbom`` and ``bifi_power_tc_calc_tbom`` test setups use this form of regression (``power ~ poa + rpoa``) with the temperature correction calculated by pvcaptest.
