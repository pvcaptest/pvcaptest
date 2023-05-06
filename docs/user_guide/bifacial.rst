
.. _bifacial:

Bifacial Tests
==============

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

You will need an :math:`E_{Total}` term in the `CapData.data` and `CapData.data_filtered` dataframes.

.. code-block:: Python
    
        CapData.data['E_Total'] = CapData.data['E_POA'] + CapData.data['E_Rear'] * bifaciality
        # either of the below lines will copy the modified data `data_filtered`
        CapData.data_filtered = CapData.data.copy()
        CapData.reset_filter()

You will then also need to adjust the `CapData.regression_columns` to map the `poa` term of the regression equation to the new `E_Total` column in the dataframe.

.. code-block:: Python

        CapData.set_regression_cols(
            power='real_power_column',
            poa='E_Total',
            t_amb='temp_col_or_group',
            w_vel='wind_speed_col_or_group'
        )


Other Bifacial Capacity Test Approaches
---------------------------------------
The regression equation can be easily modified by simply assigning an new regression formula. For example, to conduct a regression of temperature corrected power against front side POA irradiance and rear side POA irradiance, you could use the following:

.. code-block:: Python

        CapData.regression_formula = 'power_temp_adj ~ poa_front + poa_rear'

The regression columns would also need to be updated to map the regression terms to the correct columns or groups of columns. In this case a dictionary should be assigned to the `regression_cols` attribute directly rather than using the `set_regression_cols` method.

.. code-block:: Python

        CapData.regression_cols = {
            'power_temp_adj': 'poa_front',
            'poa_front': 'E_POA',
            'poa_rear': 'E_Rear'
        }
