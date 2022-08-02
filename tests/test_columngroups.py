import os
import collections
import unittest
import pytest
import pytz
import numpy as np
import pandas as pd

from .context import columngroups as cg

col_groups = {
    'irr_poa':[
        'poa pyran met 1',
        'poa pyran met 2',
        'poa pyran met 3',
    ],
    'irr-ghi':[
        'ghi pyran met 1',
        'ghi pyran met 2',
        'ghi pyran met 3',
    ],
    'wind_vel':[
        'wind speed m/sec met 1',
        'wind speed m/sec met 2',
        'wind speed m/sec met 3',
    ],
    'realpwr_mtr':[
        'sel 735 mw'
    ],
    'realpwr-inv':[
        'pcs001_inv01_power_kw',
        'pcs001_inv02_power_kw',
        'pcs001_inv03_power_kw',
        'pcs001_inv04_power_kw',
        'pcs002_inv01_power_kw',
        'pcs002_inv02_power_kw',
        'pcs002_inv03_power_kw',
        'pcs002_inv04_power_kw',
    ]
}

class TestColumnGroups():
    def test_assign_column_groups(self):
        """Check that the column group dictionary keys become attributes."""
        col_grp = cg.ColumnGroups(col_groups)
        for group in col_groups.keys():
            assert hasattr(col_grp, group)

    def test_assign_column_groups_bad_chars_in_group_ids(self):
        """Check that the column group dictionary keys become attributes."""
        col_groups_bad_ids = {
            grp_label.replace('_', '*'):columns for
            grp_label, columns in col_groups.items()
            }
        col_grp = cg.ColumnGroups(col_groups_bad_ids)
        for group in col_groups_bad_ids.keys():
            assert hasattr(col_grp, group)
