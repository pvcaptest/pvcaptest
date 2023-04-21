import os
import collections
import unittest
import pytest
import pytz
import numpy as np
import pandas as pd

from captest import columngroups as cg

col_groups = {
    'irr_poa':[
        'poa pyran met 1',
        'poa pyran met 2',
        'poa pyran met 3',
    ],
    'irr_ghi':[
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
    'realpwr_inv':[
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

@pytest.fixture
def col_grp():
    """Create an instance of ColumnGroups with col_groups dict."""
    col_grp = cg.ColumnGroups(col_groups)
    return col_grp

class TestColumnGroups():
    @pytest.fixture(autouse=True)
    def _pass_fixtures(self, capsys):
        self.capsys = capsys

    def test_assign_column_groups(self, col_grp):
        """Check that the column group dictionary keys become attributes."""
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

    def test_str(self, col_grp):
        output_str = str(col_grp)
        pretty_col_groups = (
            'irr_poa:\n'
            '    poa pyran met 1\n'
            '    poa pyran met 2\n'
            '    poa pyran met 3\n'
            'irr_ghi:\n'
            '    ghi pyran met 1\n'
            '    ghi pyran met 2\n'
            '    ghi pyran met 3\n'
            'wind_vel:\n'
            '    wind speed m/sec met 1\n'
            '    wind speed m/sec met 2\n'
            '    wind speed m/sec met 3\n'
            'realpwr_mtr:\n'
            '    sel 735 mw\n'
            'realpwr_inv:\n'
            '    pcs001_inv01_power_kw\n'
            '    pcs001_inv02_power_kw\n'
            '    pcs001_inv03_power_kw\n'
            '    pcs001_inv04_power_kw\n'
            '    pcs002_inv01_power_kw\n'
            '    pcs002_inv02_power_kw\n'
            '    pcs002_inv03_power_kw\n'
            '    pcs002_inv04_power_kw\n'
        )
        assert output_str == pretty_col_groups
