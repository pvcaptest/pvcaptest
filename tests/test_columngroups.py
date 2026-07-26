from collections.abc import Mapping

import numpy as np
import pandas as pd
import pytest

from captest import capdata, columngroups as cg, util

col_groups = {
    "irr_poa": [
        "poa pyran met 1",
        "poa pyran met 2",
        "poa pyran met 3",
    ],
    "irr_ghi": [
        "ghi pyran met 1",
        "ghi pyran met 2",
        "ghi pyran met 3",
    ],
    "wind_vel": [
        "wind speed m/sec met 1",
        "wind speed m/sec met 2",
        "wind speed m/sec met 3",
    ],
    "realpwr_mtr": ["sel 735 mw"],
    "realpwr_inv": [
        "pcs001_inv01_power_kw",
        "pcs001_inv02_power_kw",
        "pcs001_inv03_power_kw",
        "pcs001_inv04_power_kw",
        "pcs002_inv01_power_kw",
        "pcs002_inv02_power_kw",
        "pcs002_inv03_power_kw",
        "pcs002_inv04_power_kw",
    ],
}


@pytest.fixture
def col_grp():
    """Create an instance of ColumnGroups with col_groups dict."""
    col_grp = cg.ColumnGroups(col_groups)
    return col_grp


class TestColumnGroups:
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
            grp_label.replace("_", "*"): columns
            for grp_label, columns in col_groups.items()
        }
        col_grp = cg.ColumnGroups(col_groups_bad_ids)
        for group in col_groups_bad_ids.keys():
            assert hasattr(col_grp, group)

    def test_str(self, col_grp):
        output_str = str(col_grp)
        pretty_col_groups = (
            "irr_poa:\n"
            "    poa pyran met 1\n"
            "    poa pyran met 2\n"
            "    poa pyran met 3\n"
            "irr_ghi:\n"
            "    ghi pyran met 1\n"
            "    ghi pyran met 2\n"
            "    ghi pyran met 3\n"
            "wind_vel:\n"
            "    wind speed m/sec met 1\n"
            "    wind speed m/sec met 2\n"
            "    wind speed m/sec met 3\n"
            "realpwr_mtr:\n"
            "    sel 735 mw\n"
            "realpwr_inv:\n"
            "    pcs001_inv01_power_kw\n"
            "    pcs001_inv02_power_kw\n"
            "    pcs001_inv03_power_kw\n"
            "    pcs001_inv04_power_kw\n"
            "    pcs002_inv01_power_kw\n"
            "    pcs002_inv02_power_kw\n"
            "    pcs002_inv03_power_kw\n"
            "    pcs002_inv04_power_kw\n"
        )
        assert output_str == pretty_col_groups


class TestExternalTaggingColumnGroupsCompat:
    """Pin compatibility with externally produced column_groups.json files.

    External tag-management tooling writes ``{group_id: [tag, ...]}``;
    captest's io.load_data
    reads a .json column-groups file via ``util.read_json`` into
    ``ColumnGroups`` (io.py .json branch). This test exercises that exact
    path plus attribute creation on a CapData whose columns are the tags.
    The fixture is a real externally produced file trimmed to ten groups.
    """

    PATH = "./tests/data/external_tagging_column_groups.json"

    def test_json_loads_into_column_groups(self):
        groups = cg.ColumnGroups(util.read_json(self.PATH))
        # ColumnGroups is a UserDict: dict-like, not a dict subclass.
        assert isinstance(groups, Mapping)
        assert groups["real_pwr_inv"] == [
            "SMA SC4000 UP US: Active power (KwAC) Kilowatts"
        ]
        assert len(groups["current_cmb"]) == 20
        # dict-key -> attribute mirroring (ColumnGroups.__setitem__)
        assert groups.real_pwr_inv == groups["real_pwr_inv"]
        assert groups.current_cmb == groups["current_cmb"]

    def test_capdata_attributes_from_external_groups(self):
        groups = cg.ColumnGroups(util.read_json(self.PATH))
        all_tags = [t for tags in groups.values() for t in tags]
        cd = capdata.CapData("test")
        cd.data = pd.DataFrame(
            np.ones((3, len(all_tags))),
            columns=all_tags,
            index=pd.date_range("2026-01-01", periods=3, freq="h"),
        )
        cd.column_groups = groups
        cd.create_column_group_attributes()
        assert list(cd.real_pwr_inv.columns) == groups["real_pwr_inv"]
        assert list(cd.current_cmb.columns) == groups["current_cmb"]
