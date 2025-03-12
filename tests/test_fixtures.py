import pandas as pd


def test_cd_nested_col_groups_fixture(cd_nested_col_groups):
    """Test that the cd_nested_col_groups fixture creates expected data."""
    cd = cd_nested_col_groups

    # Check that we have the expected column groups
    expected_groups = {
        "irr_poa",
        "irr_poa_met1",
        "irr_poa_met2",
        "irr_rpoa",
        "irr_rpoa_met1",
        "irr_rpoa_met2",
    }
    assert set(cd.column_groups.keys()) == expected_groups

    # Check that data has correct time range and frequency
    expected_start = pd.Timestamp("2023-01-01")
    expected_end = expected_start + pd.Timedelta(days=2)
    assert cd.data.index[0] == expected_start
    assert cd.data.index[-1] == expected_end
    assert cd.data.index.freq == "5T"

    # Check that all columns from column groups exist in data
    all_expected_columns = []
    for columns in cd.column_groups.values():
        all_expected_columns.extend(columns)
    assert all(col in cd.data.columns for col in all_expected_columns)

    # Check that data values are within expected range for irradiance
    assert cd.data.min().min() >= 0
    assert cd.data.max().max() <= 1200

    # Check that nighttime values (before 6am and after 6pm) are 0
    night_data = cd.data[(cd.data.index.hour < 6) | (cd.data.index.hour > 18)]
    assert (night_data == 0).all().all()

    # Check that data_filtered is a copy of data
    pd.testing.assert_frame_equal(cd.data, cd.data_filtered)
