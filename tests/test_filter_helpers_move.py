"""Guards that row-filter helpers are importable from captest.filters."""

import pandas as pd


def test_helpers_importable_from_filters():
    from captest.filters import (
        perc_difference,
        check_all_perc_diff_comb,
        abs_diff_from_average,
        sensor_filter,
        filter_irr,
        filter_grps,
    )

    for fn in (
        perc_difference,
        check_all_perc_diff_comb,
        abs_diff_from_average,
        sensor_filter,
        filter_irr,
        filter_grps,
    ):
        assert callable(fn)


def test_filter_irr_behavior_unchanged():
    from captest.filters import filter_irr

    df = pd.DataFrame({"poa": [100, 500, 900]})
    out = filter_irr(df, "poa", 200, 800)
    assert list(out["poa"]) == [500]


def test_capdata_still_exposes_helpers():
    """capdata re-imports the helpers it still uses internally."""
    from captest import capdata

    assert callable(capdata.filter_irr)
    assert callable(capdata.filter_grps)
    assert callable(capdata.sensor_filter)
    assert callable(capdata.check_all_perc_diff_comb)
