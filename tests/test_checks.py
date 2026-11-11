import numpy as np
import pandas as pd
import pytest

from captest import checks


class FakeCapData:
    """Minimal CapData stand-in: data, data_filtered and regression_cols."""

    def __init__(self, data, regression_cols, filtered=None):
        self.data = data
        self.regression_cols = regression_cols
        self.data_filtered = data if filtered is None else filtered


def make_data(n=120, irr=500.0, start="2025-06-01 10:00"):
    index = pd.date_range(start, periods=n, freq="15min")
    return pd.DataFrame(
        {
            "poa_col": irr,
            "front_col": irr,
            "raw_col": irr,
        },
        index=index,
    )


ROLES = {"poa": "poa_col", "front_poa": "front_col"}


@pytest.fixture
def cd():
    return FakeCapData(make_data(), dict(ROLES))


def ctx(cd, **kw):
    return checks.CheckContext(dt_hours=0.25, pr_rows=cd.data.index, **kw)


class TestDuration:
    def test_pass_at_exact_hours(self, cd):
        res = checks.Duration(threshold=400, min_hours=30).evaluate(cd, ctx(cd))
        assert res.passed
        assert res.detail["hours"].iloc[0] == 30.0
        assert "30 h at or above 400" in res.summary

    def test_fail_one_short(self):
        cd = FakeCapData(make_data(119), dict(ROLES))
        res = checks.Duration(threshold=400, min_hours=30).evaluate(cd, ctx(cd))
        assert not res.passed
        assert res.detail["hours"].iloc[0] == 29.75

    def test_counts_only_pr_rows(self, cd):
        context = checks.CheckContext(dt_hours=0.25, pr_rows=cd.data.index[:100])
        res = checks.Duration(threshold=400, min_hours=30).evaluate(cd, context)
        assert not res.passed
        assert res.detail["hours"].iloc[0] == 25.0

    def test_nan_irradiance_never_qualifies(self):
        data = make_data(120)
        data.iloc[:10, :] = np.nan
        cd = FakeCapData(data, dict(ROLES))
        res = checks.Duration(threshold=400, min_hours=0).evaluate(cd, ctx(cd))
        assert res.detail["hours"].iloc[0] == 27.5


class TestIntervalCount:
    def test_boundary_is_at_or_above(self):
        cd = FakeCapData(make_data(10, irr=400.0), dict(ROLES))
        ok = checks.IntervalCount(threshold=400, min_count=10).evaluate(cd, ctx(cd))
        assert ok.passed
        assert ok.detail["count"].iloc[0] == 10
        bad = checks.IntervalCount(threshold=400.1, min_count=1).evaluate(cd, ctx(cd))
        assert not bad.passed


class TestValidation:
    @pytest.mark.parametrize("bad", [float("nan"), float("inf")])
    def test_non_finite_rejected(self, bad):
        with pytest.raises(ValueError):
            checks.Duration(threshold=bad, min_hours=1)
        with pytest.raises(ValueError):
            checks.Duration(threshold=1, min_hours=bad)

    def test_negative_rejected(self):
        with pytest.raises(ValueError):
            checks.Duration(threshold=1, min_hours=-1)
        with pytest.raises(ValueError):
            checks.IntervalCount(threshold=1, min_count=-1)

    def test_missing_required(self):
        with pytest.raises(ValueError):
            checks.Duration(threshold=1)


class TestIrradiance:
    def test_default_reads_front_poa(self):
        data = make_data()
        data["front_col"] = 100.0
        cd = FakeCapData(data, dict(ROLES))
        res = checks.Duration(threshold=400, min_hours=0).evaluate(cd, ctx(cd))
        assert res.detail["hours"].iloc[0] == 0.0

    def test_falls_back_to_poa(self):
        cd = FakeCapData(make_data(), {"poa": "poa_col"})
        assert checks.resolve_irradiance(cd, "front_poa") == "poa_col"

    def test_poa_role(self, cd):
        assert checks.resolve_irradiance(cd, "poa") == "poa_col"

    def test_raw_column(self, cd):
        assert checks.resolve_irradiance(cd, "raw_col") == "raw_col"

    def test_unknown(self, cd):
        with pytest.raises(ValueError, match="nope"):
            checks.resolve_irradiance(cd, "nope")


class TestSelection:
    def test_no_rows(self, cd):
        context = checks.CheckContext(dt_hours=0.25, pr_rows=cd.data.index[:0])
        res = checks.Duration(threshold=400, min_hours=1).evaluate(cd, context)
        assert res.passed is False
        assert res.summary == "no data in the selected rows"

    def test_pr_rows_none_falls_back_to_filtered(self):
        data = make_data()
        cd = FakeCapData(data, dict(ROLES), filtered=data.iloc[:4])
        context = checks.CheckContext(dt_hours=0.25)
        res = checks.Duration(threshold=400, min_hours=0).evaluate(cd, context)
        assert res.detail["hours"].iloc[0] == 1.0

    def test_filtered_mode(self):
        data = make_data()
        cd = FakeCapData(data, dict(ROLES), filtered=data.iloc[:8])
        chk = checks.Duration(threshold=400, min_hours=0, data="filtered")
        assert chk.evaluate(cd, ctx(cd)).detail["hours"].iloc[0] == 2.0

    def test_unfiltered_period(self):
        index = pd.date_range("2025-06-01", "2025-06-05 23:45", freq="15min")
        data = pd.DataFrame({"front_col": 500.0, "poa_col": 500.0}, index=index)
        kept = data.loc["2025-06-02 10:00":"2025-06-03 08:00"]
        cd = FakeCapData(data, dict(ROLES), filtered=kept)
        chk = checks.IntervalCount(threshold=0, min_count=0, data="unfiltered_period")
        # fallback: whole days 2 and 3
        sel = checks.select_rows(cd, chk, ctx(cd))
        assert sel.index.min() == pd.Timestamp("2025-06-02 00:00")
        assert sel.index.max() == pd.Timestamp("2025-06-03 23:45")
        # context.period beats the fallback
        period = (pd.Timestamp("2025-06-04 12:00"), pd.Timestamp("2025-06-05 01:00"))
        sel = checks.select_rows(cd, chk, ctx(cd, period=period))
        assert sel.index.min() == pd.Timestamp("2025-06-04 00:00")
        assert sel.index.max() == pd.Timestamp("2025-06-05 23:45")
        # own start/end beats context.period
        chk.start, chk.end = "2025-06-01", "2025-06-01"
        sel = checks.select_rows(cd, chk, ctx(cd, period=period))
        assert len(sel) == 96
        assert sel.index.min() == pd.Timestamp("2025-06-01 00:00")


class TestConfig:
    def test_round_trip_omits_defaults(self):
        chk = checks.Duration(threshold=400, min_hours=30)
        assert chk.to_config() == {
            "type": "Duration",
            "threshold": 400,
            "min_hours": 30,
        }
        assert checks.check_from_config(chk.to_config()).to_config() == chk.to_config()

    def test_round_trip_non_defaults(self):
        chk = checks.IntervalCount(
            threshold=400,
            min_count=5,
            irr="poa",
            data="unfiltered_period",
            start="2025-06-01",
            end="2025-06-02",
            custom_name="enough",
        )
        config = chk.to_config()
        assert config["irr"] == "poa" and config["custom_name"] == "enough"
        rebuilt = checks.check_from_config(config)
        assert rebuilt.to_config() == config
        assert rebuilt.label == "enough"

    def test_unknown_type_hint(self):
        with pytest.raises(ValueError, match="Did you mean 'Duration'"):
            checks.check_from_config({"type": "Duraton"})

    def test_strict_attrs(self):
        chk = checks.Duration(threshold=400, min_hours=30)
        with pytest.raises(AttributeError):
            chk.min_hour = 3
