import pytest
import copy
import numpy as np
import pandas as pd

from captest import util

ix = pd.date_range(start="1/1/2021 12:00", freq="h", periods=3)

ix_5min = pd.date_range(start="1/1/2021 12:00", freq="5min", periods=3)


class TestGetCommonTimestep:
    def test_output_type_str(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=True)
        assert isinstance(time_step, str)

    def test_output_type_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=False)
        assert isinstance(time_step, np.float64)

    def test_hours_string(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=True)
        assert time_step == "1H"

    def test_hours_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="h", string_output=False)
        assert time_step == 1.0

    def test_minutes_numeric(self):
        df = pd.DataFrame({"a": [1, 2, 4]}, index=ix)
        time_step = util.get_common_timestep(df, units="m", string_output=False)
        assert time_step == 60.0

    def test_mixed_intervals(self):
        df = pd.DataFrame(
            {"a": np.ones(120)},
            index=pd.date_range(start="1/1/21", freq="1min", periods=120),
        )
        assert df.index.is_monotonic_increasing
        print(df)
        df_gaps = pd.concat(
            [
                df.loc["1/1/21":"1/1/21 00:10", :],
                df.loc["1/1/21 00:15":"1/1/21 00:29", :],
                df.loc["1/1/21 00:31":, :],
            ]
        )
        assert df_gaps.shape[0] < df.shape[0]
        time_step = util.get_common_timestep(df_gaps, units="m", string_output=False)
        assert time_step == 1
        time_step = util.get_common_timestep(df_gaps, units="m", string_output=True)
        assert time_step == "1min"


@pytest.fixture
def reindex_dfs():
    df1 = pd.DataFrame(
        {"a": np.full(4, 5)},
        index=pd.date_range(start="1/1/21", end="1/1/21 00:15", freq="5min"),
    )
    df2 = pd.DataFrame(
        {"a": np.full(4, 5)},
        index=pd.date_range(start="1/1/21 00:30", end="1/1/21 00:45", freq="5min"),
    )
    return (df1, df2)


class TestReindexDatetime:
    def test_adds_missing_intervals(self, reindex_dfs):
        """Check that missing intervals in the index are added to the dataframe."""
        df1, df2 = reindex_dfs
        df = pd.concat([df1, df2])
        (df_reindexed, missing_intervals, freq_str) = util.reindex_datetime(df)
        assert df_reindexed.shape[0] == 10

        df = pd.concat([df2, df1])  # reverse order, check sorting
        (df_reindexed, missing_intervals, str) = util.reindex_datetime(df)
        assert df_reindexed.shape[0] == 10

    def test_drops_duplicate_indices(self):
        """
        Check that duplicate indices are dropped before reindexing.

        Use Nov 3, 2025 which has the 1AM hour repeated due to daylight saving time.
        """
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6]},
            index=pd.to_datetime(
                [
                    "2025-11-03 00:00",
                    "2025-11-03 01:00",
                    "2025-11-03 01:00",
                    "2025-11-03 02:00",
                    "2025-11-03 03:00",
                    "2025-11-03 04:00",
                ]
            ),
        )
        with pytest.warns(UserWarning):
            (df_reindexed, missing_intervals, freq_str) = util.reindex_datetime(df)
        assert df_reindexed.index.is_unique
        assert df_reindexed.shape[0] == 5


@pytest.fixture
def nested_calc_dict():
    """Create a nested dictionary for testing update_by_path."""

    class DummyCapData(object):
        def __init__(self):
            self.data = pd.DataFrame()

        def test_func1(self, **kwargs):
            self.test_func1_kwargs = kwargs
            self.data["test_func1"] = np.full(10, 1)

        def test_func2(self, **kwargs):
            self.test_func2_kwargs = kwargs
            self.data["test_func2"] = np.full(10, 2)

        def test_func3(self, **kwargs):
            self.test_func3_kwargs = kwargs
            self.data["test_func3"] = np.full(10, 3)

        def test_func4(self, **kwargs):
            self.test_func4_kwargs = kwargs
            self.data["test_func4"] = np.full(10, 4)

        def agg_group(self, group_id, agg_func, **kwargs):
            self.agg_group_kwargs = kwargs
            col_name = group_id + "_" + agg_func
            self.data[col_name] = np.full(10, 5)
            return col_name

        def custom_param(self, func, *args, **kwargs):
            setattr(self, f"{func.__name__}_custom_kwargs", kwargs.copy())
            func(self, **kwargs)

    dummy_cd = DummyCapData()
    dummy_cd.column_groups = {
        "real_pwr_mtr": ["metered_power_kw"],
        "irr_poa": ["pyran1", "pyran2"],
        "temp_amb": ["temp_amb1", "temp_amb2"],
        "wind_speed": ["wind_speed1", "wind_speed2"],
        "irr_rpoa": ["irr_rpoa1", "irr_rpoa2"],
    }

    test_dict = {
        "power_tc": (
            DummyCapData.test_func1,
            {
                "power": "real_pwr_mtr",
                "cell_temp": (
                    DummyCapData.test_func2,
                    {
                        "poa": ("irr_poa", "mean"),
                        "bom": (
                            DummyCapData.test_func3,
                            {
                                "poa": ("irr_poa", "mean"),
                                "temp_amb": ("temp_amb", "mean"),
                                "wind_speed": ("wind_speed", "mean"),
                            },
                        ),
                    },
                ),
            },
        ),
        "irr_total": (
            DummyCapData.test_func4,
            {
                "poa": ("irr_poa", "mean"),
                "rpoa": ("irr_rpoa", "mean"),
            },
        ),
    }
    return (dummy_cd, test_dict)


class TestUpdateByPath:
    """Test the update_by_path function."""

    def test_update_by_path_pass_new_value(self, nested_calc_dict):
        dummy_cd, test_dict = nested_calc_dict
        updated_dict = util.update_by_path(
            test_dict, ["power_tc", 1, "cell_temp", 1, "bom"], new_value="temp_bom"
        )
        assert updated_dict["power_tc"][1]["cell_temp"][1]["bom"] == "temp_bom"

    def test_update_by_path_convert_callable(self, nested_calc_dict):
        dummy_cd, test_dict = nested_calc_dict
        updated_dict = util.update_by_path(
            test_dict,
            ["power_tc", 1, "cell_temp", 1, "bom"],
            new_value=None,
            convert_callable=True,
        )
        assert updated_dict["power_tc"][1]["cell_temp"][1]["bom"] == "test_func3"

    def test_update_by_path_convert_callable_with_new_value(self, nested_calc_dict):
        dummy_cd, test_dict = nested_calc_dict
        updated_dict = util.update_by_path(
            test_dict,
            ["power_tc", 1, "cell_temp", 1, "bom"],
            new_value="temp_bom",
            convert_callable=True,
        )
        assert updated_dict["power_tc"][1]["cell_temp"][1]["bom"] == "temp_bom"

    def test_update_by_path_convert_callable_short_path(self, nested_calc_dict):
        dummy_cd, test_dict = nested_calc_dict
        updated_dict = util.update_by_path(
            test_dict, ["power_tc"], new_value=None, convert_callable=True
        )
        assert updated_dict["power_tc"] == "test_func1"


class TestProcessRegCols:
    """Test the process_reg_cols function."""

    def test_scalar_literal_kwargs_pass_through(self, nested_calc_dict):
        """Numeric literals inside calc-tuple kwargs are forwarded unchanged.

        This locks in the behavior that calcparams functions like
        ``scale(data, col, factor)`` receive their ``factor`` scalar from
        the TEST_SETUPS nested tuple rather than only from the function
        signature default.
        """
        dummy_cd, _ = nested_calc_dict
        reg_cols = {
            "scaled": (
                type(dummy_cd).test_func1,
                {
                    "power": "real_pwr_mtr",
                    "factor": 100,
                    "enabled": True,
                    "offset": 1.5,
                },
            ),
        }
        dummy_cd.regression_cols = copy.deepcopy(reg_cols)
        util.process_reg_cols(reg_cols, cd=dummy_cd)
        # Scalars survived the walk and were passed to custom_param.
        assert dummy_cd.test_func1_kwargs["factor"] == 100
        assert dummy_cd.test_func1_kwargs["enabled"] is True
        assert dummy_cd.test_func1_kwargs["offset"] == 1.5
        # The calc tuple was replaced by the function name at the top level.
        assert reg_cols["scaled"] == "test_func1"

    def test_modifies_original_calc_params(self, nested_calc_dict):
        dummy_cd, test_dict = nested_calc_dict
        dummy_cd.regression_cols = copy.deepcopy(test_dict)
        util.process_reg_cols(test_dict, cd=dummy_cd)
        expected_modified_reg_cols = {
            "power_tc": "test_func1",
            "irr_total": "test_func4",
        }
        # Check that methods of the DummyCapData instance are called with the
        # correct kwargs in the correct order based on columns added to the
        # data DataFrame attribute and the kwargs attributes
        assert isinstance(dummy_cd.data, pd.DataFrame)
        print(dummy_cd.data)
        print(dummy_cd.regression_cols)
        assert dummy_cd.data.shape == (10, 8)
        expected_columns = pd.Index(
            [
                "irr_poa_mean",
                "temp_amb_mean",
                "wind_speed_mean",
                "test_func3",
                "test_func2",
                "test_func1",
                "irr_rpoa_mean",
                "test_func4",
            ]
        )
        assert dummy_cd.data.columns.equals(expected_columns)
        assert dummy_cd.test_func1_kwargs == {
            "power": "metered_power_kw",
            "cell_temp": "test_func2",
            "verbose": True,
        }
        assert dummy_cd.test_func2_kwargs == {
            "poa": "irr_poa_mean",
            "bom": "test_func3",
            "verbose": True,
        }
        assert dummy_cd.test_func3_kwargs == {
            "poa": "irr_poa_mean",
            "temp_amb": "temp_amb_mean",
            "wind_speed": "wind_speed_mean",
            "verbose": True,
        }
        # check that reg_cols is rolled up all the way correctly
        for k, v in expected_modified_reg_cols.items():
            assert k in test_dict
            assert v == test_dict[k]


class TestParseRegressionFormula:
    def test_astm(self):
        lhs, rhs = util.parse_regression_formula(
            "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"
        )
        assert lhs == ["power"]
        assert rhs == ["poa", "t_amb", "w_vel"]

    def test_power_temp_corr_poa(self):
        lhs, rhs = util.parse_regression_formula("power_tc ~ poa")
        assert lhs == ["power_tc"]
        assert rhs == ["poa"]

    def test_power_temp_corr_poa_intercept(self):
        lhs, rhs = util.parse_regression_formula("power_tc ~ poa - 1")
        assert lhs == ["power_tc"]
        assert rhs == ["poa"]

    def test_power_temp_corr_poa_rpoa(self):
        lhs, rhs = util.parse_regression_formula("power_tc ~ poa + rpoa")
        assert lhs == ["power_tc"]
        assert rhs == ["poa", "rpoa"]

    def test_outboard_poa_total(self):
        lhs, rhs = util.parse_regression_formula(
            "power ~ poa_total + I(poa_total * fpoa) + I(poa_total * rpoa) +"
            "I(poa_total * t_amb) + I(poa_total * w_vel)"
        )
        assert lhs == ["power"]
        assert rhs == ["poa_total", "fpoa", "rpoa", "t_amb", "w_vel"]

    def test_outboard_poa_rpoa_separate(self):
        lhs, rhs = util.parse_regression_formula(
            "power ~ (poa + rpoa) * (1 + poa + rpoa + t_amb + w_vel - 1)"
        )
        assert lhs == ["power"]
        assert rhs == ["poa", "rpoa", "t_amb", "w_vel"]


class TestDetectSolarNoon:
    """Tests for util.detect_solar_noon."""

    def _solar_like_index(self, days=3, freq="15min"):
        return pd.date_range("2024-06-01 00:00", periods=24 * 4 * days, freq=freq)

    def _peak_at(self, idx, peak_hour, peak_minute):
        """Build a clear-sky-ish series whose per-clock-time mean peaks at
        ``peak_hour:peak_minute`` regardless of date."""
        minutes_of_day = idx.hour * 60 + idx.minute
        peak_minutes = peak_hour * 60 + peak_minute
        # Cosine bell centered on peak_minutes (period = 1 day).
        deltas = (minutes_of_day - peak_minutes) / (24 * 60)
        values = np.maximum(np.cos(2 * np.pi * deltas), 0) ** 2
        return values

    def test_returns_clock_time_of_idxmax(self):
        idx = self._solar_like_index()
        ghi = self._peak_at(idx, 13, 0)
        df = pd.DataFrame({"ghi_mod_csky": ghi}, index=idx)
        assert util.detect_solar_noon(df) == "13:00"

    def test_zero_padded_hour_and_minute(self):
        # 1-minute frequency so the 09:05 peak aligns with an actual sample.
        idx = pd.date_range("2024-06-01 00:00", periods=24 * 60 * 2, freq="1min")
        ghi = self._peak_at(idx, 9, 5)
        df = pd.DataFrame({"ghi_mod_csky": ghi}, index=idx)
        assert util.detect_solar_noon(df) == "09:05"

    def test_missing_column_warns_and_uses_default(self):
        idx = self._solar_like_index()
        df = pd.DataFrame({"other": np.zeros(len(idx))}, index=idx)
        with pytest.warns(UserWarning, match="ghi_mod_csky"):
            result = util.detect_solar_noon(df)
        assert result == "12:30"

    def test_empty_index_warns_and_uses_default(self):
        df = pd.DataFrame({"ghi_mod_csky": []}, index=pd.DatetimeIndex([]))
        with pytest.warns(UserWarning, match="no rows"):
            result = util.detect_solar_noon(df)
        assert result == "12:30"

    def test_custom_default_returned(self):
        df = pd.DataFrame({"other": [1.0]}, index=pd.DatetimeIndex(["2024-01-01"]))
        with pytest.warns(UserWarning):
            result = util.detect_solar_noon(df, default="11:45")
        assert result == "11:45"

    def test_custom_ghi_col(self):
        idx = self._solar_like_index()
        ghi = self._peak_at(idx, 12, 30)
        df = pd.DataFrame({"clear_sky": ghi}, index=idx)
        assert util.detect_solar_noon(df, ghi_col="clear_sky") == "12:30"
