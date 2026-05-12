"""Tests for the scatter-plot pieces added to ``captest.plotting``.

Covers:
- ``add_am_pm_dim`` clock-time tagging and validation.
- ``calc_tc_power_column`` calc-params materialization, idempotency, and
  the missing-column-group error path.
- ``ScatterPlot.view`` shape across split_day / tc_mode / timeseries
  combinations and the ``add_panel + timeseries`` ValueError.
- ``ScatterBifiPowerTc.view`` two-panel shape and the warn-and-ignore
  path for ``tc_power=True``.
- Backward-compat: ``scatter_default(cd)`` returns a single-element
  Layout containing a Scatter.
"""

from __future__ import annotations

import holoviews as hv
import numpy as np
import pandas as pd
import pytest

from captest import captest as ct
from captest import columngroups as cg
from captest import plotting
from captest.capdata import CapData


# --- shared fixtures --------------------------------------------------------


def _build_synthetic_cd():
    """Minimal CapData with morning + afternoon points and resolved cols.

    Used by tests that need a regression-formula-driven scatter without
    going through the full ``CapTest.setup`` flow. The fixture uses 1-min
    spacing across a single day so AM/PM splitting has rows on both
    sides of the boundary.
    """
    idx = pd.date_range("2024-06-01 06:00", periods=24 * 60, freq="1min")
    n = len(idx)
    cd = CapData("synthetic")
    # Realistic-ish increasing irradiance with a midday peak.
    minutes_of_day = idx.hour * 60 + idx.minute
    poa = np.maximum(np.sin(np.pi * (minutes_of_day - 360) / 720), 0) * 1000
    cd.data = pd.DataFrame(
        {
            "power": poa * 5.0,
            "poa": poa,
            "t_amb": np.full(n, 25.0),
            "w_vel": np.full(n, 2.0),
            "ghi_mod_csky": poa,
            "temp_bom": np.full(n, 35.0),
        },
        index=idx,
    )
    cd.data_filtered = cd.data.copy()
    cd.column_groups = cg.ColumnGroups(
        {
            "real_pwr_mtr": ["power"],
            "irr_poa": ["poa"],
            "temp_amb": ["t_amb"],
            "wind_speed": ["w_vel"],
            "temp_bom": ["temp_bom"],
        }
    )
    cd.regression_cols = {
        "power": "power",
        "poa": "poa",
        "t_amb": "t_amb",
        "w_vel": "w_vel",
    }
    cd.regression_formula = (
        "power ~ poa + I(poa * poa) + I(poa * t_amb) + I(poa * w_vel) - 1"
    )
    # Scalars consumed by power_temp_correct via custom_param's auto-fill.
    cd.power_temp_coeff = -0.32
    cd.base_temp = 25
    return cd


@pytest.fixture
def synth_cd():
    return _build_synthetic_cd()


@pytest.fixture
def synth_cd_bifi():
    """Synthetic CapData with a bifi-style ``power ~ poa + rpoa`` formula."""
    cd = _build_synthetic_cd()
    cd.data["rpoa"] = cd.data["poa"] * 0.15
    cd.data_filtered = cd.data.copy()
    cd.regression_cols = {
        "power": "power",
        "poa": "poa",
        "rpoa": "rpoa",
    }
    cd.regression_formula = "power ~ poa + rpoa"
    return cd


# --- add_am_pm_dim ----------------------------------------------------------


class TestAddAmPmDim:
    def test_categorizes_around_boundary(self):
        idx = pd.date_range("2024-06-01 06:00", periods=12, freq="1h")
        df = pd.DataFrame({"value": range(12)}, index=idx)
        out = plotting.add_am_pm_dim(df, "12:30")
        # Hours 06-12 are am, 13-17 are pm.
        am_count = (out["period"] == "am").sum()
        pm_count = (out["period"] == "pm").sum()
        assert am_count == 7  # 06, 07, 08, 09, 10, 11, 12
        assert pm_count == 5  # 13, 14, 15, 16, 17

    def test_returns_copy_does_not_mutate_input(self):
        idx = pd.date_range("2024-06-01 09:00", periods=4, freq="1h")
        df = pd.DataFrame({"value": [1, 2, 3, 4]}, index=idx)
        out = plotting.add_am_pm_dim(df, "12:00")
        assert "period" not in df.columns
        assert "period" in out.columns

    def test_accepts_unpadded_hour(self):
        idx = pd.date_range("2024-06-01 08:00", periods=2, freq="2h")
        df = pd.DataFrame({"value": [1, 2]}, index=idx)
        out = plotting.add_am_pm_dim(df, "9:00")
        assert out.iloc[0]["period"] == "am"
        assert out.iloc[1]["period"] == "pm"

    def test_invalid_format_raises(self):
        idx = pd.date_range("2024-06-01 09:00", periods=2, freq="1h")
        df = pd.DataFrame({"value": [1, 2]}, index=idx)
        with pytest.raises(ValueError, match="HH:MM"):
            plotting.add_am_pm_dim(df, "noon")

    def test_invalid_hour_raises(self):
        idx = pd.date_range("2024-06-01 09:00", periods=2, freq="1h")
        df = pd.DataFrame({"value": [1, 2]}, index=idx)
        with pytest.raises(ValueError, match="out of range"):
            plotting.add_am_pm_dim(df, "25:00")

    def test_invalid_minute_raises(self):
        idx = pd.date_range("2024-06-01 09:00", periods=2, freq="1h")
        df = pd.DataFrame({"value": [1, 2]}, index=idx)
        with pytest.raises(ValueError, match="out of range"):
            plotting.add_am_pm_dim(df, "12:60")

    def test_non_string_split_time_raises(self):
        idx = pd.date_range("2024-06-01 09:00", periods=2, freq="1h")
        df = pd.DataFrame({"value": [1, 2]}, index=idx)
        with pytest.raises(ValueError, match="HH:MM"):
            plotting.add_am_pm_dim(df, 1230)


# --- calc_tc_power_column ---------------------------------------------------


class TestCalcTcPowerColumn:
    def _calc_spec(self):
        # Mirror DEFAULT_TC_POWER_CALC but force the synthetic fixture's
        # column-group ids so it works without depending on default
        # column-group inference.
        from captest.calcparams import cell_temp, power_temp_correct

        return {
            "power": (
                power_temp_correct,
                {
                    "power": ("real_pwr_mtr", "sum"),
                    "cell_temp": (
                        cell_temp,
                        {
                            "poa": ("irr_poa", "mean"),
                            "bom": ("temp_bom", "mean"),
                        },
                    ),
                },
            ),
        }

    def test_writes_to_data_and_data_filtered(self, synth_cd):
        col = plotting.calc_tc_power_column(synth_cd, self._calc_spec())
        assert col == plotting.TC_POWER_PLOT_COL
        assert col in synth_cd.data.columns
        assert col in synth_cd.data_filtered.columns

    def test_does_not_mutate_regression_state(self, synth_cd):
        before_cols = dict(synth_cd.regression_cols)
        before_formula = synth_cd.regression_formula
        plotting.calc_tc_power_column(synth_cd, self._calc_spec())
        assert synth_cd.regression_cols == before_cols
        assert synth_cd.regression_formula == before_formula

    def test_temperature_corrected_power_differs_from_input_power(self, synth_cd):
        """Verify the plot-only temperature-corrected power is not raw power."""
        input_power = synth_cd.data["power"].copy()
        col = plotting.calc_tc_power_column(synth_cd, self._calc_spec())
        assert not np.allclose(synth_cd.data[col], input_power)
        assert not np.allclose(synth_cd.data_filtered[col], input_power)

    def test_idempotent_short_circuit(self, synth_cd):
        plotting.calc_tc_power_column(synth_cd, self._calc_spec())
        first_values = synth_cd.data[plotting.TC_POWER_PLOT_COL].copy()
        # Replace the column with sentinel values to detect a recomputation.
        synth_cd.data[plotting.TC_POWER_PLOT_COL] = -1.0
        plotting.calc_tc_power_column(synth_cd, self._calc_spec())
        assert (synth_cd.data[plotting.TC_POWER_PLOT_COL] == -1.0).all()
        # Sanity: the original first-pass values were not all -1.
        assert not (first_values == -1.0).all()

    def test_force_recompute(self, synth_cd):
        plotting.calc_tc_power_column(synth_cd, self._calc_spec())
        synth_cd.data[plotting.TC_POWER_PLOT_COL] = -1.0
        plotting.calc_tc_power_column(synth_cd, self._calc_spec(), force_recompute=True)
        assert not (synth_cd.data[plotting.TC_POWER_PLOT_COL] == -1.0).all()

    def test_missing_column_group_raises(self, synth_cd):
        # Drop ``temp_bom`` so the spec references a missing group.
        groups = dict(synth_cd.column_groups)
        groups.pop("temp_bom")
        synth_cd.column_groups = cg.ColumnGroups(groups)
        synth_cd.data = synth_cd.data.drop(columns=["temp_bom"])
        synth_cd.data_filtered = synth_cd.data.copy()
        with pytest.raises(KeyError):
            plotting.calc_tc_power_column(synth_cd, self._calc_spec())

    def test_rejects_spec_without_top_level_power_calculation(self, synth_cd):
        """Verify raw-power-only calc specs are rejected instead of copied."""
        from captest.calcparams import cell_temp

        calc_spec = {
            "power": ("real_pwr_mtr", "sum"),
            "cell_temp": (
                cell_temp,
                {
                    "poa": ("irr_poa", "mean"),
                    "bom": ("temp_bom", "mean"),
                },
            ),
        }
        with pytest.raises(ValueError, match="top-level 'power' calculation tuple"):
            plotting.calc_tc_power_column(synth_cd, calc_spec)


# --- ScatterPlot.view -------------------------------------------------------


class TestScatterPlotView:
    def test_returns_layout_with_single_scatter(self, synth_cd):
        layout = plotting.ScatterPlot(cd=synth_cd).view()
        assert isinstance(layout, hv.Layout)
        principal = list(layout)[0]
        assert isinstance(principal, hv.Scatter)

    def test_split_day_principal_is_overlay(self, synth_cd):
        layout = plotting.ScatterPlot(cd=synth_cd, split_day=True).view()
        principal = list(layout)[0]
        # The unified split principal is an Overlay of the real Scatter
        # and two NaN-coord decoy Scatters that supply legend entries.
        assert isinstance(principal, hv.Overlay)

    def test_split_day_unified_principal_has_one_real_scatter_and_two_decoys(
        self, synth_cd
    ):
        """Split-day principal has one real CDS + two NaN decoys for legend."""
        layout = plotting.ScatterPlot(cd=synth_cd, split_day=True).view()
        principal = list(layout)[0]
        elements = list(principal)
        scatters = [el for el in elements if isinstance(el, hv.Scatter)]
        assert len(scatters) == 3
        # Identify the decoys: their data is a single NaN row.
        non_decoy = [s for s in scatters if len(s) > 1]
        decoys = [s for s in scatters if len(s) == 1]
        assert len(non_decoy) == 1
        assert len(decoys) == 2
        # The single real scatter holds every filtered row.
        assert len(non_decoy[0]) == len(synth_cd.data_filtered)
        # Decoys carry am/pm labels for the bokeh legend.
        labels = sorted(d.label for d in decoys)
        assert labels == ["am", "pm"]

    def test_split_day_with_explicit_split_time(self, synth_cd):
        # Explicit override should not consult detect_solar_noon.
        layout = plotting.ScatterPlot(
            cd=synth_cd, split_day=True, split_time="11:00"
        ).view()
        principal = list(layout)[0]
        assert isinstance(principal, hv.Overlay)

    def test_tc_power_replace_uses_tc_y(self, synth_cd):
        sp = plotting.ScatterPlot(
            cd=synth_cd,
            tc_power=True,
            tc_mode="replace",
            tc_power_calc=TestCalcTcPowerColumn()._calc_spec(),
        )
        layout = sp.view()
        principal = list(layout)[0]
        assert isinstance(principal, hv.Scatter)
        # The tc-power column should now exist on cd.data.
        assert plotting.TC_POWER_PLOT_COL in synth_cd.data.columns
        # vdims include the tc column rather than raw 'power'.
        vdims = [d.name for d in principal.vdims]
        assert plotting.TC_POWER_PLOT_COL in vdims

    def test_tc_power_add_panel_two_panels(self, synth_cd):
        sp = plotting.ScatterPlot(
            cd=synth_cd,
            tc_power=True,
            tc_mode="add_panel",
            tc_power_calc=TestCalcTcPowerColumn()._calc_spec(),
        )
        layout = sp.view()
        assert isinstance(layout, hv.Layout)
        assert len(layout) == 2

    def test_tc_power_overlay_single_panel_overlay(self, synth_cd):
        sp = plotting.ScatterPlot(
            cd=synth_cd,
            tc_power=True,
            tc_mode="overlay",
            tc_power_calc=TestCalcTcPowerColumn()._calc_spec(),
        )
        layout = sp.view()
        assert len(layout) == 1
        principal = list(layout)[0]
        assert isinstance(principal, hv.Overlay)

    def test_add_panel_plus_timeseries_raises(self, synth_cd):
        sp = plotting.ScatterPlot(
            cd=synth_cd,
            tc_power=True,
            tc_mode="add_panel",
            tc_power_calc=TestCalcTcPowerColumn()._calc_spec(),
            timeseries=True,
        )
        with pytest.raises(ValueError, match="add_panel"):
            sp.view()

    def test_tc_overlay_plus_timeseries_raises(self, synth_cd):
        """tc_power + tc_mode='overlay' + timeseries=True is unsupported.

        The linked timeseries panel can only display a single y-series,
        so an overlaid raw + tc-power principal is ambiguous. The error
        must surface at construction time rather than as a bokeh
        KeyError at render time.
        """
        sp = plotting.ScatterPlot(
            cd=synth_cd,
            tc_power=True,
            tc_mode="overlay",
            tc_power_calc=TestCalcTcPowerColumn()._calc_spec(),
            timeseries=True,
        )
        with pytest.raises(ValueError, match="tc_mode='overlay'"):
            sp.view()

    def test_split_day_with_timeseries_renders_without_error(self, synth_cd):
        """split_day=True + timeseries=True must render through bokeh.

        Previously this combination raised ``KeyError: 'source'`` deep
        inside ``DataLinkCallback`` because the principal panel was an
        AM/PM Overlay of two separate ColumnDataSources, leaving no
        top-level ``source`` handle for ``DataLink`` to attach to. The
        unified single-CDS principal makes ``DataLink`` work.
        """
        sp = plotting.ScatterPlot(cd=synth_cd, split_day=True, timeseries=True)
        layout = sp.view()
        # Force bokeh to materialize the plot tree; this is what was
        # raising KeyError before the fix.
        hv.renderer("bokeh").get_plot(layout)
        assert isinstance(layout, hv.Layout)
        assert len(layout) == 2

    def test_timeseries_replace_returns_two_panel_layout(self, synth_cd):
        sp = plotting.ScatterPlot(cd=synth_cd, timeseries=True)
        layout = sp.view()
        assert isinstance(layout, hv.Layout)
        assert len(layout) == 2

    def test_timeseries_overlays_unfiltered_power_as_thin_gray_curve(self, synth_cd):
        """Timeseries panel includes a thin gray Curve of unfiltered power.

        The bottom panel of the layout should be an Overlay containing
        an ``hv.Curve`` of the full ``cd.data`` y-series (rendered
        underneath) and an ``hv.Scatter`` of the filtered y-series
        (rendered on top). The Curve must use a gray color and a thin
        line_width so the filtered scatter remains the visual focus.
        """
        # Drop every other row from the filtered data so the Curve
        # (full data) and the Scatter (filtered data) have different
        # row counts and we can verify which series each plots.
        synth_cd.data_filtered = synth_cd.data.iloc[::2].copy()

        sp = plotting.ScatterPlot(cd=synth_cd, timeseries=True)
        layout = sp.view()

        panels = list(layout)
        assert len(panels) == 2
        timeseries_panel = panels[1]
        assert isinstance(timeseries_panel, hv.Overlay)

        elements = list(timeseries_panel)
        curves = [el for el in elements if isinstance(el, hv.Curve)]
        scatters = [el for el in elements if isinstance(el, hv.Scatter)]
        assert len(curves) == 1
        assert len(scatters) == 1
        curve, scatter = curves[0], scatters[0]

        # Curve is drawn before the scatter so it sits underneath.
        assert elements.index(curve) < elements.index(scatter)

        # Curve plots the unfiltered series; scatter plots the filtered.
        assert len(curve) == len(synth_cd.data)
        assert len(scatter) == len(synth_cd.data_filtered)

        # Both series share the same y-column.
        assert curve.vdims[0].name == scatter.vdims[0].name

        # Curve is styled thin and gray.
        style_kwargs = hv.Store.lookup_options("bokeh", curve, "style").kwargs
        assert style_kwargs.get("color") == "gray"
        assert style_kwargs.get("line_width", 1.0) <= 1.0

    def test_timeseries_curve_resolves_semantic_y_via_regression_cols(self, synth_cd):
        """Curve is built when ``y_col`` is a semantic regression name.

        Real ``CapData`` instances built through ``CapTest`` have
        ``regression_formula`` written in semantic names (e.g.
        ``power ~ poa + ...``) while ``cd.data`` holds the underlying
        aggregated columns (e.g. ``real_pwr_mtr_sum_agg``). The
        timeseries Curve must look up the underlying column through
        ``regression_cols`` rather than expecting the semantic name to
        appear directly on ``cd.data``.
        """
        # Rename underlying data columns so they no longer match the
        # semantic names used by the regression formula.
        synth_cd.data = synth_cd.data.rename(
            columns={
                "power": "real_pwr_mtr_sum_agg",
                "poa": "irr_poa_mean_agg",
                "t_amb": "temp_amb_mean_agg",
                "w_vel": "wind_speed_mean_agg",
            }
        )
        synth_cd.data_filtered = synth_cd.data.copy()
        synth_cd.regression_cols = {
            "power": "real_pwr_mtr_sum_agg",
            "poa": "irr_poa_mean_agg",
            "t_amb": "temp_amb_mean_agg",
            "w_vel": "wind_speed_mean_agg",
        }

        layout = plotting.ScatterPlot(cd=synth_cd, timeseries=True).view()
        timeseries_panel = list(layout)[1]
        assert isinstance(timeseries_panel, hv.Overlay)
        elements = list(timeseries_panel)
        curves = [el for el in elements if isinstance(el, hv.Curve)]
        assert len(curves) == 1
        # Curve plots the full underlying ``real_pwr_mtr_sum_agg`` series
        # but exposes it under the semantic ``power`` vdim so it shares a
        # y-axis with the linked scatter.
        assert len(curves[0]) == len(synth_cd.data)
        assert curves[0].vdims[0].name == "power"

    def test_view_requires_cd(self):
        with pytest.raises(ValueError, match="cd must be set"):
            plotting.ScatterPlot().view()

    def test_tc_power_with_already_tc_regression_warns(self, synth_cd):
        # Simulate the bifi_power_tc shape: regression_cols['power'] is the
        # function name 'power_temp_correct'.
        synth_cd.regression_cols["power"] = "power_temp_correct"
        synth_cd.data["power_temp_correct"] = synth_cd.data["power"]
        synth_cd.data_filtered = synth_cd.data.copy()
        with pytest.warns(UserWarning, match="already targets"):
            plotting.ScatterPlot(
                cd=synth_cd,
                tc_power=True,
                tc_power_calc=TestCalcTcPowerColumn()._calc_spec(),
            ).view()


# --- ScatterBifiPowerTc.view ------------------------------------------------


class TestScatterBifiPowerTcView:
    def test_returns_two_panel_layout(self, synth_cd_bifi):
        layout = plotting.ScatterBifiPowerTc(cd=synth_cd_bifi).view()
        assert isinstance(layout, hv.Layout)
        assert len(layout) == 2

    def test_split_day_two_panel_overlays(self, synth_cd_bifi):
        layout = plotting.ScatterBifiPowerTc(cd=synth_cd_bifi, split_day=True).view()
        assert len(layout) == 2
        for panel in layout:
            assert isinstance(panel, hv.Overlay)

    def test_tc_power_is_warned_and_ignored(self, synth_cd_bifi):
        with pytest.warns(UserWarning, match="ignores tc_power"):
            layout = plotting.ScatterBifiPowerTc(cd=synth_cd_bifi, tc_power=True).view()
        # Layout shape unchanged (still 2 panels, no extra tc-power panel).
        assert len(layout) == 2


# --- backward-compat regression --------------------------------------------


class TestScatterDefaultBackwardCompat:
    def test_returns_single_element_layout_with_scatter(self, synth_cd):
        layout = ct.scatter_default(synth_cd)
        assert isinstance(layout, hv.Layout)
        assert len(layout) == 1
        assert isinstance(list(layout)[0], hv.Scatter)
