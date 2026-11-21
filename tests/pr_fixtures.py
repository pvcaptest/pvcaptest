"""Synthetic performance-ratio test pairs shared by the PR test modules."""

import numpy as np
import pandas as pd

from captest import captest as ct, columngroups as cg
from captest.calcparams import EMP_HEAT_COEFF
from captest.capdata import CapData
from captest.setup import PrSetup

DC_NAMEPLATE = 1.0e6
POWER_TEMP_COEFF = -0.4
MEAS_PR_FACTOR = 0.8
SIM_PR_FACTOR = 0.78
FIXED_T_REF = 40.0


def pr_doc(*, temp_corrected=False, availability_role=False):
    """A PR setup document with ``power`` and ``poa`` (and ``cell_temp``)."""
    side = {"power": {"column": "p"}, "poa": {"column": "g"}}
    if temp_corrected:
        side["cell_temp"] = {"column": "tc"}
    meas = dict(side)
    if availability_role:
        meas["availability"] = {"column": "avail"}
    return {
        "kind": "pr",
        "name": "pr_synth",
        "description": "synthetic PR test setup",
        "temp_corrected": temp_corrected,
        "meas": {"reg_cols": meas},
        "sim": {"reg_cols": dict(side)},
    }


def _expected_power_w(g, tc, t_ref):
    factor = 1.0
    if t_ref is not None:
        factor = 1 + POWER_TEMP_COEFF / 100 * (tc - t_ref)
    return DC_NAMEPLATE * g / 1000.0 * factor


def make_pr_pair(
    monkeypatch,
    *,
    temp_corrected=False,
    reference_source="derived",
    granularity="single",
    start="2024-06-01",
    periods=96,
    freq="15min",
    gen_t_ref=FIXED_T_REF,
    availability_role=False,
    **params,
):
    """Build a set-up PR ``CapTest`` on synthetic data.

    Both sides share one irradiance and cell temperature series. Measured
    power is ``MEAS_PR_FACTOR`` times the expected power at the generation
    temperature (the plain expected power for a plain test), so the measured
    PR is ``MEAS_PR_FACTOR`` when the reference temperature is that
    temperature: ``gen_t_ref`` for a fixed reference, the irradiance-weighted
    cell temperature of all rows for a derived one (until a filter changes
    the rows). The modeled side's power is ``SIM_PR_FACTOR`` times the plain
    expected power. Power is in W; ``dc_nameplate`` is ``DC_NAMEPLATE``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Registers the ``pr_synth`` setup in ``TEST_SETUPS``.
    temp_corrected : bool, default False
        Use a temperature-corrected setup (``cell_temp`` role, coefficient
        ``POWER_TEMP_COEFF``).
    reference_source : {"derived", "fixed"}, default "derived"
        A fixed reference builds no ``sim``.
    granularity : {"single", "monthly"}, default "single"
        Reference granularity.
    start, periods, freq : str, int, str
        The measured and modeled index.
    gen_t_ref : float, default FIXED_T_REF
        Temperature the measured power is generated at with a fixed
        reference.
    availability_role : bool, default False
        Add a measured ``availability`` role (column ``avail``, all 1.0).
    **params
        Further ``CapTest`` parameters (override the defaults set here).

    Returns
    -------
    captest.CapTest
        Set up; filters are not applied.
    """
    monkeypatch.setitem(
        ct.TEST_SETUPS,
        "pr_synth",
        PrSetup.model_validate(
            pr_doc(temp_corrected=temp_corrected, availability_role=availability_role)
        ),
    )
    index = pd.date_range(start, periods=periods, freq=freq)
    rng = np.random.default_rng(1)
    g = rng.uniform(300.0, 900.0, len(index))
    tc = rng.uniform(20.0, 60.0, len(index))
    if reference_source == "derived":
        gen_t_ref = float((g * tc).sum() / g.sum())
    if not temp_corrected:
        gen_t_ref = None
    meas_frame = pd.DataFrame(
        {
            "g": g,
            "tc": tc,
            "p": MEAS_PR_FACTOR * _expected_power_w(g, tc, gen_t_ref),
        },
        index=index,
    )
    if availability_role:
        meas_frame["avail"] = 1.0
    meas = CapData("meas")
    meas.data = meas_frame
    meas.column_groups = cg.ColumnGroups({})
    sim = None
    if reference_source == "derived":
        sim = CapData("sim")
        sim.data = pd.DataFrame(
            {"g": g, "tc": tc, "p": SIM_PR_FACTOR * _expected_power_w(g, tc, None)},
            index=index,
        )
        sim.column_groups = cg.ColumnGroups({})
    kwargs = {
        "test_setup": "pr_synth",
        "dc_nameplate": DC_NAMEPLATE,
        "reference_source": reference_source,
        "reference_granularity": granularity,
        "test_tolerance": "- 5",
    }
    if temp_corrected:
        kwargs["power_temp_coeff"] = POWER_TEMP_COEFF
    kwargs.update(params)
    tst = ct.CapTest(meas=meas, sim=sim, **kwargs)
    tst.setup(verbose=False)
    return tst


def bell_irradiance(index, peak=1000.0):
    """Daily half-sine irradiance, zero from 18:00 to 06:00.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Timestamps to evaluate.
    peak : float, default 1000.0
        Irradiance at solar noon (12:00) in W/m².

    Returns
    -------
    pd.Series
        ``peak * sin(pi * (hour - 6) / 12)`` between 06:00 and 18:00, else 0.
    """
    hour = index.hour + index.minute / 60.0
    shape = np.sin(np.pi * (hour - 6.0) / 12.0)
    return pd.Series(peak * np.where((hour >= 6.0) & (hour <= 18.0), shape, 0.0), index)


def _by_month(value, index):
    """A scalar or a ``{month: value}`` mapping as a Series aligned to ``index``."""
    if isinstance(value, dict):
        missing = sorted(set(index.month) - set(value))
        if missing:
            raise ValueError(f"No value for month(s) {missing} in {value!r}.")
        return pd.Series(index.month.map(value).to_numpy(float), index=index)
    return pd.Series(float(value), index=index)


def make_meas_pr(
    index,
    *,
    dc_nameplate,
    pr,
    t_ref,
    coeff,
    bifacial=False,
    availability=None,
    bifaciality=0.7,
    temp_delta=None,
    peak=1000.0,
):
    """Measured ``CapData`` whose PR is exactly ``pr`` by construction.

    Power (W) is ``pr`` times the expected power of the PR definition:
    ``dc_nameplate * irradiance / 1000 * (1 + coeff / 100 * (cell - t_ref))
    * availability``, so the availability-weighted measured PR over any rows is
    ``pr``. The irradiance is the front POA, or with ``bifacial`` the total
    effective irradiance ``poa + bifaciality * rpoa``; the cell temperature is
    the back-of-module temperature plus ``poa / 1000 * temp_delta``.

    Column groups are ``real_pwr_mtr`` (``meter_w``), ``irr_poa`` (``poa_1``,
    ``poa_2``), ``temp_bom`` (``bom_1``) and, with ``bifacial``, ``irr_rpoa``
    (``rpoa_1``, 15 % of the front POA). With ``availability`` an ``avail``
    column is added.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Measured timestamps.
    dc_nameplate : float
        DC nameplate capacity in W.
    pr : float
        Measured PR to build in.
    t_ref : float or dict or None
        Reference temperature in °C, or ``{month: value}``; None for a plain
        (not temperature-corrected) PR.
    coeff : float
        Power temperature coefficient in %/°C.
    bifacial : bool, default False
        Add a rear irradiance column and use the total effective irradiance.
    availability : float or pd.Series, optional
        Per-interval availability factor applied to the power and stored in
        the ``avail`` column.
        A Series must have an index equal to ``index``.
    bifaciality : float, default 0.7
        Bifaciality used in the total effective irradiance.
    temp_delta : float, optional
        Cell-to-back-of-module temperature difference at 1000 W/m²; defaults
        to the open-rack glass/cell/poly value of the cell temperature
        calculation.
    peak : float, default 1000.0
        Noon front irradiance in W/m².

    Returns
    -------
    captest.capdata.CapData
        With ``data`` and ``column_groups`` set.

    Raises
    ------
    ValueError
        If ``availability`` is a Series not indexed like ``index``, or a
        ``t_ref`` mapping lacks a month of ``index``.
    """
    if temp_delta is None:
        temp_delta = EMP_HEAT_COEFF["open_rack"]["glass_cell_poly"]["del_tcnd"]
    poa = bell_irradiance(index, peak)
    rpoa = 0.15 * poa
    bom = 20.0 + 0.02 * poa + 2.0 * (index.day_of_year % 3)
    cell = bom + poa / 1000.0 * temp_delta
    irradiance = poa + bifaciality * rpoa if bifacial else poa
    expected_w = dc_nameplate * irradiance / 1000.0
    if t_ref is not None:
        expected_w = expected_w * (1 + coeff / 100.0 * (cell - _by_month(t_ref, index)))
    frame = pd.DataFrame({"poa_1": poa, "poa_2": poa, "bom_1": bom}, index=index)
    groups = {
        "real_pwr_mtr": ["meter_w"],
        "irr_poa": ["poa_1", "poa_2"],
        "temp_bom": ["bom_1"],
    }
    if bifacial:
        frame["rpoa_1"] = rpoa
        groups["irr_rpoa"] = ["rpoa_1"]
    if availability is not None:
        if isinstance(availability, pd.Series) and not availability.index.equals(index):
            raise ValueError("availability must be indexed like index.")
        frame["avail"] = availability
        expected_w = expected_w * frame["avail"]
    frame["meter_w"] = pr * expected_w
    cd = CapData("meas")
    cd.data = frame
    cd.column_groups = cg.ColumnGroups(groups)
    return cd


def make_sim_pr(
    year_index,
    *,
    pr_by_month,
    dc_nameplate=DC_NAMEPLATE,
    bifacial=False,
    bifaciality=0.7,
    t_array=45.0,
    peak=950.0,
):
    """Modeled (PVsyst-shaped) ``CapData`` with a known PR in each month.

    ``E_Grid`` (W) is ``pr_by_month[month]`` times ``dc_nameplate`` times the
    effective irradiance over 1000, so the plain PR of the rows of a month is
    that month's value. Columns are ``GlobInc``, ``TArray``, ``E_Grid`` and,
    with ``bifacial``, ``GlobBak`` (20 % of ``GlobInc``) and ``BackShd``
    (-5 %), whose sum is the rear irradiance.

    Parameters
    ----------
    year_index : pd.DatetimeIndex
        Modeled timestamps (typically hourly).
    pr_by_month : dict
        ``{month number: PR}`` for every month of the index.
    dc_nameplate : float, default DC_NAMEPLATE
        DC nameplate capacity in W.
    bifacial : bool, default False
        Add the rear irradiance columns and use the total effective
        irradiance for ``E_Grid``.
    bifaciality : float, default 0.7
        Bifaciality used in the total effective irradiance.
    t_array : float or dict, default 45.0
        Modeled cell temperature in °C, or ``{month: value}``.
    peak : float, default 950.0
        Noon front irradiance in W/m².

    Returns
    -------
    captest.capdata.CapData
        With ``data`` and ``column_groups`` set.

    Raises
    ------
    ValueError
        If ``pr_by_month`` or a ``t_array`` mapping lacks a month of
        ``year_index``.
    """
    glob_inc = bell_irradiance(year_index, peak)
    frame = pd.DataFrame(
        {"GlobInc": glob_inc, "TArray": _by_month(t_array, year_index)},
        index=year_index,
    )
    irradiance = glob_inc
    if bifacial:
        frame["GlobBak"] = 0.20 * glob_inc
        frame["BackShd"] = -0.05 * glob_inc
        irradiance = glob_inc + bifaciality * (frame["GlobBak"] + frame["BackShd"])
    pr = _by_month(pr_by_month, year_index)
    frame["E_Grid"] = pr * dc_nameplate * irradiance / 1000.0
    cd = CapData("sim")
    cd.data = frame
    cd.column_groups = cg.ColumnGroups({})
    return cd
