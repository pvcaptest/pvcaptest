"""Synthetic performance-ratio test pairs shared by the PR test modules."""

import numpy as np
import pandas as pd

from captest import captest as ct, columngroups as cg
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
