"""Every shipped preset reproduces the numbers captured before the
document migration (``tests/tools/capture_setup_oracles.py``)."""

import json
from pathlib import Path

import numpy as np
import pytest

from captest.captest import pr_setups, regression_setups
from tests.setup_fixtures import build_captest, snapshot, snapshot_pr

ORACLES = Path("tests/data/setup_oracles")


@pytest.mark.parametrize("preset", regression_setups())
def test_preset_reproduces_oracle(preset):
    expected = json.loads((ORACLES / f"{preset}.json").read_text())
    actual = snapshot(build_captest(preset))
    for side in ("meas", "sim"):
        exp, act = expected[side], actual[side]
        assert set(act["regression_cols"]) == set(exp["regression_cols"])
        for var, e in exp["regression_cols"].items():
            a = act["regression_cols"][var]
            np.testing.assert_allclose(a["sum"], e["sum"], rtol=1e-9)
            np.testing.assert_allclose(a["mean"], e["mean"], rtol=1e-9)
        for key in ("params", "pvalues"):
            assert set(act[key]) == set(exp[key])
            for term, value in exp[key].items():
                np.testing.assert_allclose(act[key][term], value, rtol=1e-9)


@pytest.mark.parametrize("preset", pr_setups())
def test_pr_preset_reproduces_oracle(preset):
    expected = json.loads((ORACLES / f"{preset}.json").read_text())
    actual = snapshot_pr(build_captest(preset))
    assert set(actual) == set(expected)
    for side in set(expected) - {"reference"}:
        assert set(actual[side]) == set(expected[side])
        for role, e in expected[side].items():
            assert actual[side][role]["column"] == e["column"]
            np.testing.assert_allclose(actual[side][role]["sum"], e["sum"], rtol=1e-9)
            np.testing.assert_allclose(actual[side][role]["mean"], e["mean"], rtol=1e-9)
    for key, value in expected["reference"].items():
        if value is None:
            assert actual["reference"][key] is None
        else:
            np.testing.assert_allclose(actual["reference"][key], value, rtol=1e-9)


def test_oracle_files_are_pairwise_distinct():
    """No two presets' captured oracles are byte-identical.

    Guards against fixture data that happens not to exercise the difference
    between two presets (e.g. a ``*_rear_shade_sim`` / ``*_rear_shade_meas``
    pair): if a later change swapped their trees, an identical pair of
    oracles would let ``test_preset_reproduces_oracle`` pass anyway.
    """
    texts = {
        preset: (ORACLES / f"{preset}.json").read_text()
        for preset in (*regression_setups(), *pr_setups())
    }
    seen = {}
    for preset, text in texts.items():
        assert text not in seen, f"{preset} oracle is identical to {seen[text]}"
        seen[text] = preset


@pytest.mark.parametrize("preset", regression_setups())
def test_preset_fixture_regressions_are_well_conditioned(preset):
    """Each preset's fixture data gives a full-rank, well-conditioned fit.

    A rank-deficient design (e.g. rear POA an exact multiple of front POA)
    has no unique coefficients: statsmodels' pinv then returns whichever
    solution the platform's LAPACK rounding selects, and the oracle drifts
    between Linux, macOS and Windows.
    """
    tst = build_captest(preset)
    for side in ("meas", "sim"):
        cd = getattr(tst, side)
        cd.fit_regression(filter=False, summary=False)
        exog = cd.regression_results.model.exog
        assert np.linalg.matrix_rank(exog) == exog.shape[1], side
        assert np.linalg.cond(exog) < 1e8, side
