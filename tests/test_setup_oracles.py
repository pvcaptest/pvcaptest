"""Every shipped preset reproduces the numbers captured before the
document migration (``tests/tools/capture_setup_oracles.py``)."""

import json
from pathlib import Path

import numpy as np
import pytest

from captest.captest import TEST_SETUPS
from tests.setup_fixtures import build_captest, snapshot

ORACLES = Path("tests/data/setup_oracles")


@pytest.mark.parametrize("preset", sorted(TEST_SETUPS))
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
