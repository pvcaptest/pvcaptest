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


def test_oracle_files_are_pairwise_distinct():
    """No two presets' captured oracles are byte-identical.

    Guards against fixture data that happens not to exercise the difference
    between two presets (e.g. a ``*_rear_shade_sim`` / ``*_rear_shade_meas``
    pair): if a later change swapped their trees, an identical pair of
    oracles would let ``test_preset_reproduces_oracle`` pass anyway.
    """
    texts = {preset: (ORACLES / f"{preset}.json").read_text() for preset in TEST_SETUPS}
    seen = {}
    for preset, text in texts.items():
        assert text not in seen, f"{preset} oracle is identical to {seen[text]}"
        seen[text] = preset
