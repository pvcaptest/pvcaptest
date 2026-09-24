"""Shipped presets: normalisation, identity, and the stored digests."""

import json
from pathlib import Path

import pytest

from captest import captest as ct
from captest.setup import TestSetup

DIGESTS = Path("tests/data/setup_digests.json")


@pytest.mark.parametrize("preset", sorted(ct.TEST_SETUPS))
def test_normalisation_is_idempotent(preset):
    tsd = ct.TEST_SETUPS[preset]
    assert TestSetup.model_validate(tsd.to_dict()).to_dict() == tsd.to_dict()


@pytest.mark.parametrize("preset", sorted(ct.TEST_SETUPS))
def test_digest_matches_the_stored_value(preset):
    stored = json.loads(DIGESTS.read_text())
    assert ct.TEST_SETUPS[preset].content_digest() == stored[preset], (
        f"{preset} changed; if intended, regenerate tests/data/setup_digests.json"
    )


def test_every_preset_has_a_stored_digest():
    assert set(json.loads(DIGESTS.read_text())) == set(ct.TEST_SETUPS)
