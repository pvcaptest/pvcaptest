"""Write ``tests/data/setup_oracles/<preset>.json`` for every preset.

Run once on the code that is being replaced:

    uv run python -m tests.tools.capture_setup_oracles

Re-run only when a preset's numbers are meant to change.
"""

import json
from pathlib import Path

from captest.captest import TEST_SETUPS
from tests.setup_fixtures import PRESET_FIXTURES, build_captest, snapshot

OUT_DIR = Path("tests/data/setup_oracles")


def main():
    """Capture and write the oracle snapshot for every shipped preset."""
    missing = set(TEST_SETUPS) - set(PRESET_FIXTURES)
    if missing:
        raise SystemExit(f"no fixture mapping for presets: {sorted(missing)}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for preset in sorted(TEST_SETUPS):
        tst = build_captest(preset)
        path = OUT_DIR / f"{preset}.json"
        path.write_text(json.dumps(snapshot(tst), indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
