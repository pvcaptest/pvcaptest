"""Check that basic features work.

Catch cases where e.g. files are missing so the import doesn't work. It is
recommended to check that e.g. assets are included.
"""

import sys


def test_smoke():
    """Smoke test to verify basic package functionality."""
    # Test that the package can be imported
    import captest

    # Test that version is available
    assert hasattr(captest, "__version__")
    assert isinstance(captest.__version__, str)
    assert captest.__version__ != "unknown"

    # Test that main modules can be imported
    from captest import capdata, util, prtest, columngroups, io, plotting  # noqa: F401

    # Test that key classes/functions are available
    assert hasattr(capdata, "CapData")
    assert hasattr(io, "load_data")
    assert hasattr(io, "load_pvsyst")
    assert hasattr(io, "DataLoader")
    assert hasattr(columngroups, "ColumnGroups")
    assert hasattr(plotting, "plot")

    # Test that the package exports are accessible
    assert hasattr(captest, "load_data")
    assert hasattr(captest, "load_pvsyst")
    assert hasattr(captest, "DataLoader")

    print("Smoke test succeeded")


if __name__ == "__main__":
    try:
        test_smoke()
        sys.exit(0)
    except Exception as e:
        print(f"Smoke test failed: {e}")
        raise
