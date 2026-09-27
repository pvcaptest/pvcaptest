"""Load provenance and run fingerprints (setup source-of-truth spec, P0)."""

import pytest

from captest import CapTest
from captest.io import load_data, load_pvsyst, loader_id
from tests.setup_fixtures import build_meas_default, build_sim_default

_SIM_WINDOW = ("1990-10-09", "1990-10-13 23:55:00")


@loader_id("test.meas")
def meas_loader(path, **kwargs):
    return build_meas_default()


@loader_id("test.sim")
def sim_loader(path, **kwargs):
    return build_sim_default()


def undeclared_meas_loader(path, **kwargs):
    return build_meas_default()


def build_loaded(**overrides):
    kwargs = {
        "test_setup": "e2848_default",
        "meas_path": "meas.csv",
        "sim_path": "sim.csv",
        "meas_loader": meas_loader,
        "sim_loader": sim_loader,
        "ac_nameplate": 6_000_000,
        "test_tolerance": "- 4",
    }
    kwargs.update(overrides)
    return CapTest.from_params(**kwargs)


def filter_and_run(tst, **run_kwargs):
    tst.meas.filter_irr(tst.min_irr, tst.max_irr)
    tst.sim.filter_irr(tst.min_irr, tst.max_irr)
    tst.sim.filter_shade(fshdbm=tst.fshdbm)
    tst.sim.filter_time(start=_SIM_WINDOW[0], end=_SIM_WINDOW[1])
    tst.rep_cond()
    return tst.run_test(**run_kwargs)


class TestLoaderId:
    def test_sets_attribute_and_returns_function(self):
        def f(path):
            return path

        assert loader_id("x.y")(f) is f
        assert f.loader_id == "x.y"

    @pytest.mark.parametrize("bad", ["", None, 3])
    def test_rejects_bad_identifier(self, bad):
        with pytest.raises(ValueError):
            loader_id(bad)

    def test_shipped_loaders_carry_ids(self):
        assert load_data.loader_id == "captest.csv_meas"
        assert load_pvsyst.loader_id == "captest.pvsyst"


class TestLoadSnapshots:
    def test_path_load_records_provenance(self):
        tst = build_loaded()
        assert tst.load_provenance == {"meas": "test.meas", "sim": "test.sim"}
        assert tst._load_snapshots["meas"]["keys"] == {"meas_path": "meas.csv"}

    def test_snapshot_uses_to_mapping_normal_form(self):
        tst = build_loaded(
            meas_prep=[{"type": "Scale", "group": "real_pwr_mtr", "factor": 1.0}],
            meas_load_kwargs={"period": {"start_day": "2019-01-01"}},
        )
        keys = tst._load_snapshots["meas"]["keys"]
        mapping = tst.to_mapping()
        for k in ("meas_path", "meas_load_kwargs", "meas_prep"):
            assert keys[k] == mapping[k]
        assert "offset" in keys["meas_prep"][0]  # expanded by to_config()

    def test_snapshot_is_an_independent_copy(self):
        tst = build_loaded(meas_load_kwargs={"period": {"start_day": "2019-01-01"}})
        tst.meas_load_kwargs["period"]["start_day"] = "2020-01-01"
        keys = tst._load_snapshots["meas"]["keys"]
        assert keys["meas_load_kwargs"]["period"]["start_day"] == "2019-01-01"

    def test_undeclared_loader_records_no_keys(self):
        tst = build_loaded(meas_loader=undeclared_meas_loader)
        assert tst.load_provenance["meas"] is None
        snap = tst._load_snapshots["meas"]
        assert snap["keys"] is None
        assert "without a loader_id" in snap["reason"]

    def test_prebuilt_capdata_records_reason(self, ct_default):
        assert ct_default.load_provenance == {"meas": None, "sim": None}
        assert "prebuilt CapData" in ct_default._load_snapshots["meas"]["reason"]

    def test_reload_refreshes_snapshot(self):
        tst = build_loaded()
        tst.reload("sim", path="other.csv", verbose=False)
        assert tst._load_snapshots["sim"]["keys"] == {"sim_path": "other.csv"}
        assert tst._load_snapshots["sim"]["capdata"] is tst.sim

    def test_from_mapping_relative_paths_snapshot_raw_spelling(self, tmp_path):
        sub = build_loaded().to_mapping()
        tst = CapTest.from_mapping(
            sub, base_dir=tmp_path, meas_loader=meas_loader, sim_loader=sim_loader
        )
        assert tst._load_snapshots["meas"]["keys"]["meas_path"] == "meas.csv"
        assert tst.to_mapping()["meas_path"] == "meas.csv"

    def test_replaced_side_has_no_provenance(self):
        tst = build_loaded()
        tst.meas = build_meas_default()
        assert tst.load_provenance["meas"] is None
        assert tst.loader_implementations["meas"] is None

    def test_implementation_is_captured_at_load(self):
        tst = build_loaded()
        expected = f"{meas_loader.__module__}:{meas_loader.__qualname__}"
        assert tst.loader_implementations["meas"] == expected

        @loader_id("test.meas")  # borrows the id
        def impostor(path, **kwargs):
            return build_meas_default()

        tst.meas_loader = impostor  # swapped after the load
        assert tst.loader_implementations["meas"] == expected
        assert tst.load_provenance["meas"] == "test.meas"

    def test_prep_snapshot_is_deep_copied(self, monkeypatch):
        from captest.capdata import CapData

        shared = [
            {
                "type": "Scale",
                "group": "real_pwr_mtr",
                "factor": 1.0,
                "offset": 0.0,
                "extra": {"nested": [1]},
            }
        ]
        monkeypatch.setattr(CapData, "prep_to_config", lambda self: shared)
        tst = build_loaded(
            meas_prep=[{"type": "Scale", "group": "real_pwr_mtr", "factor": 1.0}]
        )
        shared[0]["extra"]["nested"].append(2)
        keys = tst._load_snapshots["meas"]["keys"]
        assert keys["meas_prep"][0]["extra"]["nested"] == [1]

    def test_loader_metadata_failure_is_recorded_not_raised(self):
        class Weird:
            loader_id = "test.weird"

            def __getattribute__(self, name):
                # The class keeps a valid __qualname__; only the instance
                # lookup the snapshot performs fails.
                if name == "__qualname__":
                    raise RuntimeError("no name for you")
                return object.__getattribute__(self, name)

            def __call__(self, path, **kwargs):
                return build_meas_default()

        tst = build_loaded(meas_loader=Weird())
        assert tst.meas is not None
        assert tst.load_provenance["meas"] is None
        assert "RuntimeError" in tst._load_snapshots["meas"]["reason"]

    def test_snapshot_failure_is_recorded_not_raised(self, monkeypatch):
        def broken(self, side):
            raise RuntimeError("cannot serialize")

        monkeypatch.setattr(CapTest, "_side_load_keys", broken)
        tst = build_loaded()  # the load itself succeeds
        assert tst.meas is not None
        snap = tst._load_snapshots["meas"]
        assert snap["keys"] is None
        assert "RuntimeError" in snap["reason"]
