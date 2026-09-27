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
        with pytest.warns(UserWarning, match="programmatic-only"):
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
        with pytest.warns(UserWarning, match="programmatic-only"):
            sub = build_loaded().to_mapping()
        tst = CapTest.from_mapping(
            sub, base_dir=tmp_path, meas_loader=meas_loader, sim_loader=sim_loader
        )
        assert tst._load_snapshots["meas"]["keys"]["meas_path"] == "meas.csv"
        with pytest.warns(UserWarning, match="programmatic-only"):
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
            def __getattribute__(self, name):
                # The class keeps a valid __qualname__; only the instance
                # lookup the snapshot performs fails.
                if name == "__qualname__":
                    raise RuntimeError("no name for you")
                return object.__getattribute__(self, name)

            def __call__(self, path, **kwargs):
                return build_meas_default()

        # Labelled on the instance, so the label check passes and the
        # failure comes from the implementation-name lookup.
        weird = loader_id("test.weird")(Weird())
        tst = build_loaded(meas_loader=weird)
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


class TestRunFingerprint:
    def test_untouched_run_matches(self):
        tst = build_loaded()
        filter_and_run(tst)
        assert tst.run_fingerprint == tst.mapping_fingerprint()
        assert tst.run_fingerprint_error is None
        assert len(tst.run_fingerprint) == 64

    def test_from_mapping_untouched_matches(self, tmp_path):
        with pytest.warns(UserWarning, match="programmatic-only"):
            sub = build_loaded().to_mapping()
        tst = CapTest.from_mapping(
            sub, base_dir=tmp_path, meas_loader=meas_loader, sim_loader=sim_loader
        )
        filter_and_run(tst)
        assert tst.run_fingerprint == tst.mapping_fingerprint()

    def test_abbreviated_prep_matches(self):
        tst = build_loaded(
            meas_prep=[{"type": "Scale", "group": "real_pwr_mtr", "factor": 1.0}]
        )
        filter_and_run(tst)
        assert tst.run_fingerprint == tst.mapping_fingerprint()

    def test_load_setting_edit_without_reload_mismatches(self):
        tst = build_loaded(meas_load_kwargs={"period": {"start_day": "2019-01-01"}})
        filter_and_run(tst)
        tst.meas_load_kwargs = {"period": {"start_day": "2019-02-01"}}
        tst.run_test()
        assert tst.run_fingerprint is not None
        assert tst.run_fingerprint != tst.mapping_fingerprint()
        tst.reload("meas", verbose=False)
        tst.run_test()
        assert tst.run_fingerprint == tst.mapping_fingerprint()

    def test_override_edit_after_run_mismatches(self):
        tst = build_loaded()
        filter_and_run(tst)
        tst.reg_cols_meas = {"power": {"group": "real_pwr_mtr", "agg": "mean"}}
        assert tst.run_fingerprint != tst.mapping_fingerprint()

    def test_interactive_prep_after_load_mismatches(self):
        tst = build_loaded()
        tst.meas.run_prep([{"type": "Scale", "group": "real_pwr_mtr", "factor": 1.0}])
        filter_and_run(tst)
        assert tst.run_fingerprint != tst.mapping_fingerprint()

    def test_check_pvalues_run_has_no_fingerprint(self):
        tst = build_loaded()
        results = filter_and_run(tst, check_pvalues=True)
        assert tst.last_results is results
        assert tst.run_fingerprint is None
        assert "check_pvalues=True" in tst.run_fingerprint_error

    def test_nondefault_pval_has_no_fingerprint(self):
        tst = build_loaded()
        filter_and_run(tst, pval=1.0)
        assert tst.run_fingerprint is None
        assert "pval" in tst.run_fingerprint_error

    def test_default_pval_passed_explicitly_fingerprints(self):
        tst = build_loaded()
        filter_and_run(tst, pval=0.05)
        assert tst.run_fingerprint == tst.mapping_fingerprint()

    def test_auto_wrap_off_has_no_fingerprint(self):
        tst = build_loaded(auto_wrap_sim=False)
        filter_and_run(tst)
        assert tst.run_fingerprint is None
        assert "auto_wrap_sim" in tst.run_fingerprint_error

    def test_last_results_is_the_returned_object(self):
        tst = build_loaded()
        results = filter_and_run(tst)
        assert tst.last_results is results
        tst.run_test(side="meas")
        assert tst.last_results is None

    def test_reload_invalidates_run(self):
        tst = build_loaded()
        filter_and_run(tst)
        tst.reload("meas", verbose=False)  # same path, maybe new bytes
        assert tst.run_fingerprint is None
        assert tst.last_results is None
        assert "reloaded" in tst.run_fingerprint_error
        tst.run_test()
        assert tst.run_fingerprint == tst.mapping_fingerprint()

    def test_reload_failure_partway_still_clears(self, monkeypatch):
        tst = build_loaded()
        filter_and_run(tst)

        def broken_replay_prep(side):
            raise RuntimeError("boom")

        monkeypatch.setattr(tst, "_replay_prep", broken_replay_prep)
        with pytest.raises(RuntimeError):
            tst.reload("meas", verbose=False)
        assert tst.last_results is None
        assert tst.run_fingerprint is None

    def test_fingerprint_failure_is_recorded_not_raised(self, monkeypatch):
        import captest.captest as cc

        def broken(obj):
            raise RuntimeError("encoder exploded")

        tst = build_loaded()
        monkeypatch.setattr(cc, "canonical_json", broken)
        results = filter_and_run(tst)
        assert results is not None
        assert tst.run_fingerprint is None
        assert "RuntimeError" in tst.run_fingerprint_error

    def test_single_side_run_clears(self):
        tst = build_loaded()
        filter_and_run(tst)
        tst.run_test(side="meas")
        assert tst.run_fingerprint is None
        assert "single-side" in tst.run_fingerprint_error

    def test_failed_run_clears(self, monkeypatch):
        tst = build_loaded()
        filter_and_run(tst)

        def boom(**kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(tst, "captest_results", boom)
        with pytest.raises(RuntimeError):
            tst.run_test()
        assert tst.run_fingerprint is None
        assert "did not complete" in tst.run_fingerprint_error

    def test_prebuilt_capdata(self, ct_default):
        filter_and_run(ct_default)
        assert ct_default.run_fingerprint is None
        assert "prebuilt CapData" in ct_default.run_fingerprint_error

    def test_undeclared_loader(self):
        tst = build_loaded(meas_loader=undeclared_meas_loader)
        results = filter_and_run(tst)
        assert results is not None
        assert tst.run_fingerprint is None
        assert "without a loader_id" in tst.run_fingerprint_error

    def test_side_replaced_after_load(self):
        tst = build_loaded()
        tst.meas = build_meas_default()
        # A direct assignment bypasses reload()'s own setup() call; a fresh
        # CapData has no regression_cols wired until setup() runs, same as
        # any real workflow that swaps ``tst.meas`` outside ``reload()``.
        tst.setup(verbose=False, side="meas")
        filter_and_run(tst)
        assert tst.run_fingerprint is None
        assert "replaced" in tst.run_fingerprint_error

    def test_uncanonical_mapping_never_raises(self):
        import pandas as pd

        tst = build_loaded(meas_load_kwargs={"when": pd.Timestamp("2019-01-01")})
        results = filter_and_run(tst)
        assert results is not None
        assert tst.run_fingerprint is None
        assert "cannot be canonicalised" in tst.run_fingerprint_error

    def test_loader_id_is_not_hashed(self):
        a = build_loaded()
        filter_and_run(a)

        @loader_id("other.meas")
        def other(path, **kwargs):
            return build_meas_default()

        b = build_loaded(meas_loader=other)
        filter_and_run(b)
        assert a.run_fingerprint == b.run_fingerprint
        with pytest.warns(UserWarning, match="programmatic-only"):
            mapping_str = str(a.to_mapping())
        assert "loader_id" not in mapping_str


class TestCopiedLoaderLabels:
    """A ``loader_id`` label counts only on the object it was put on."""

    @staticmethod
    def _assert_unlabelled(tst):
        assert tst.load_provenance["meas"] is None
        assert tst.loader_implementations["meas"] is None
        assert "without a loader_id" in tst._load_snapshots["meas"]["reason"]
        filter_and_run(tst)
        assert tst.run_fingerprint is None
        assert "without a loader_id" in tst.run_fingerprint_error

    def test_decorator_marks_its_target(self):
        def f(path):
            return path

        loader_id("x.y")(f)
        assert f._loader_id_target is f

    def test_functools_wraps_wrapper_is_unlabelled(self):
        import functools

        @functools.wraps(meas_loader)
        def wrapper(path, **kwargs):
            return build_meas_default()

        assert wrapper.loader_id == "test.meas"  # the copied label
        self._assert_unlabelled(build_loaded(meas_loader=wrapper))

    def test_lru_cache_wrapper_is_unlabelled(self):
        import functools

        cached = functools.lru_cache(meas_loader)
        assert cached.loader_id == "test.meas"  # the copied label
        self._assert_unlabelled(build_loaded(meas_loader=cached))

    def test_wrapper_decorated_itself_is_accepted(self):
        import functools

        @loader_id("test.wrapped_meas")
        @functools.wraps(meas_loader)
        def wrapper(path, **kwargs):
            return build_meas_default()

        tst = build_loaded(meas_loader=wrapper)
        assert tst.load_provenance["meas"] == "test.wrapped_meas"
        assert tst.loader_implementations["meas"] is not None
        filter_and_run(tst)
        assert tst.run_fingerprint == tst.mapping_fingerprint()

    def test_labelled_callable_instance_is_accepted(self):
        class Loader:
            def __call__(self, path, **kwargs):
                return build_meas_default()

        inst = loader_id("test.instance_meas")(Loader())
        tst = build_loaded(meas_loader=inst)
        assert tst.load_provenance["meas"] == "test.instance_meas"

    def test_class_level_label_is_unlabelled(self):
        class Loader:
            loader_id = "test.class_meas"

            def __call__(self, path, **kwargs):
                return build_meas_default()

        self._assert_unlabelled(build_loaded(meas_loader=Loader()))

    def test_bound_method_of_labelled_function_is_accepted(self):
        class Source:
            @loader_id("test.method_meas")
            def load(self, path, **kwargs):
                return build_meas_default()

        tst = build_loaded(meas_loader=Source().load)
        assert tst.load_provenance["meas"] == "test.method_meas"


class TestSideReplacedAfterRun:
    def test_replacing_a_side_after_the_run_clears_the_run(self):
        tst = build_loaded()
        filter_and_run(tst)
        assert tst.run_fingerprint == tst.mapping_fingerprint()
        tst.meas = tst.meas.copy()
        assert tst.run_fingerprint is None
        assert tst.last_results is None
        assert "replaced" in tst.run_fingerprint_error

    def test_replacing_a_side_clears_unfingerprinted_results(self):
        tst = build_loaded()
        filter_and_run(tst, pval=1.0)
        assert tst.last_results is not None
        tst.sim = tst.sim.copy()
        assert tst.last_results is None

    def test_putting_the_same_side_back_restores_the_run(self):
        tst = build_loaded()
        results = filter_and_run(tst)
        original = tst.meas
        tst.meas = original.copy()
        tst.meas = original
        assert tst.last_results is results
        assert tst.run_fingerprint == tst.mapping_fingerprint()


class TestReloadBaseDir:
    def test_reload_resolves_relative_path_against_base_dir(
        self, tmp_path, monkeypatch
    ):
        calls = []
        prebuilt = build_meas_default()  # the fixture data path is CWD-relative

        @loader_id("test.recording_meas")
        def recording(path, **kwargs):
            calls.append(str(path))
            return prebuilt.copy()

        with pytest.warns(UserWarning, match="programmatic-only"):
            sub = build_loaded().to_mapping()
        base = tmp_path / "proj"
        base.mkdir()
        tst = CapTest.from_mapping(
            sub, base_dir=base, meas_loader=recording, sim_loader=sim_loader
        )
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        tst.reload("meas", verbose=False)
        assert calls == [str(base / "meas.csv")] * 2
        assert tst._load_snapshots["meas"]["keys"]["meas_path"] == "meas.csv"
        filter_and_run(tst)
        assert tst.run_fingerprint == tst.mapping_fingerprint()

    def test_reload_keeps_cwd_resolution_without_base_dir(self):
        calls = []

        @loader_id("test.recording_meas")
        def recording(path, **kwargs):
            calls.append(str(path))
            return build_meas_default()

        tst = build_loaded(meas_loader=recording)
        tst.reload("meas", verbose=False)
        assert calls == ["meas.csv", "meas.csv"]


class TestReloadValidationKeepsRun:
    def test_no_stored_path_error_leaves_the_run(self, ct_default):
        filter_and_run(ct_default)
        results = ct_default.last_results
        assert results is not None
        with pytest.raises(ValueError, match="no stored data path"):
            ct_default.reload("meas", verbose=False)
        assert ct_default.last_results is results

    def test_bad_side_error_leaves_the_run(self):
        tst = build_loaded()
        results = filter_and_run(tst)
        with pytest.raises(ValueError):
            tst.reload("both", verbose=False)
        assert tst.last_results is results
        assert tst.run_fingerprint == tst.mapping_fingerprint()
