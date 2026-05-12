import pytest
import numpy as np
import pandas as pd
from captest import capdata as pvc
from captest import util
from captest import columngroups as cg
from captest import CapTest, load_pvsyst


@pytest.fixture
def meas():
    """Create an instance of CapData with example data loaded."""
    meas = pvc.CapData("meas")
    meas.data = pd.read_csv(
        "./tests/data/example_measured_data.csv",
        index_col=0,
        parse_dates=True,
    )
    meas.data_filtered = meas.data.copy(deep=True)
    meas.column_groups = cg.ColumnGroups(
        util.read_json("./tests/data/example_measured_data_column_groups.json")
    )
    meas.set_regression_cols(
        power="meter_power", poa="irr_poa_pyran", t_amb="temp_amb", w_vel="wind"
    )
    return meas


@pytest.fixture
def meas_groups_with_one_tag():
    """Create an instance of CapData with example data loaded."""
    meas = pvc.CapData("meas")
    meas.data = pd.read_csv(
        "./tests/data/example_measured_data.csv",
        index_col=0,
        parse_dates=True,
    )
    meas.data_filtered = meas.data.copy(deep=True)
    meas.column_groups = cg.ColumnGroups(
        util.read_json("./tests/data/example_measured_data_column_groups.json")
    )
    meas.set_regression_cols(
        power="meter_power", poa="irr_poa_pyran", t_amb="temp_amb", w_vel="wind"
    )
    meas.drop_cols(["met2_ghi_pyranometer", "met2_poa_pyranometer"])
    return meas


@pytest.fixture
def location_and_system():
    """Create a dictionary with a nested dictionary for location and system."""
    loc_sys = {
        "location": {
            "latitude": 30.274583,
            "longitude": -97.740352,
            "altitude": 500,
            "tz": "America/Chicago",
        },
        "system": {
            "surface_tilt": 20,
            "surface_azimuth": 180,
            "albedo": 0.2,
        },
    }
    return loc_sys


@pytest.fixture
def nrel():
    nrel = pvc.CapData("nrel")
    nrel.data = pd.read_csv("./tests/data/nrel_data.csv", index_col=0, parse_dates=True)
    nrel.data_filtered = nrel.data.copy()
    nrel.column_groups = {
        "irr-ghi-": [
            "Global CMP22 (vent/cor) [W/m^2]",
        ],
        "irr-poa-": [
            "POA 40-South CMP11 [W/m^2]",
        ],
        "temp--": [
            "Deck Dry Bulb Temp [deg C]",
        ],
        "wind--": [
            "Avg Wind Speed @ 19ft [m/s]",
        ],
    }
    nrel.regression_cols = {
        "power": "",
        "poa": "irr-poa-",
        "t_amb": "temp--",
        "w_vel": "wind--",
    }
    return nrel


@pytest.fixture
def pvsyst():
    # load pvsyst csv file
    df = pd.read_csv(
        "./tests/data/pvsyst_example_HourlyRes_2.CSV",
        skiprows=9,
        encoding="latin1",
    ).iloc[1:, :]
    df["Timestamp"] = pd.to_datetime(df["date"], format="%m/%d/%y %H:%M")
    df = df.set_index("Timestamp", drop=True)
    df = df.drop(columns=["date"]).astype(np.float64)
    df.rename(columns={"T Amb": "T_Amb"}, inplace=True)
    # set pvsyst DataFrame to CapData data attribute
    pvsyst = pvc.CapData("pvsyst")
    pvsyst.data = df
    pvsyst.data_filtered = pvsyst.data.copy()
    pvsyst.column_groups = {
        "irr-poa-": ["GlobInc"],
        "shade--": ["FShdBm"],
        "index--": ["index"],
        "wind--": ["WindVel"],
        "-inv-": ["EOutInv"],
        "pvsyt_losses--": ["IL Pmax", "IL Pmin", "IL Vmax", "IL Vmin"],
        "temp-amb-": ["T_Amb"],
        "irr-ghi-": ["GlobHor"],
        "temp-mod-": ["TArray"],
        "real_pwr--": ["E_Grid"],
    }
    pvsyst.regression_cols = {
        "power": "real_pwr--",
        "poa": "irr-poa-",
        "t_amb": "temp-amb-",
        "w_vel": "wind--",
    }
    return pvsyst


@pytest.fixture
def pvsyst_irr_filter(pvsyst):
    pvsyst.filter_irr(200, 800)
    pvsyst.tolerance = "+/- 5"
    return pvsyst


@pytest.fixture
def nrel_clear_sky(nrel):
    """Modeled clear sky data was created using the pvlib fixed tilt clear sky
    models with the following parameters:
         loc = {
            'latitude': 39.742,
            'longitude': -105.18,
            'altitude': 1828.8,
            'tz': 'Etc/GMT+7'
        }
        sys = {'surface_tilt': 40, 'surface_azimuth': 180, 'albedo': 0.2}
    """
    clear_sky = pd.read_csv(
        "./tests/data/nrel_data_modelled_csky.csv", index_col=0, parse_dates=True
    )
    nrel.data = pd.concat([nrel.data, clear_sky], axis=1)
    nrel.data_filtered = nrel.data.copy()
    nrel.column_groups["irr-poa-clear_sky"] = ["poa_mod_csky"]
    nrel.column_groups["irr-ghi-clear_sky"] = ["ghi_mod_csky"]
    return nrel


@pytest.fixture
def capdata_irr():
    """
    Creates a CapData instance with dummy irradiance data"""
    start_time = pd.Timestamp("2023-10-01 12:00")
    end_time = start_time + pd.Timedelta(minutes=15)
    datetime_index = pd.date_range(start_time, end_time, freq="1min")

    np.random.seed(42)
    random_values = np.random.uniform(876, 900, size=(len(datetime_index), 4))

    # for i in range(1, len(random_values)):
    #     random_values[i] = np.clip(
    #         random_values[i] + np.random.randint(-25, 26), 850, 900)

    df = pd.DataFrame(
        random_values, index=datetime_index, columns=["poa1", "poa2", "poa3", "poa4"]
    )

    cd = pvc.CapData("cd")
    cd.data = df
    cd.data_filtered = df.copy()
    cd.column_groups = {"poa": ["poa1", "poa2", "poa3", "poa4"]}
    return cd


# -- CapTest fixtures -----------------------------------------------------


@pytest.fixture
def meas_cd_default():
    """Minimal measured CapData sourced from the existing example data.

    Column groups are renamed so the shipped ``e2848_default`` preset matches
    without extra overrides: ``real_pwr_mtr``, ``irr_poa``, ``temp_amb``,
    ``wind_speed``. A synthetic ``irr_rpoa`` group is added (scaled fraction
    of the POA sensors) so the ``bifi_e2848_etotal`` and ``bifi_power_tc``
    presets also resolve against this fixture without additional wiring.
    """
    cd = pvc.CapData("meas")
    df = pd.read_csv(
        "./tests/data/example_measured_data.csv",
        index_col=0,
        parse_dates=True,
    )
    # Synthesize rear-POA irradiance as a fraction of the front-POA sensors.
    # Real rpoa is typically 10-20% of front POA for bifacial sites; 15%
    # keeps the test fixture in a realistic range without requiring a new
    # data file.
    df["met1_rpoa"] = df["met1_poa_pyranometer"] * 0.15
    df["met2_rpoa"] = df["met2_poa_pyranometer"] * 0.15
    cd.data = df
    cd.data_filtered = cd.data.copy(deep=True)
    cd.column_groups = cg.ColumnGroups(
        {
            "real_pwr_mtr": ["meter_power"],
            "irr_poa": ["met1_poa_pyranometer", "met2_poa_pyranometer"],
            "irr_rpoa": ["met1_rpoa", "met2_rpoa"],
            "temp_amb": ["met1_amb_temp", "met2_amb_temp"],
            "wind_speed": ["met1_windspeed", "met2_windspeed"],
        }
    )
    return cd


@pytest.fixture
def sim_cd_default():
    """Minimal modeled CapData loaded from the shipped PVsyst example.

    Synthetic ``GlobBak`` and ``BackShd`` columns are added so presets that
    rely on ``rpoa_pvsyst(globbak=..., backshd=...)`` (``bifi_e2848_etotal``,
    ``bifi_power_tc``) resolve without needing a new PVsyst export.
    """
    cd = load_pvsyst(path="./tests/data/pvsyst_example_HourlyRes_2.CSV")
    cd.data["GlobBak"] = cd.data["GlobInc"] * 0.15
    cd.data["BackShd"] = 0.0
    cd.data_filtered = cd.data.copy(deep=True)
    return cd


@pytest.fixture
def ct_default(meas_cd_default, sim_cd_default):
    """CapTest bound to both default CapData fixtures with setup() run."""
    return CapTest.from_params(
        test_setup="e2848_default",
        meas=meas_cd_default,
        sim=sim_cd_default,
        ac_nameplate=6_000_000,
        test_tolerance="- 4",
    )


@pytest.fixture
def ct_etotal(meas_cd_default, sim_cd_default):
    """CapTest for the bifi_e2848_etotal preset with setup() run."""
    return CapTest.from_params(
        test_setup="bifi_e2848_etotal",
        meas=meas_cd_default,
        sim=sim_cd_default,
        ac_nameplate=6_000_000,
        bifaciality=0.15,
        test_tolerance="- 4",
    )


@pytest.fixture
def ct_bifi_power_tc(meas_cd_default, sim_cd_default):
    """CapTest for the bifi_power_tc preset with setup() run."""
    return CapTest.from_params(
        test_setup="bifi_power_tc",
        meas=meas_cd_default,
        sim=sim_cd_default,
        ac_nameplate=6_000_000,
        bifaciality=0.15,
        power_temp_coeff=-0.32,
        base_temp=25,
        test_tolerance="- 4",
    )


@pytest.fixture
def meas_cd_spec_corrected(meas_cd_default):
    """Measured CapData extended with humidity, pressure, and a site dict.

    Needed to exercise the ``e2848_spec_corrected_poa`` preset, whose meas
    tree requires ``humidity`` and ``pressure`` column groups plus
    ``cd.site`` for apparent-zenith calculations.
    """
    cd = meas_cd_default
    rng = np.random.default_rng(seed=42)
    n = cd.data.shape[0]
    cd.data["met1_humidity"] = np.clip(rng.normal(60.0, 10.0, n), 5.0, 95.0)
    cd.data["met2_humidity"] = np.clip(rng.normal(60.0, 10.0, n), 5.0, 95.0)
    cd.data["met1_pressure"] = rng.normal(1013.0, 3.0, n)
    cd.data["met2_pressure"] = rng.normal(1013.0, 3.0, n)
    cd.data_filtered = cd.data.copy(deep=True)
    groups = dict(cd.column_groups)
    groups["humidity"] = ["met1_humidity", "met2_humidity"]
    groups["pressure"] = ["met1_pressure", "met2_pressure"]
    cd.column_groups = cg.ColumnGroups(groups)
    cd.site = {
        "loc": {
            "latitude": 33.0,
            "longitude": -99.5,
            "altitude": 500,
            "tz": "America/Chicago",
        },
        "sys": {
            "surface_tilt": 20,
            "surface_azimuth": 180,
            "albedo": 0.2,
        },
    }
    return cd


@pytest.fixture
def sim_cd_spec_corrected(sim_cd_default):
    """PVsyst CapData extended with a synthetic ``PrecWat`` column.

    The shipped PVsyst fixture does not include a PrecWat column; the
    ``e2848_spec_corrected_poa`` sim tree references it directly. A
    realistic-looking synthetic value (0.5-3 cm expressed in meters, matching
    PVsyst units) is added so the preset can resolve end-to-end in tests.
    """
    cd = sim_cd_default
    rng = np.random.default_rng(seed=43)
    n = cd.data.shape[0]
    cd.data["PrecWat"] = rng.uniform(0.005, 0.03, n)  # meters
    cd.data_filtered = cd.data.copy(deep=True)
    return cd


@pytest.fixture
def captest_yaml(tmp_path):
    """Minimal yaml file exercising CapTest.from_yaml.

    Writes a file at ``tmp_path / 'captest.yaml'`` with no data paths so
    from_yaml does not try to load anything. Returns the path.
    """
    yaml_text = (
        "captest:\n"
        "  test_setup: e2848_default\n"
        "  ac_nameplate: 6000000\n"
        "  test_tolerance: '- 4'\n"
        "  min_irr: 400\n"
        "  max_irr: 1400\n"
        "  fshdbm: 1.0\n"
        "  hrs_req: 12.5\n"
        "  bifaciality: 0.0\n"
    )
    path = tmp_path / "captest.yaml"
    path.write_text(yaml_text)
    return path


@pytest.fixture
def cd_nested_col_groups():
    """
    Creates a CapData instance with from the column groups defined in
    ./tests/data/example_measured_data_column_groups_subgroups.json.

    Creates dummy timeseries data at 5min intervals for 2 days with one column for
    each column in the column groups file.

    Used to test CapData.agg_sensors when agg_map is nested. E.g. aggregating groups
    of rear POA irradiance sensors and then aggregating the results.
    """
    # Create CapData instance
    cd = pvc.CapData("nested_groups")

    # Load column groups from json file
    cd.column_groups = cg.ColumnGroups(
        util.read_json(
            "./tests/data/example_measured_data_column_groups_subgroups.json"
        )
    )

    # Create datetime index for 2 days at 5min intervals
    start_time = pd.Timestamp("2023-01-01")
    end_time = start_time + pd.Timedelta(days=2)
    datetime_index = pd.date_range(start_time, end_time, freq="5min")

    # Get all column names from column groups
    all_columns = []
    for columns in cd.column_groups.values():
        all_columns.extend(columns)
    all_columns = list(set(all_columns))

    # Create dummy data for each column
    np.random.seed(42)  # For reproducibility
    data = {}

    # Generate data for each column with realistic irradiance patterns
    for col in all_columns:
        # Base pattern: daily cycle with peak around noon
        time_of_day = (datetime_index.hour * 60 + datetime_index.minute) / (24 * 60)
        base_pattern = 1000 * np.sin(np.pi * time_of_day) ** 2

        # Add some random variation (±5% of the base value)
        noise = np.random.normal(0, 0.05 * base_pattern)

        # Set nighttime values to 0 and clip negative values
        base_pattern = np.where(
            (datetime_index.hour >= 6) & (datetime_index.hour <= 18),
            base_pattern + noise,
            0,
        )
        data[col] = np.clip(base_pattern, 0, 1200)

    # Create DataFrame with the generated data
    cd.data = pd.DataFrame(data, index=datetime_index)
    cd.data_filtered = cd.data.copy()

    return cd
