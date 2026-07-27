from captest import (
    calcparams as calcparams,
    capdata as capdata,
    captest as captest,
    clearsky as clearsky,
    columngroups as columngroups,
    io as io,
    plotting as plotting,
    prtest as prtest,
    util as util,
)
from captest.captest import (
    TEST_SETUPS as TEST_SETUPS,
    CapTest as CapTest,
    load_config as load_config,
    test_setups as test_setups,
)
from captest.io import (
    DataLoader as DataLoader,
    load_data as load_data,
    load_pvsyst as load_pvsyst,
)

try:
    from importlib.metadata import version

    __version__ = version("captest")
except Exception:
    __version__ = "unknown"
