from captest import (
    capdata as capdata,
    util as util,
    prtest as prtest,
    columngroups as columngroups,
    io as io,
    plotting as plotting,
    calcparams as calcparams,
    captest as captest,
)

from captest.io import (
    load_data as load_data,
    load_pvsyst as load_pvsyst,
    DataLoader as DataLoader,
)

from captest.captest import (
    CapTest as CapTest,
    TEST_SETUPS as TEST_SETUPS,
    load_config as load_config,
)

try:
    from importlib.metadata import version

    __version__ = version("captest")
except Exception:
    __version__ = "unknown"
