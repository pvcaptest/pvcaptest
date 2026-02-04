from captest import (
    capdata as capdata,
    util as util,
    prtest as prtest,
    columngroups as columngroups,
    io as io,
    plotting as plotting,
)

from captest.io import (
    load_data as load_data,
    load_pvsyst as load_pvsyst,
    DataLoader as DataLoader,
)

try:
    from importlib.metadata import version

    __version__ = version("captest")
except Exception:
    __version__ = "unknown"
