from captest import (
    capdata,
    util,
    prtest,
    columngroups,
    io,
    plotting,
)

from captest.io import (
    load_data,
    load_pvsyst,
    DataLoader,
)

try:
    from importlib.metadata import version
    __version__ = version("captest")
except Exception:
    __version__ = "unknown"
