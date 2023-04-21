from captest import (
    capdata,
    util,
    prtest,
    columngroups,
    io,
)

from captest.io import (
    load_data,
    load_pvsyst,
    DataLoader,
)

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from . import _version
__version__ = _version.get_versions()['version']
