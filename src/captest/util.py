import copy
import difflib
import importlib
import json
import re
import warnings

import numpy as np
import pandas as pd
import yaml
from upath import UPath

from captest.calcparams import CALC_REGISTRY
from captest.setup import (  # noqa: F401
    Calc,
    Column,
    Group,
    canonical_json,
    parse_regression_formula,
)


def read_json(path):
    """Load a JSON file into a python object.

    Works transparently for local filesystem paths and remote URIs
    (e.g. ``s3://bucket/path/file.json``).

    Parameters
    ----------
    path : str, Path, or UPath
        Path to the JSON file.

    Returns
    -------
    dict or list
    """
    return json.loads(UPath(path).read_text())


def read_yaml(path):
    """Load a YAML file into a python object.

    Works transparently for local filesystem paths and remote URIs
    (e.g. ``s3://bucket/path/file.yaml``).

    Parameters
    ----------
    path : str, Path, or UPath
        Path to the YAML file.

    Returns
    -------
    dict or list or None
        The parsed YAML content, or None if the file cannot be parsed.
    """
    data = None
    try:
        data = yaml.safe_load(UPath(path).read_text())
    except yaml.YAMLError as exc:
        print(exc)
    return data


def get_common_timestep(data, units="m", string_output=True):
    """
    Get the most commonly occuring timestep of data as frequency string.

    Parameters
    ----------
    data : Series or DataFrame
        Data with a DateTimeIndex.
    units : str, default 'm'
        String representing date/time unit, such as (D)ay, (M)onth, (Y)ear,
        (h)ours, (m)inutes, or (s)econds.
    string_output : bool, default True
        Set to False to return a numeric value.

    Returns
    -------
    str or numeric
        If the `string_output` is True and the most common timestep is an integer
        in the specified units then a valid pandas frequency or offset alias is
        returned.
        If `string_output` is false, then a numeric value is returned.
    """
    units_abbrev = {"D": "D", "M": "M", "Y": "Y", "h": "H", "m": "min", "s": "S"}
    common_timestep = data.index.to_series().diff().mode().values[0]
    common_timestep_tdelta = common_timestep.astype("timedelta64[m]")
    freq = common_timestep_tdelta / np.timedelta64(1, units)
    if string_output:
        try:
            return str(int(freq)) + units_abbrev[units]
        except Exception:
            return str(freq) + units_abbrev[units]
    else:
        return freq


def reindex_datetime(data, file_name=None, report=False):
    """
    Find dataframe index frequency and reindex to add any missing intervals.

    Sorts index of passed dataframe before reindexing.

    Parameters
    ----------
    data : DataFrame
        DataFrame to be reindexed.
    file_name : str, default None
        Name of file being reindexed. Used for warning message.

    Returns
    -------
    Reindexed DataFrame
    """
    data_index_length = data.shape[0]
    df = data.copy()
    df.sort_index(inplace=True)
    freq_str = get_common_timestep(data, string_output=True)
    full_ix = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq_str)
    try:
        df = df.reindex(index=full_ix)
    except ValueError:
        duplicated = df.index.duplicated()
        dropped_indices = df[duplicated].index
        # warning prints out of order in jupyter lab but not ipython, jupyter lab issue
        warnings.warn(
            f"Dropping duplicate indices from {file_name} before reindexing: {dropped_indices}",
            UserWarning,
        )
        df = df[~duplicated]  # drop rows with duplicate indices before reindexing
        df = df.reindex(index=full_ix)
    df_index_length = df.shape[0]
    missing_intervals = df_index_length - data_index_length

    if report:
        print("Frequency determined to be " + freq_str + " minutes.")
        print(f"{missing_intervals:,} intervals added to index.")
        print("")

    return df, missing_intervals, freq_str


def generate_irr_distribution(lowest_irr, highest_irr, rng=np.random.default_rng(82)):
    """
    Create a list of increasing values similar to POA irradiance data.

    Default parameters result in increasing values where the difference
    between each subsquent value is randomly chosen from the typical range
    of steps for a POA tracker.

    Parameters
    ----------
    lowest_irr : numeric
        Lowest value in the list of values returned.
    highest_irr : numeric
        Highest value in the list of values returned.
    rng : Numpy Random Generator
        Instance of the default Generator.

    Returns
    -------
    irr_values : list
    """
    irr_values = [
        lowest_irr,
    ]
    possible_steps = rng.integers(1, high=8, size=10000) + rng.random(size=10000) - 1
    below_max = True
    while below_max:
        next_val = irr_values[-1] + rng.choice(possible_steps, replace=False)
        if next_val >= highest_irr:
            below_max = False
        else:
            irr_values.append(next_val)
    return irr_values


def tags_by_regex(tag_list, regex_str):
    regex = re.compile(regex_str, re.IGNORECASE)
    return [tag for tag in tag_list if regex.search(tag) is not None]


def detect_solar_noon(data, ghi_col="ghi_mod_csky", default="12:30"):
    """
    Estimate a single representative solar-noon clock time from clear-sky GHI.

    Groups ``data[ghi_col]`` by the clock time of each timestamp (hour and
    minute, ignoring date), takes the mean of each clock-time bucket, and
    returns the bucket with the largest mean formatted as ``"HH:MM"``.

    Used by plotting helpers that split observations into morning and
    afternoon at solar noon.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame with a ``DatetimeIndex``. Must contain ``ghi_col`` for the
        idxmax-based detection to apply.
    ghi_col : str, default ``"ghi_mod_csky"``
        Column to use as the clear-sky GHI signal. ``ghi_mod_csky`` is the
        column added to ``CapData.data`` by ``captest.io.load_data`` when a
        ``site`` dictionary is provided.
    default : str, default ``"12:30"``
        Fallback clock-time string returned when ``ghi_col`` is absent from
        ``data`` or when ``data`` is empty.

    Returns
    -------
    str
        Clock time formatted as ``"HH:MM"``.

    Warns
    -----
    UserWarning
        Emitted when ``ghi_col`` is missing from ``data.columns`` or the
        index is empty; the ``default`` is then returned.
    """
    if ghi_col not in data.columns:
        warnings.warn(
            f"Column {ghi_col!r} not found in data; "
            f"falling back to split_time={default!r}.",
            stacklevel=2,
        )
        return default
    if data.shape[0] == 0:
        warnings.warn(
            f"data has no rows; falling back to split_time={default!r}.",
            stacklevel=2,
        )
        return default
    grouped = data[ghi_col].groupby([data.index.hour, data.index.minute]).mean()
    if grouped.dropna().empty:
        warnings.warn(
            f"All values in {ghi_col!r} are NaN; "
            f"falling back to split_time={default!r}.",
            stacklevel=2,
        )
        return default
    hour, minute = grouped.idxmax()
    return f"{int(hour):02d}:{int(minute):02d}"


def append_tags(sel_tags, tags, regex_str):
    new_list = sel_tags.copy()
    new_list.extend(tags_by_regex(tags, regex_str))
    return new_list


def get_agg_column_name(group_id, agg_func):
    """Generate a column name for an aggregated column.

    Parameters
    ----------
    group_id : str
        Identifier for the group of columns being aggregated.
    agg_func : str or callable
        Aggregation function used.

    Returns
    -------
    str
        Name for the aggregated column.
    """
    if isinstance(agg_func, str):
        col_name = group_id + "_" + agg_func + "_agg"
    else:
        col_name = group_id + "_" + agg_func.__name__ + "_agg"
    return col_name


def reg_col_label(value):
    """Return the column group id or column name a regression column refers to.

    ``CapData.regression_cols`` values are ``Group`` / ``Column`` nodes
    before ``process_regression_columns`` (or ``agg_sensors``) runs and
    plain strings afterwards. Readers that look a value up in
    ``column_groups`` or ``data`` use this to handle both forms.

    Parameters
    ----------
    value : Group, Column, str or other
        A value of ``CapData.regression_cols``.

    Returns
    -------
    object
        ``value.group`` for a ``Group``, ``value.column`` for a ``Column``,
        otherwise ``value`` unchanged.
    """
    if isinstance(value, Group):
        return value.group
    if isinstance(value, Column):
        return value.column
    return value


def _get_or_create_aggregation(node, cd, agg_cache, verbose):
    """Return the column a ``Group`` node resolves to, aggregating if needed.

    An existing ``<group>_<agg>_agg`` column (e.g. measured data loaded from
    an exported test-data file) is reused rather than re-aggregated, with a
    message when ``verbose``.

    Parameters
    ----------
    node : captest.setup.Group
    cd : CapData
    agg_cache : dict
        ``Group`` node -> aggregated column name, shared across one walk.
    verbose : bool

    Returns
    -------
    str
    """
    if node in agg_cache:
        return agg_cache[node]
    expected = get_agg_column_name(node.group, node.agg)
    if expected in cd.data.columns:
        if verbose:
            print(
                f"Reusing existing column '{expected}'; skipping "
                f"aggregation of the {node.group} group.\n"
            )
        agg_name = expected
    else:
        agg_name = cd.agg_group(group_id=node.group, agg_func=node.agg, verbose=verbose)
    agg_cache[node] = agg_name
    return agg_name


def transform_calc_params(node, cd, agg_cache=None, verbose=True):
    """Evaluate a regression-columns tree bottom-up, returning column names.

    Node types (see :mod:`captest.setup`):

    - dict: transform each value (``Side.reg_cols`` or ``Calc.args``);
    - ``Group``: aggregate the column group, return the new column name;
    - ``Column``: return the column name after checking it exists;
    - ``Calc``: transform ``args``, run the registered function through
      ``cd.custom_param`` and return the column it wrote (its registry name);
    - anything else is a literal argument, passed through unchanged.

    Parameters
    ----------
    node : dict, Group, Column, Calc or literal
    cd : CapData
    agg_cache : dict or None
    verbose : bool

    Returns
    -------
    object
        The transformed node.

    Raises
    ------
    KeyError
        If a ``Column`` names a column absent from ``cd.data``.
    """
    if agg_cache is None:
        agg_cache = {}
    if isinstance(node, dict):
        return {
            key: transform_calc_params(value, cd, agg_cache, verbose)
            for key, value in node.items()
        }
    if isinstance(node, Group):
        return _get_or_create_aggregation(node, cd, agg_cache, verbose)
    if isinstance(node, Column):
        if node.column not in cd.data.columns:
            raise KeyError(f"column {node.column!r} not in data")
        return node.column
    if isinstance(node, Calc):
        entry = CALC_REGISTRY[node.calc]
        resolved = transform_calc_params(node.args, cd, agg_cache, verbose)
        cd.custom_param(entry.func, output=node.calc, verbose=verbose, **resolved)
        return node.calc
    return node


def process_reg_cols(
    original_calc_params,
    calc_params=None,
    key_id=None,
    dict_path=None,
    cd=None,
    agg_cache=None,
    verbose=True,
):
    """
    Evaluate a regression-columns tree in place, flattening it to column names.

    ``original_calc_params`` maps each regression variable to a node from
    :mod:`captest.setup`: ``Group`` (aggregate a column group), ``Column``
    (use an existing column) or ``Calc`` (run a registered calculation from
    :data:`captest.calcparams.CALC_REGISTRY` whose ``args`` are themselves
    nodes or literals). In a setup document the tree is written as plain
    mappings, for example::

        meas:
          reg_cols:
            power:
              calc: power_temp_correct
              args:
                power: {group: real_pwr_mtr, agg: sum}
                cell_temp:
                  calc: cell_temp
                  args:
                    poa: {group: irr_poa, agg: mean}
                    bom: {group: temp_bom, agg: mean}
            poa:
              calc: e_total
              args:
                poa:  {group: irr_poa,  agg: mean}
                rpoa: {group: irr_rpoa, agg: mean}

    which ``captest.setup.Side`` validates into nodes. Evaluation starts at the
    leaves: each ``Group`` is aggregated once per call (an existing
    ``<group>_<agg>_agg`` column is reused), each ``Calc`` writes a column
    named after its registry name, and every variable is replaced by the name
    of the column that holds it.

    Parameters
    ----------
    original_calc_params : dict
        The original dictionary to be modified
    calc_params : dict or tuple
        Deprecated. Ignored if provided.
    key_id : str
        Deprecated. Ignored if provided.
    dict_path : list
        Deprecated. Ignored if provided.
    cd : CapData
        CapData instance the aggregations and calculations act on.
    agg_cache : dict, optional
        Cache of already aggregated column groups to avoid redundant calls to agg_group.
        Keys are ``Group`` nodes and values are the aggregated column names.
    verbose : bool, default True
        Passed to the group aggregations and the parameter calculations. Set to False
        to prevent all summary output.

    Returns
    -------
    None
        Modifies the original_calc_params and the data attribute of the CapData object
        passed to the `cd` argument.
    """
    if agg_cache is None:
        agg_cache = {}

    result = transform_calc_params(original_calc_params, cd, agg_cache, verbose)

    original_calc_params.clear()
    original_calc_params.update(result)


_PERC_N_PREFIX = "perc_"


def perc_wrap(p):
    """Return a callable that computes the ``p``-th percentile of a Series.

    Pass the result directly as a ``func`` value to ``CapData.rep_cond`` (or
    ``CapTest.rep_cond``) for a percentile-based reporting condition (e.g.
    ``perc_wrap(60)`` for the 60th percentile POA). Test setup documents and
    ``CapTest`` overrides do not accept callables; they spell the same thing as
    the string ``"perc_N"`` (e.g. ``"perc_60"``), which ``CapTest.rep_cond``
    resolves to ``perc_wrap(N)``.

    Parameters
    ----------
    p : numeric
        Percentile in [0, 100].

    Returns
    -------
    callable
        Function that takes a pandas Series or 1-d array-like and returns the
        p-th percentile via ``Series.quantile(p / 100, interpolation='nearest')``.
        Missing values are skipped, so a NaN in the data does not make the
        result NaN. Called directly on a DataFrame it returns a Series of
        per-column percentiles. Its ``__name__`` is ``"perc_wrap(p)"``, which
        is how it is displayed in filter summaries and serialized as
        ``"perc_N"``.
    """

    def percentile(x):
        if not isinstance(x, (pd.Series, pd.DataFrame)):
            x = pd.Series(x)
        return x.quantile(p / 100, interpolation="nearest")

    percentile.__name__ = f"perc_wrap({p})"
    return percentile


def _resolve_perc_string(val):
    """Resolve a "perc_N" string to ``perc_wrap(N)``.

    Non-matching strings pass through unchanged. Malformed ``perc_*`` strings
    raise ``ValueError``.
    """
    if not isinstance(val, str) or not val.startswith(_PERC_N_PREFIX):
        return val
    suffix = val[len(_PERC_N_PREFIX) :]
    if not suffix:
        raise ValueError(f"Malformed percentile string {val!r}: expected 'perc_<int>'.")
    try:
        n = int(suffix)
    except ValueError as exc:
        raise ValueError(
            f"Malformed percentile string {val!r}: expected 'perc_<int>', "
            f"got suffix {suffix!r}."
        ) from exc
    return perc_wrap(n)


def _resolve_func_strings(func_dict):
    """Resolve ``perc_N`` strings inside a rep_conditions.func dict."""
    if not isinstance(func_dict, dict):
        return func_dict
    return {key: _resolve_perc_string(val) for key, val in func_dict.items()}


def to_native(value):
    """Coerce a numpy scalar to its native Python equivalent for YAML output.

    ``yaml.safe_dump`` cannot represent numpy scalar types (e.g.
    ``np.float64``), so values that originate from pandas/numpy (such as a
    reporting-irradiance ``ref_val`` pulled from ``cd.rc['poa'].iloc[0]``)
    raise ``yaml.representer.RepresenterError`` when serialized. This converts
    any ``np.generic`` scalar to a plain ``int``/``float``/etc.; all other
    values pass through unchanged.

    Parameters
    ----------
    value : object
        Any value bound for serialization.

    Returns
    -------
    object
        ``value.item()`` when ``value`` is a numpy scalar, otherwise ``value``.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def _perc_wrap_to_string(val):
    """Inverse of :func:`_resolve_perc_string`.

    Converts a callable produced by :func:`perc_wrap` back into its
    round-trippable ``"perc_N"`` string form. Non-perc_wrap values pass
    through unchanged.
    """
    if not callable(val):
        return val
    name = getattr(val, "__name__", "")
    prefix = "perc_wrap("
    if name.startswith(prefix) and name.endswith(")"):
        inner = name[len(prefix) : -1]
        try:
            int(inner)
        except ValueError:
            return val
        return f"perc_{inner}"
    return val


def callable_to_qualname(func):
    """Return a ``'module:qualname'`` import string for a named callable.

    Raises ``ValueError`` for lambdas and closures (``<lambda>`` / ``<locals>``
    in the qualname) — they are not importable and cannot round-trip.
    """
    module = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if not module or not qualname:
        raise ValueError(
            f"Cannot serialize callable {func!r}: missing __module__/__qualname__."
        )
    if "<lambda>" in qualname or "<locals>" in qualname:
        raise ValueError(
            f"Cannot serialize callable {func!r}: lambdas and closures are not "
            f"importable. Use a module-level named function."
        )
    return f"{module}:{qualname}"


def callable_from_qualname(ref):
    """Import a callable from a ``'module:qualname'`` string (inverse of above)."""
    if not isinstance(ref, str) or ":" not in ref:
        raise ValueError(
            f"Malformed callable reference {ref!r}: expected 'module:qualname'."
        )
    module_name, _, qualname = ref.partition(":")
    try:
        obj = importlib.import_module(module_name)
        for part in qualname.split("."):
            obj = getattr(obj, part)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import callable {ref!r}: {exc}") from exc
    return obj


class StrictAttrs:
    """Reject assignment to names that are neither params nor runtime state.

    Mixin for ``param.Parameterized`` subclasses. ``param`` accepts arbitrary
    attribute assignment, which makes an edit-then-replay workflow silently do
    nothing when a param name is mistyped. Raising at the assignment surfaces
    the typo instead of producing an unchanged re-run.

    A subclass that writes a plain (non-param) attribute must name it in its
    own ``_runtime_attrs``; ``__init_subclass__`` unions those declarations
    down the hierarchy, so each subclass declares only what it adds.
    Leading-underscore names pass through unchecked; they cover ``param``'s
    own instance internals as well as subclasses' private state.

    Notes
    -----
    Must be combined with ``param.Parameterized`` (``self.param`` is
    required)::

        class Step(StrictAttrs, param.Parameterized):
            ...
    """

    _runtime_attrs = frozenset()

    def __init_subclass__(cls, **kwargs):
        """Union each subclass's ``_runtime_attrs`` with those of its bases."""
        super().__init_subclass__(**kwargs)
        cls._runtime_attrs = frozenset().union(
            *(base.__dict__.get("_runtime_attrs", frozenset()) for base in cls.__mro__)
        )

    def __setattr__(self, name, value):
        """Allow params, runtime attrs, and private names; reject the rest.

        Raises
        ------
        AttributeError
            If ``name`` is neither a declared param, a name in
            ``_runtime_attrs``, nor leading-underscore. The message lists the
            settable names and suggests the closest match.
        """
        if (
            name.startswith("_")
            or name in self.param
            or name in type(self)._runtime_attrs
        ):
            super().__setattr__(name, value)
            return
        settable = sorted(set(self.param) | type(self)._runtime_attrs)
        suggestion = difflib.get_close_matches(name, settable, n=1)
        hint = f" Did you mean {suggestion[0]!r}?" if suggestion else ""
        raise AttributeError(
            f"{type(self).__name__} has no parameter or attribute {name!r}. "
            f"Settable names: {settable}.{hint}"
        )


def params_to_config(obj):
    """Serialize a ``param.Parameterized``'s values to a yaml-safe dict.

    Every param value except the param-system ``name`` is deep-copied and
    passed through :func:`to_native`, so the result is an independent snapshot
    that survives ``yaml.safe_dump``.

    Parameters
    ----------
    obj : param.Parameterized
        Object whose param values are serialized.

    Returns
    -------
    dict
        Mapping of param name to native-python value.
    """
    return {
        k: to_native(copy.deepcopy(v))
        for k, v in obj.param.values().items()
        if k != "name"
    }


def format_name_list(names, cutoff=10, edge=3):
    """Render a sequence of names as a string, truncating long sequences.

    Mirrors the truncation ``CapData.agg_group`` applies to its verbose
    output: sequences longer than ``cutoff`` are shown as the first and last
    ``edge`` names with an ellipsis between them, followed by the full count.
    A site with dozens of thermocouples otherwise turns one prep explanation
    or error message into an unreadable wall of column names.

    Parameters
    ----------
    names : sequence of str
        Column names or group ids to render.
    cutoff : int, default 10
        Maximum number of names to list in full. A sequence at or below this
        length is joined unchanged.
    edge : int, default 3
        Number of names shown at each end when the sequence is truncated.

    Returns
    -------
    str
        Comma-separated names, e.g. ``"a, b, c"`` or
        ``"a, b, c, ..., x, y, z (26 total)"``.
    """
    names = [str(name) for name in names]
    if len(names) <= cutoff:
        return ", ".join(names)
    head = ", ".join(names[:edge])
    tail = ", ".join(names[-edge:])
    return f"{head}, ..., {tail} ({len(names)} total)"
