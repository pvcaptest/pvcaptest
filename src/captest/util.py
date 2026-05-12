import json
import re
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml
from patsy import ModelDesc


def read_json(path):
    with open(path) as f:
        json_data = json.load(f)
    return json_data


def read_yaml(path):
    with open(path, "r") as stream:
        try:
            data = yaml.safe_load(stream)
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
    print("before calling get common timestep")
    freq_str = get_common_timestep(data, string_output=True)
    print(freq_str)
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
        print("{:,} intervals added to index.".format(missing_intervals))
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


def update_by_path(dictionary, path, new_value=None, convert_callable=False):
    """
    Update a nested dictionary value by following a path list.

    Parameters
    ----------
    dictionary : dict
        The dictionary to update
    path : list
        A list representing the path to the target key
    new_value : optional
        The new value to set (if None and convert_callable=True,
        will convert existing tuple to function name)
    convert_callable : bool, optional
        If True and new_value is None, converts tuple to function name

    Returns
    -------
    updated_dictionary : dict
        The updated dictionary
    """
    # Get a reference to the current level in the dictionary
    current = dictionary

    # Navigate to the parent of the target key
    for key in path[:-1]:
        current = current[key]

    # If convert_callable is True and no new value provided, convert existing tuple
    if convert_callable and new_value is None:
        target_value = current[path[-1]]
        if isinstance(target_value, tuple) and callable(target_value[0]):
            current[path[-1]] = target_value[0].__name__
    else:
        # Update the target key with the new value
        current[path[-1]] = new_value

    return dictionary


def _is_aggregation_tuple(node):
    """Check if node is an aggregation tuple: (group_id: str, agg_func: str)."""
    return (
        isinstance(node, tuple)
        and len(node) == 2
        and isinstance(node[0], str)
        and isinstance(node[1], str)
    )


def _is_calculation_tuple(node):
    """Check if node is a calculation tuple: (callable, dict)."""
    return (
        isinstance(node, tuple)
        and len(node) == 2
        and callable(node[0])
        and isinstance(node[1], dict)
    )


def _resolve_column_group(value, cd):
    """
    Resolve a column group ID to an actual column name.

    Parameters
    ----------
    value : str
        The column group ID or column name.
    cd : CapData
        CapData instance with column_groups attribute.

    Returns
    -------
    str
        The resolved column name.

    Raises
    ------
    ValueError
        If the column group has more than one column.
    """
    if value in cd.column_groups:
        if len(cd.column_groups[value]) == 1:
            return cd.column_groups[value][0]
        else:
            raise ValueError(
                f'Looks like you specified a column group ID "{value}" that '
                f"points to a group with more than one column. "
                f'Try replacing it with ("{value}", "mean") or a different '
                f"aggregation method."
            )
    return value


def _get_or_create_aggregation(group_id, agg_func, cd, agg_cache, verbose):
    """
    Get an aggregated column name, creating it if necessary.

    Parameters
    ----------
    group_id : str
        The column group ID to aggregate.
    agg_func : str
        The aggregation function name.
    cd : CapData
        CapData instance.
    agg_cache : dict
        Cache of already aggregated columns.
    verbose : bool
        Whether to print verbose output.

    Returns
    -------
    str
        The aggregated column name.
    """
    cache_key = (group_id, agg_func)
    if cache_key in agg_cache:
        return agg_cache[cache_key]

    expected_agg_name = get_agg_column_name(group_id, agg_func)
    if expected_agg_name in cd.data.columns:
        agg_name = expected_agg_name
    else:
        agg_name = cd.agg_group(group_id=group_id, agg_func=agg_func, verbose=verbose)

    agg_cache[cache_key] = agg_name
    return agg_name


def transform_calc_params(node, cd, agg_cache=None, verbose=True):
    """
    Recursively transform a calc_params node, returning resolved values.

    This function processes a nested dictionary structure that defines regression
    parameters, executing aggregations and calculations as needed, and returns
    a flattened structure with resolved column names.

    Node types handled:
    - dict: Transform each value recursively
    - tuple (str, str): Aggregation - returns aggregated column name
    - tuple (callable, dict): Calculation - executes function, returns function name
    - str: Column group ID - resolved to column name if single column
    - other: Passed through unchanged (e.g., numeric values)

    Parameters
    ----------
    node : dict, tuple, str, or other
        The current node in the calc_params structure.
    cd : CapData
        CapData instance that functions will act on.
    agg_cache : dict, optional
        Cache of already aggregated column groups to avoid redundant calls.
        Keys are tuples of (group_id, agg_func), values are aggregated column names.
    verbose : bool, default True
        Passed to aggregations and calculations. Set to False to suppress output.

    Returns
    -------
    transformed
        The transformed node with all aggregations executed and calculations
        replaced by their function names.
    """
    if agg_cache is None:
        agg_cache = {}

    if isinstance(node, dict):
        return {
            key: transform_calc_params(value, cd, agg_cache, verbose)
            for key, value in node.items()
        }

    if _is_aggregation_tuple(node):
        group_id, agg_func = node
        return _get_or_create_aggregation(group_id, agg_func, cd, agg_cache, verbose)

    if _is_calculation_tuple(node):
        func, kwargs = node
        resolved_kwargs = transform_calc_params(kwargs, cd, agg_cache, verbose)
        cd.custom_param(func, **resolved_kwargs, verbose=verbose)
        return func.__name__

    if isinstance(node, str):
        return _resolve_column_group(node, cd)

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
    Recursively process a regression columns dictionary that includes calculated parameters.

    The regression parameters dictionary attribute of CapData can be defined with a
    nested structure which includes tuples with two values where the first is a
    CapData method to calculate a new value (column of Data attribute) and the second
    is a dictionary of the kwargs to be passed to the function.

    An example tuple:
    (bom_temp, {'poa': 'irr_poa', 'temp_amb':'temp_amb', 'wind_speed':'wind_speed'})

    Where bom_temp is a CapData method that accepts the kwargs poa, temp_amb,
    and wind_speed, which have the values (column group ids) irr_poa, temp_amb, wind_speed,
    respectively.

    Additionally, column groups can be aggregated by specifying a tuple which contains
    two strings - the column group id (e.g., 'irr_poa') and the aggregation method
    (e.g. 'mean'). This will result in the CapData.agg_group method being called and
    the first value in the tuple passed to the group_id kwarg and the second passed
    to the agg_func kwarg.

    If a regression parameter key is paired with a column groups id for a column
    group with only a single column, then that column name will replace the column group
    id.

    The dictionary passed to `original_calc_params` may be nested like this example:

    calc_params_map = {
        'power_tc': (CapData.power_tc, {
            'power': 'real_pwr_mtr',
            'cell_temp': (CapData.cell_temp, {
                'poa': ('irr_poa', 'mean'),
                'bom': (CapData.bom_temp, {
                    'poa': ('irr_poa', 'mean'),
                    'temp_amb': ('temp_amb', 'mean'),
                    'wind_speed': ('wind_speed', 'mean')
                })
            })
        }),
    }

    This function will start at the bottom of nested dictionaries and progressively
    call the functions with the kwargs replacing the function tuples with the function
    names or the aggregated column names.

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
        CapData instance that functions in original_calc_params will act on.
    agg_cache : dict, optional
        Cache of already aggregated column groups to avoid redundant calls to agg_group.
        Keys are tuples of (group_id, agg_func) and values are the aggregated column names.
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


def parse_regression_formula(formula: str) -> Tuple[List[str], List[str]]:
    """
    Return (lhs_list, rhs_list) for `formula`.

    Rules
    -----
    • Each list contains the **unique raw variable names** appearing on
      that side, sorted.
    • `- 1` (intercept-removal) is ignored.
    • `I(...)` blocks are unwrapped; products like `I(poa * t_amb)` are
      split into their component symbols (`poa`, `t_amb`).

    Parameters
    ----------
    formula : str
        Regression formula to parse.

    Returns
    -------
    Tuple[List[str], List[str]]
        Tuple of (lhs_list, rhs_list).
    """
    # --- helpers ------------------------------------------------------
    _sym_re = re.compile(r"[A-Za-z_]\w*")

    def _extract_raw_names(factor_str: str) -> List[str]:
        """
        Turn 'I(poa * t_amb)'  ->  ['poa', 't_amb']
             'poa'             ->  ['poa']
        """
        # strip outer I(…)
        if factor_str.startswith("I(") and factor_str.endswith(")"):
            factor_str = factor_str[2:-1]
        # split by * or :  (products/interactions)
        parts = re.split(r"[\*\:]", factor_str)
        names = []
        for part in parts:
            # pull out identifier tokens
            names.extend(_sym_re.findall(part))
        return names

    # --- main logic ---------------------------------------------------
    md = ModelDesc.from_formula(formula)

    lhs_list: List[str] = []
    rhs_list: List[str] = []

    # left
    for term in md.lhs_termlist:
        for f in term.factors:
            for name in _extract_raw_names(f.name()):
                if name not in lhs_list:
                    lhs_list.append(name)

    # right
    for term in md.rhs_termlist:
        for f in term.factors:
            for name in _extract_raw_names(f.name()):
                if name not in rhs_list:
                    rhs_list.append(name)

    # discard the Patsy-built-in intercept symbol if present
    rhs_list = [n for n in rhs_list if n != "Intercept"]

    return lhs_list, rhs_list
