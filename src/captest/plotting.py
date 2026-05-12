from pathlib import Path
from typing import TYPE_CHECKING
import copy
import json
import re
import warnings
import itertools
from functools import partial
import importlib

import numpy as np
import pandas as pd
import param
from bokeh.models import NumeralTickFormatter

from captest import util
from captest.calcparams import cell_temp, power_temp_correct
from .util import tags_by_regex, read_json

if TYPE_CHECKING:
    # Imported only for static type hints / IDE support. Doing the import at
    # runtime would create a cycle: capdata.py imports plotting at module
    # top, so plotting cannot import CapData from a partially-initialized
    # capdata module. See docs/specs/2026-04-26-scatter-plot-am-pm-tc-power-design.md.
    from captest.capdata import CapData  # noqa: F401

pn_spec = importlib.util.find_spec("panel")
if pn_spec is not None:
    import panel as pn

    pn.extension()
    # disable error messages for panel dashboard
    pn.config.console_output = "disable"
else:
    warnings.warn(
        "The ReportingIrradiance.dashboard method will not work without "
        "the panel package."
    )

hv_spec = importlib.util.find_spec("holoviews")
if hv_spec is not None:
    import holoviews as hv
    from holoviews import opts
    from holoviews.plotting.links import DataLink
else:
    warnings.warn("The plotting methods will not work without the holoviews package.")

# Stable column name written by ``calc_tc_power_column`` for plot-only
# temperature-corrected power. Distinct from ``power_temp_correct`` (the
# function-name column produced by ``CapData.process_regression_columns``
# under the bifi_power_tc preset) to avoid colliding with that workflow.
TC_POWER_PLOT_COL = "power_tc_plot"

# Default calc-params expression used by ``ScatterPlot`` when ``tc_power``
# is enabled and ``tc_power_calc`` is not provided. Tuned for measured DAS
# data following the standard column-group inference
# (``captest.columngroups.group_columns``); assumes a measured
# back-of-module temperature group ``temp_bom`` is available. Sim users
# must pass an explicit ``tc_power_calc``.
DEFAULT_TC_POWER_CALC = {
    "power": (
        power_temp_correct,
        {
            "power": ("real_pwr_mtr", "sum"),
            "cell_temp": (
                cell_temp,
                {
                    "poa": ("irr_poa", "mean"),
                    "bom": ("temp_bom", "mean"),
                },
            ),
        },
    ),
}

COMBINE = {
    "poa_ghi": "irr.*(poa|ghi)$",
    "poa_csky": "(?=.*poa)(?=.*irr)",
    "ghi_csky": "(?=.*ghi)(?=.*irr)",
    "temp_amb_bom": "(?=.*temp)((?=.*amb)|(?=.*bom))",
    "inv_sum_mtr_pwr": ["(?=.*real)(?=.*pwr)(?=.*mtr)", "(?=.*pwr)(?=.*agg)"],
}

DEFAULT_GROUPS = [
    "inv_sum_mtr_pwr",
    "(?=.*real)(?=.*pwr)(?=.*inv)",
    "(?=.*real)(?=.*pwr)(?=.*mtr)",
    "poa_ghi",
    "poa_csky",
    "ghi_csky",
    "temp_amb_bom",
]


def find_default_groups(groups, default_groups):
    """
    Find the default groups in the list of groups.

    Parameters
    ----------
    groups : list of str
        The list of groups to search for the default groups.
    default_groups : list of str
        List of regex strings to use to identify default groups.

    Returns
    -------
    list of str
        The default groups found in the list of groups.
    """
    found_groups = []
    for re_str in default_groups:
        found_grp = tags_by_regex(groups, re_str)
        if len(found_grp) == 1:
            found_groups.append(found_grp[0])
        elif len(found_grp) > 1:
            warnings.warn(
                f"More than one group found for regex string {re_str}. "
                "Refine regex string to find only one group. "
                f"Groups found: {found_grp}"
            )
    return found_groups


def parse_combine(combine, column_groups=None, data=None, cd=None):
    """
    Parse regex strings for identifying groups of columns or tags to combine.

    Parameters
    ----------
    combine : dict
        Dictionary of group names and regex strings to use to identify groups from
        column groups and individual tags (columns) to combine into new groups.
        Keys should be strings for names of new groups. Values should be either a string
        or a list of two strings. If a string, the string is used as a regex to identify
        groups to combine. If a list, the first string is used to identify groups to
        combine and the second is used to identify individual tags (columns) to combine.
    column_groups : ColumnGroups, optional
        The column groups object to add new groups to. Required if `cd` is not provided.
    data : pd.DataFrame, optional
        The data to use to identify groups and columns to combine. Required if `cd` is
        not provided.
    cd : captest.CapData, optional
        The captest.CapData object with the `data` and `column_groups` attributes set.
        Required if `columng_groups` and `data` are not provided.

    Returns
    -------
        ColumnGroups
            New column groups object with new groups added.
    """
    if cd is not None:
        data = cd.data
        column_groups = cd.column_groups
    cg_out = copy.deepcopy(column_groups)
    orig_groups = list(cg_out.keys())

    tags = list(data.columns)

    for grp_name, re_str in combine.items():
        group_re = None
        tag_re = None
        tags_in_matched_groups = []
        matched_tags = []
        if isinstance(re_str, str):
            group_re = re_str
        elif isinstance(re_str, list):
            if len(re_str) != 2:
                warnings.warn(
                    "When passing a list of regex. There should be two strings. One for "
                    "identifying groups and one for identifying individual tags (columns)."
                )
                return None
            else:
                group_re = re_str[0]
                tag_re = re_str[1]
        if group_re is not None:
            matched_groups = tags_by_regex(orig_groups, group_re)
            if len(matched_groups) >= 1:
                tags_in_matched_groups = list(
                    itertools.chain(*[cg_out[grp] for grp in matched_groups])
                )
        if tag_re is not None:
            matched_tags = tags_by_regex(tags, tag_re)
        cg_out[grp_name] = tags_in_matched_groups + matched_tags
    return cg_out


def msel_from_column_groups(column_groups, groups=True):
    """
    Create a multi-select widget from a column groups object.

    Parameters
    ----------
    column_groups : ColumnGroups
        The column groups object.
    groups : bool, default True
        By default creates list of groups i.e. the keys of `column_groups`,
        otherwise creates list of individual columns i.e. the values of `column_groups`
        concatenated together.
    """
    if groups:
        keys = list(column_groups.data.keys())
        keys.sort()
        options = {k: column_groups.data[k] for k in keys}
        name = "Groups"
        value = column_groups.data[list(column_groups.keys())[0]]
    else:
        options = []
        for columns in column_groups.values():
            options += columns
        options.sort()
        name = "Columns"
        value = [options[0]]
    return pn.widgets.MultiSelect(
        name=name, value=value, options=options, size=8, height=400, width=400
    )


def plot_tag(data, tag, width=1500, height=250):
    if len(tag) == 1:
        plot = hv.Curve(data[tag])
    elif len(tag) > 1:
        curves = {}
        for column in tag:
            try:
                curves[column] = hv.Curve(data[column])
            except KeyError:
                continue
        plot = hv.NdOverlay(curves)
    elif len(tag) == 0:
        plot = hv.Curve(
            pd.DataFrame({"no_data": [np.nan] * data.shape[0]}, index=data.index)
        )
    plot.opts(
        opts.Curve(
            line_width=1,
            width=width,
            height=height,
            muted_alpha=0,
            tools=["hover"],
            yformatter=NumeralTickFormatter(format="0,0"),
        ),
        opts.NdOverlay(
            width=width, height=height, legend_position="right", batched=False
        ),
    )
    return plot


def group_tag_overlay(group_tags, column_tags):
    """
    Overlay curves of groups and individually selected columns.

    Parameters
    ----------
    group_tags : list of str
        The tags to plot from the groups selected.
    column_tags : list of str
        The tags to plot from the individually selected columns.
    """
    joined_tags = [t for tag_list in group_tags for t in tag_list] + column_tags
    return joined_tags


def plot_group_tag_overlay(data, group_tags, column_tags, width=1500, height=400):
    """
    Overlay curves of groups and individually selected columns.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot.
    group_tags : list of str
        The tags to plot from the groups selected.
    column_tags : list of str
        The tags to plot from the individually selected columns.
    """
    joined_tags = group_tag_overlay(group_tags, column_tags)
    return plot_tag(data, joined_tags, width=width, height=height)


def plot_tag_groups(data, tags_to_plot, width=1500, height=250):
    """
    Plot groups of tags, one of overlayed curves per group.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot.
    tags_to_plot : list
        List of lists of strings. One plot for each inner list.
    """
    group_plots = []
    if len(tags_to_plot) == 0:
        tags_to_plot = [[]]
    for group in tags_to_plot:
        plot = plot_tag(data, group, width=width, height=height)
        group_plots.append(plot)
    return hv.Layout(group_plots).cols(1)


def filter_list(text_input, ms_to_filter, names, event=None):
    """
    Filter a multi-select widget by a regex string.

    Parameters
    ----------
    text_input : pn.widgets.TextInput
        The text input widget to get the regex string from.
    ms_to_filter : pn.widgets.MultiSelect
        The multi-select widget to update.
    names : list of str
        The list of names to filter.
    event : pn.widgets.event, optional
        Passed by the `param.watch` method. Not used.

    Returns
    -------
    None
    """
    if text_input.value == "":
        re_value = ".*"
    else:
        re_value = text_input.value
    names_ = copy.deepcopy(names)
    if isinstance(names_, dict):
        selected_groups = tags_by_regex(list(names_.keys()), re_value)
        selected_groups.sort()
        options = {k: names_[k] for k in selected_groups}
    else:
        options = tags_by_regex(names_, re_value)
        options.sort()
    ms_to_filter.param.update(options=options)


def scatter_dboard(data, **kwargs):
    """
    Create a dashboard to plot any two columns of data against each other.

    Parameters
    ----------
    data : pd.DataFrame
        The data to plot.
    **kwargs : optional
        Pass additional keyword arguments to the holoviews options of the scatter plot.

    Returns
    -------
    pn.Column
        The dashboard with a scatter plot of the data.
    """
    cols = list(data.columns)
    cols.sort()
    x = pn.widgets.Select(name="x", value=cols[0], options=cols)
    y = pn.widgets.Select(name="y", value=cols[1], options=cols)
    # slope = pn.widgets.Checkbox(name='Slope', value=False)

    defaults = {
        "width": 500,
        "height": 500,
        "fill_alpha": 0.4,
        "line_alpha": 0,
        "size": 4,
        "yformatter": NumeralTickFormatter(format="0,0"),
        "xformatter": NumeralTickFormatter(format="0,0"),
    }
    for opt, value in defaults.items():
        kwargs.setdefault(opt, value)

    def scatter(data, x, y, slope=True, **kwargs):
        scatter_plot = hv.Scatter(data, x, y).opts(**kwargs)
        # if slope:
        #     slope_line = hv.Slope.from_scatter(scatter_plot).opts(
        #         line_color='red',
        #         line_width=1,
        #         line_alpha=0.4,
        #         line_dash=(5,3)
        #     )
        # if slope:
        #     return scatter_plot * slope_line
        # else:
        return scatter_plot

    # dboard = pn.Column(
    #     pn.Row(x, y, slope),
    #     pn.bind(scatter, data, x, y, slope=slope, **kwargs)
    # )
    dboard = pn.Column(pn.Row(x, y), pn.bind(scatter, data, x, y, **kwargs))
    return dboard


def plot(
    cd=None,
    cg=None,
    data=None,
    combine=COMBINE,
    default_groups=DEFAULT_GROUPS,
    group_width=1500,
    group_height=250,
    plot_defaults_path=None,
    **kwargs,
):
    """
    Create plotting dashboard.

    NOTE: If a plot defaults JSON file exists in the current working directory, the
    default groups will be read from that file instead of using the `default_groups`
    argument. When a `cd` (CapData) object is provided, the file is named
    ``plot_defaults_{cd.name}.json`` to avoid conflicts between multiple CapData objects
    in the same session. Otherwise the file is named ``plot_defaults.json``. Use the
    `plot_defaults_path` argument to override the path. Delete or manually edit the
    file to change the default groups. Columns in the file that are no longer present
    in the data are ignored with a warning.

    Parameters
    ----------
    cd : captest.CapData, optional
        The captest.CapData object.
    cg : captest.ColumnGroups, optional
        The captest.ColumnGroups object. `data` must also be provided.
    data : pd.DataFrame, optional
        The data to plot. `cg` must also be provided.
    combine : dict, optional
        Dictionary of group names and regex strings to use to identify groups from
        column groups and individual tags (columns) to combine into new groups. See the
        `parse_combine` function for more details.
    default_groups : list of str, optional
        List of regex strings to use to identify default groups to plot. See the
        `find_default_groups` function for more details.
    group_width : int, optional
        The width of the plots on the Groups tab.
    group_height : int, optional
        The height of the plots on the Groups tab.
    plot_defaults_path : str or Path, optional
        Path to the plot defaults JSON file. Overrides the default naming scheme.
        When None and `cd` is provided, defaults to
        ``./plot_defaults_{cd.name}.json``. When None and `cd` is not provided,
        defaults to ``./plot_defaults.json``.
    **kwargs : optional
        Pass additional keyword arguments to the holoviews options of the scatter plot
        on the 'Scatter' tab.
    """
    if cd is not None:
        data = cd.data
        cg = cd.column_groups
    # determine path for plot defaults file
    if plot_defaults_path is not None:
        defaults_path = Path(plot_defaults_path)
    elif cd is not None:
        safe_name = re.sub(r"[^\w\-]", "_", cd.name)
        defaults_path = Path(f"./plot_defaults_{safe_name}.json")
    else:
        defaults_path = Path("./plot_defaults.json")
    # make sure data is numeric
    data = data.apply(pd.to_numeric, errors="coerce")
    bool_columns = data.select_dtypes(include="bool").columns
    data.loc[:, bool_columns] = data.loc[:, bool_columns].astype(int)
    # setup custom plot for 'Custom' tab
    groups = msel_from_column_groups(cg)
    tags = msel_from_column_groups({"all_tags": list(data.columns)}, groups=False)
    columns_re_input = pn.widgets.TextInput(name="Input regex to filter columns list")
    groups_re_input = pn.widgets.TextInput(name="Input regex to filter groups list")

    columns_re_input.param.watch(
        partial(filter_list, columns_re_input, tags, tags.options), "value"
    )
    groups_re_input.param.watch(
        partial(filter_list, groups_re_input, groups, groups.options), "value"
    )

    custom_plot_name = pn.widgets.TextInput()
    update = pn.widgets.Button(name="Update")
    width_custom = pn.widgets.IntInput(
        name="Plot Width", value=1500, start=200, end=2800, step=100, width=200
    )
    height_custom = pn.widgets.IntInput(
        name="Plot height", value=400, start=150, end=800, step=50, width=200
    )
    custom_plot = pn.Column(
        pn.Row(custom_plot_name, update, width_custom, height_custom),
        pn.Row(
            pn.WidgetBox(groups_re_input, groups),
            pn.WidgetBox(columns_re_input, tags),
        ),
        pn.Row(
            pn.bind(
                plot_group_tag_overlay,
                data,
                groups,
                tags,
                width=width_custom,
                height=height_custom,
            )
        ),
    )

    # setup group plotter for 'Main' tab
    cg_layout = parse_combine(combine, column_groups=cg, data=data)
    main_ms = msel_from_column_groups(cg_layout)

    def add_custom_plot_group(event):
        column_groups_ = copy.deepcopy(main_ms.options)
        column_groups_ = add_custom_plot(
            custom_plot_name.value,
            column_groups_,
            groups.value,
            tags.value,
        )
        main_ms.options = column_groups_

    update.on_click(add_custom_plot_group)
    plots_to_layout = pn.widgets.Button(name="Set plots to current layout")
    width_main = pn.widgets.IntInput(
        name="Plot Width", value=1500, start=200, end=2800, step=100, width=200
    )
    height_main = pn.widgets.IntInput(
        name="Plot height", value=250, start=150, end=800, step=50, width=200
    )
    main_plot = pn.Column(
        pn.Row(pn.WidgetBox(plots_to_layout, main_ms, pn.Row(width_main, height_main))),
        pn.Row(
            pn.bind(
                plot_tag_groups, data, main_ms, width=width_main, height=height_main
            )
        ),
    )

    def set_defaults(event):
        with open(defaults_path, "w") as file:
            json.dump(main_ms.value, file)

    plots_to_layout.on_click(set_defaults)

    # setup default groups
    if defaults_path.exists():
        default_tags = read_json(str(defaults_path))
        valid_columns = set(data.columns)
        filtered_tags = []
        missing_columns = []
        for tag_group in default_tags:
            valid_group_tags = [t for t in tag_group if t in valid_columns]
            missing_columns.extend(t for t in tag_group if t not in valid_columns)
            if valid_group_tags:
                filtered_tags.append(valid_group_tags)
        if missing_columns:
            warnings.warn(
                f"The following columns from {defaults_path.name} were not found "
                f"in the data and will be ignored: {missing_columns}"
            )
        if filtered_tags:
            default_tags = filtered_tags
        else:
            warnings.warn(
                f"No valid columns found in {defaults_path.name}. "
                "Falling back to default groups."
            )
            default_groups = find_default_groups(list(cg_layout.keys()), default_groups)
            default_tags = [cg_layout.get(grp, []) for grp in default_groups]
    else:
        default_groups = find_default_groups(list(cg_layout.keys()), default_groups)
        default_tags = [cg_layout.get(grp, []) for grp in default_groups]

    # layout dashboard
    plotter = pn.Tabs(
        (
            "Groups",
            plot_tag_groups(data, default_tags, width=group_width, height=group_height),
        ),
        ("Layout", main_plot),
        ("Overlay", custom_plot),
        ("Scatter", scatter_dboard(data, **kwargs)),
    )
    return plotter


def add_custom_plot(name, column_groups, group_tags, column_tags):
    """
    Append a new custom group to column groups for plotting.

    Parameters
    ----------
    """
    column_groups[name] = group_tag_overlay(group_tags, column_tags)
    return column_groups


def get_resid_exog_frame(cd):
    """
    Get a DataFrame of residuals and exogenous variables from a CapData object.

    Parameters
    ----------
    cd : captest.CapData
        The CapData object.

    Returns
    -------
    pd.DataFrame
        DataFrame with residuals and exogenous variables.
    """
    exog_names = cd.regression_results.model.exog_names
    meas_resid_exog = (
        pd.concat(
            [
                cd.regression_results.resid.rename("resid"),
                pd.DataFrame(
                    cd.regression_results.model.exog,
                    columns=exog_names,
                    index=cd.data_filtered.index,
                ),
            ],
            axis=1,
        )
        .rename_axis(index="Timestamp")
        .reset_index()
    )
    meas_resid_exog["source"] = cd.name
    return exog_names, meas_resid_exog


_SPLIT_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def _missing_column_groups(node, available_groups):
    """Return the set of column-group ids referenced by ``node`` but absent
    from ``available_groups``.

    Walks the same calc-params nested-dict / aggregation-tuple /
    calculation-tuple grammar as ``util.transform_calc_params`` and
    collects only the leaves that look like column-group references --
    aggregation tuples ``(group_id, agg_func)`` and bare strings. Bare
    strings that match an existing column on ``cd.data`` are intentionally
    NOT validated here; ``transform_calc_params`` accepts them.
    """
    missing = set()
    if isinstance(node, dict):
        for value in node.values():
            missing.update(_missing_column_groups(value, available_groups))
        return missing
    if isinstance(node, tuple) and len(node) == 2:
        first, second = node
        if isinstance(first, str) and isinstance(second, str):
            # Aggregation tuple: (group_id, agg_func).
            if first not in available_groups:
                missing.add(first)
            return missing
        if callable(first) and isinstance(second, dict):
            # Calculation tuple: (func, kwargs).
            missing.update(_missing_column_groups(second, available_groups))
            return missing
    return missing


def add_am_pm_dim(df, split_time):
    """
    Tag rows of ``df`` as morning or afternoon based on a clock-time split.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with a ``DatetimeIndex``.
    split_time : str
        Clock-time string in ``"HH:MM"`` format (24-hour, leading zeros
        optional, e.g. ``"12:30"`` or ``"9:05"``). Rows whose index time is
        strictly before ``split_time`` are tagged ``"am"``; rows at or after
        ``split_time`` are tagged ``"pm"``.

    Returns
    -------
    pandas.DataFrame
        Copy of ``df`` with a new ``period`` column whose values are
        ``"am"`` or ``"pm"``.

    Raises
    ------
    ValueError
        If ``split_time`` does not match ``"HH:MM"`` or specifies an
        invalid hour/minute.
    """
    if not isinstance(split_time, str):
        raise ValueError(f"split_time must be a 'HH:MM' string; got {split_time!r}.")
    match = _SPLIT_TIME_RE.match(split_time)
    if match is None:
        raise ValueError(
            f"split_time must match 'HH:MM' (e.g. '12:30'); got {split_time!r}."
        )
    hour = int(match.group(1))
    minute = int(match.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(f"split_time hour/minute out of range; got {split_time!r}.")
    boundary_minutes = hour * 60 + minute
    out = df.copy()
    index_minutes = out.index.hour * 60 + out.index.minute
    out["period"] = np.where(index_minutes < boundary_minutes, "am", "pm")
    return out


def calc_tc_power_column(
    cd,
    tc_power_calc,
    col_name=TC_POWER_PLOT_COL,
    verbose=False,
    force_recompute=False,
):
    """
    Materialize a temperature-corrected power column for plotting only.

    Walks ``tc_power_calc`` (a calc-params nested dict using the same
    grammar as ``TEST_SETUPS`` ``reg_cols_*`` values) via
    ``captest.util.transform_calc_params`` and writes the resulting
    ``power_temp_correct`` Series to ``cd.data[col_name]`` and
    ``cd.data_filtered[col_name]``.

    This helper is intentionally isolated from
    ``CapData.process_regression_columns``: it does NOT touch
    ``cd.regression_cols``, ``cd.regression_formula``, ``cd.summary``,
    ``cd.kept``, or ``cd.removed``.

    Parameters
    ----------
    cd : CapData
        The CapData instance whose ``data`` and ``data_filtered`` will be
        extended with ``col_name``. ``power_temp_coeff`` and ``base_temp``
        attributes (propagated by ``CapTest.setup`` for shipped presets)
        are auto-injected by ``CapData.custom_param`` if not present in
        ``tc_power_calc``.
    tc_power_calc : dict
        Calc-params nested dict mirroring the bifi_power_tc preset's
        ``reg_cols_meas['power']`` value. The outermost callable must
        produce a Series of temperature-corrected power values; in
        practice this is ``calcparams.power_temp_correct``. The dict must
        contain a top-level ``"power"`` calculation tuple.
    col_name : str, default ``TC_POWER_PLOT_COL``
        Name of the column written to ``cd.data`` / ``cd.data_filtered``.
    verbose : bool, default False
        Forwarded to ``transform_calc_params``.
    force_recompute : bool, default False
        When False (default), short-circuits and returns ``col_name`` if
        the column already exists in ``cd.data``. Pass True to recompute.

    Returns
    -------
    str
        ``col_name``.

    Raises
    ------
    KeyError
        When ``tc_power_calc`` references a column-group id that is
        missing from ``cd.column_groups``.
    ValueError
        When ``tc_power_calc`` does not contain a top-level ``"power"``
        calculation tuple that produces a column in ``cd.data``.
    """
    if (not force_recompute) and (col_name in cd.data.columns):
        return col_name

    # Pre-validate column-group references so a missing group surfaces as a
    # clear KeyError instead of the AttributeError raised downstream by
    # ``CapData.agg_group`` when ``cd.loc[group_id]`` returns ``None``.
    missing_groups = sorted(
        _missing_column_groups(tc_power_calc, set(cd.column_groups.keys()))
    )
    if missing_groups:
        raise KeyError(
            f"calc_tc_power_column could not resolve column group(s) "
            f"{missing_groups}: not present in cd.column_groups. Pass an "
            f"explicit `tc_power_calc` dict that matches this CapData's "
            f"groups."
        )
    if (
        not isinstance(tc_power_calc, dict)
        or "power" not in tc_power_calc
        or not isinstance(tc_power_calc["power"], tuple)
        or len(tc_power_calc["power"]) != 2
        or not callable(tc_power_calc["power"][0])
        or not isinstance(tc_power_calc["power"][1], dict)
    ):
        raise ValueError(
            "calc_tc_power_column requires tc_power_calc to include a "
            "top-level 'power' calculation tuple, such as "
            "'power': (power_temp_correct, {...})."
        )

    # Deep-copy the calc spec so transform_calc_params doesn't mutate the
    # caller's dict in place. transform_calc_params operates by walking
    # the structure and producing a transformed return value, so we never
    # need to expose the mutated form to the caller.
    spec = copy.deepcopy(tc_power_calc)

    result = util.transform_calc_params(spec, cd, verbose=verbose)

    # ``result`` is a dict with the same shape as the input, but with the
    # calculation tuples replaced by the function names of the columns
    # they wrote to. The top-level ``power`` calculation's product (e.g.
    # ``power_temp_correct``) is the column we expose under ``col_name``.
    if not isinstance(result, dict) or "power" not in result:
        raise ValueError(
            "calc_tc_power_column requires tc_power_calc to include a "
            "top-level 'power' calculation."
        )
    produced_col = result["power"]
    if not isinstance(produced_col, str) or produced_col not in cd.data.columns:
        raise ValueError(
            "calc_tc_power_column could not identify the produced "
            "temperature-corrected power column. Ensure tc_power_calc "
            "includes a calculation tuple at the top level (e.g. "
            "'power': (power_temp_correct, {...}))."
        )

    cd.data[col_name] = cd.data[produced_col]
    if cd.data_filtered is not None:
        cd.data_filtered[col_name] = cd.data.loc[cd.data_filtered.index, produced_col]
    return col_name


# ---------------------------------------------------------------------
# ScatterPlot: param.Parameterized class powering the shipped scatter
# callables. Provides AM/PM split, temperature-corrected power view, and
# (optional) linked-timeseries pairing on a single composable surface.
# ---------------------------------------------------------------------


_VIEW_DEPENDS = (
    "split_day",
    "split_time",
    "am_color",
    "pm_color",
    "am_marker",
    "pm_marker",
    "tc_power",
    "tc_mode",
    "tc_power_calc",
    "tc_force_recompute",
    "timeseries",
    "filtered",
    "height",
    "width",
)


class ScatterPlot(param.Parameterized):
    """
    Composable scatter plot for ``CapTest`` regression diagnostics.

    Resolves x and y from ``cd.regression_formula`` (lhs vs first rhs) and
    optionally:

    - splits points into morning / afternoon glyphs (``split_day=True``),
    - swaps the y-axis to a temperature-corrected power column
      (``tc_power=True``, with mode ``replace`` / ``add_panel`` /
      ``overlay``), and / or
    - pairs the scatter with a linked timeseries panel
      (``timeseries=True``).


    Parameters
    ----------
    cd : CapData or None
        CapData instance whose ``data`` / ``column_groups`` /
        ``regression_formula`` drive the plot. Required at view time.
    filtered : bool, default True
        When True (default), pulls regression columns from
        ``cd.data_filtered``; when False, from ``cd.data``.
    split_day : bool, default False
        Render morning and afternoon points as two distinct overlaid
        Scatters with different colors and markers.
    split_time : str or None, default None
        Clock-time override (``"HH:MM"``) for the AM/PM boundary. When
        None and ``split_day=True``, the boundary is detected via
        ``captest.util.detect_solar_noon`` (idxmax of clock-time-binned
        ``ghi_mod_csky`` mean) with a 12:30 fallback.
    am_color, pm_color : str, default ``"#1f77b4"`` / ``"#d62728"``
        Glyph colors for the AM and PM Scatters when ``split_day=True``.
    am_marker, pm_marker : str, default ``"circle"`` / ``"triangle"``
        Glyph markers for the AM and PM Scatters when ``split_day=True``.
    tc_power : bool, default False
        Plot against temperature-corrected power instead of (or in
        addition to) raw power.
    tc_mode : {"replace", "add_panel", "overlay"}, default "replace"
        Layout strategy when ``tc_power=True``.
    tc_power_calc : dict or None, default None
        Calc-params nested dict that produces the tc-power column. When
        None, ``DEFAULT_TC_POWER_CALC`` is used (tuned for measured DAS
        data; sim users must override).
    tc_force_recompute : bool, default False
        When True, recomputes the tc-power column even if it already
        exists on ``cd.data``.
    timeseries : bool, default False
        Pair the principal scatter with a linked timeseries panel below.
        The timeseries panel overlays a thin gray curve of the full
        unfiltered y-series under the linked scatter of the filtered
        data so removed points remain visible as background context.
        Only valid for the single-panel ``tc_mode`` values
        (``replace`` and ``overlay``); raises ``ValueError`` if combined
        with ``tc_mode='add_panel'``.
    height, width : int, default 400 / 500
        Pixel dimensions forwarded to the Scatter / Curve options.
    """

    cd = param.Parameter(
        default=None,
        precedence=-1,
        doc="CapData instance whose data + column groups drive the plot.",
    )
    filtered = param.Boolean(default=True)

    # AM/PM split
    split_day = param.Boolean(default=False)
    split_time = param.String(default=None, allow_None=True)
    am_color = param.Color(default="#1f77b4")
    pm_color = param.Color(default="#d62728")
    am_marker = param.Selector(
        objects=["circle", "triangle", "square", "x", "diamond"],
        default="circle",
    )
    pm_marker = param.Selector(
        objects=["circle", "triangle", "square", "x", "diamond"],
        default="triangle",
    )

    # Temperature-corrected power
    tc_power = param.Boolean(default=False)
    tc_mode = param.Selector(
        objects=["replace", "add_panel", "overlay"], default="replace"
    )
    tc_power_calc = param.Dict(default=None, allow_None=True)
    tc_force_recompute = param.Boolean(default=False)

    # Timeseries pairing
    timeseries = param.Boolean(default=False)

    # Sizing
    height = param.Integer(default=400)
    width = param.Integer(default=500)

    def _require_holoviews(self):
        if hv_spec is None:
            raise ImportError(
                "holoviews is required for ScatterPlot.view. Install with "
                "`uv add holoviews` or equivalent."
            )

    def _require_cd(self):
        if self.cd is None:
            raise ValueError("ScatterPlot.cd must be set before calling view().")

    def _resolve_xy(self):
        """Return (lhs_list, rhs_list, df) from the bound CapData.

        ``df`` is ``cd.get_reg_cols(filtered_data=self.filtered)`` with
        its DatetimeIndex preserved (the index reset is performed later
        in :meth:`view`, after tc-power columns are joined and AM/PM
        tagging is applied).
        """
        lhs, rhs = util.parse_regression_formula(self.cd.regression_formula)
        df = self.cd.get_reg_cols(filtered_data=self.filtered).copy()
        return lhs, rhs, df

    def _ensure_tc_power(self, df, lhs):
        """Resolve and join the tc-power column onto ``df``.

        Returns ``(df, tc_col_name, tc_active)``. ``tc_active`` is False
        when the regression already targets a tc-power column; in that
        case ``view()`` falls back to ``tc_mode='replace'`` semantics
        without writing a redundant column.
        """
        regression_already_tc = (
            self.cd.regression_cols.get("power") == "power_temp_correct"
        )
        if regression_already_tc:
            warnings.warn(
                "Regression formula already targets temperature-corrected "
                "power; ignoring tc_power=True.",
                stacklevel=3,
            )
            return df, lhs[0], False

        spec = (
            self.tc_power_calc
            if self.tc_power_calc is not None
            else DEFAULT_TC_POWER_CALC
        )
        tc_col = calc_tc_power_column(
            self.cd,
            spec,
            col_name=TC_POWER_PLOT_COL,
            verbose=False,
            force_recompute=self.tc_force_recompute,
        )
        source = self.cd.data_filtered if self.filtered else self.cd.data
        df[tc_col] = source.loc[df.index, tc_col]
        return df, tc_col, True

    def _build_scatter(self, df, x_col, y_col, label=None):
        """Return a single ``hv.Scatter`` (no AM/PM split)."""
        scatter_kwargs = dict(
            size=5,
            tools=["hover", "lasso_select", "box_select"],
            legend_position="right",
            height=self.height,
            width=self.width,
            yformatter=NumeralTickFormatter(format="0,0"),
        )
        if label is None:
            return hv.Scatter(df, x_col, [y_col, "index"]).opts(**scatter_kwargs)
        return hv.Scatter(df, x_col, [y_col, "index"], label=label).opts(
            **scatter_kwargs
        )

    def _build_split_unified(self, df, x_col, y_col):
        """Single-CDS Scatter with categorical AM/PM color/marker.

        Returns ``(display, link_target)`` where ``display`` is an
        ``hv.Overlay`` containing one real Scatter (single
        ColumnDataSource, glyphs colored/markered per ``period`` via
        ``hv.dim`` transforms) plus two NaN-coord decoy Scatters whose
        sole purpose is to populate the bokeh legend with ``am``/``pm``
        entries (their ``apply_ranges=False`` opt keeps them out of
        autoscale). ``link_target`` is the inner real Scatter and is
        what callers pass to ``DataLink`` so the source-side plot
        exposes the ``'source'`` handle that ``DataLinkCallback``
        requires.
        """
        color_map = {"am": self.am_color, "pm": self.pm_color}
        marker_map = {"am": self.am_marker, "pm": self.pm_marker}
        real = hv.Scatter(df, x_col, [y_col, "index", "period"]).opts(
            color=hv.dim("period").categorize(color_map),
            marker=hv.dim("period").categorize(marker_map),
            size=5,
            tools=["hover", "lasso_select", "box_select"],
            height=self.height,
            width=self.width,
            yformatter=NumeralTickFormatter(format="0,0"),
            show_legend=False,
        )
        decoy_am = hv.Scatter([(np.nan, np.nan)], label="am").opts(
            color=color_map["am"],
            marker=marker_map["am"],
            size=5,
            apply_ranges=False,
        )
        decoy_pm = hv.Scatter([(np.nan, np.nan)], label="pm").opts(
            color=color_map["pm"],
            marker=marker_map["pm"],
            size=5,
            apply_ranges=False,
        )
        display = (real * decoy_am * decoy_pm).opts(
            legend_position="right",
            show_legend=True,
        )
        return display, real

    def _principal(self, df, x_col, y_col):
        """Return ``(display, link_target)`` for the principal panel.

        ``display`` is the element placed in the layout. ``link_target``
        is a single ``hv.Scatter`` (single ColumnDataSource) suitable
        for passing to :class:`DataLink`; for the no-split case it is
        identical to ``display``.
        """
        if self.split_day:
            return self._build_split_unified(df, x_col, y_col)
        scatter = self._build_scatter(df, x_col, y_col)
        return scatter, scatter

    def _attach_timeseries(self, principal_display, principal_link, df, x_col, y_col):
        """Append a linked timeseries panel below ``principal_display``.

        ``principal_link`` is the single ``hv.Scatter`` (single
        ColumnDataSource) that the timeseries scatter is data-linked
        to; ``principal_display`` is what is placed in the layout and
        may be an ``hv.Overlay`` containing legend decoys.

        The bottom panel overlays the linked scatter of the filtered
        ``y_col`` series on top of a thin gray ``hv.Curve`` of the
        full unfiltered ``y_col`` series, so points removed by
        filtering remain visible as background context.
        """
        timeseries = hv.Scatter(df, "index", [y_col, x_col]).opts(
            tools=["hover", "lasso_select", "box_select"],
            height=self.height,
            width=self.width * 2,
            yformatter=NumeralTickFormatter(format="0,0"),
            selection_fill_color="red",
            selection_line_color="red",
        )
        DataLink(principal_link, timeseries)

        # ``y_col`` is the semantic regression-formula name (e.g. ``power``)
        # which may not exist as a literal column on ``cd.data``; resolve
        # it through ``regression_cols`` to recover the underlying column
        # holding the unfiltered series.
        if y_col in self.cd.data.columns:
            full_series = self.cd.data[y_col]
        elif y_col in self.cd.regression_cols and (
            self.cd.regression_cols[y_col] in self.cd.data.columns
        ):
            full_series = self.cd.data[self.cd.regression_cols[y_col]]
        else:
            full_series = None

        if full_series is not None:
            full_df = full_series.rename(y_col).reset_index()
            full_df = full_df.rename(columns={full_df.columns[0]: "index"})
            background = hv.Curve(full_df, "index", y_col).opts(
                color="gray",
                line_width=0.75,
                height=self.height,
                width=self.width * 2,
            )
            timeseries_panel = background * timeseries
        else:
            timeseries_panel = timeseries

        return (principal_display + timeseries_panel).cols(1)

    @param.depends(*_VIEW_DEPENDS)
    def view(self):
        """
        Build and return the ``hv.Layout`` for the configured options.

        Returns
        -------
        holoviews.Layout
            A Layout whose first element is the principal scatter (a
            ``Scatter`` for the single-glyph case, an ``Overlay`` when
            ``split_day=True``). Additional panels appear when
            ``tc_mode='add_panel'`` or ``timeseries=True``.

        Raises
        ------
        ValueError
            If ``cd`` is unset, or if ``timeseries=True`` is combined with
            ``tc_mode='add_panel'``, or if ``timeseries=True`` is combined
            with ``tc_power=True`` and ``tc_mode='overlay'`` (the linked
            timeseries panel can only display a single y-series, so an
            overlaid raw + tc-power principal is ambiguous).
        ImportError
            If ``holoviews`` is not installed.
        """
        self._require_holoviews()
        self._require_cd()

        if self.timeseries and self.tc_mode == "add_panel":
            raise ValueError(
                "ScatterPlot does not support timeseries=True with "
                "tc_mode='add_panel'; pick 'replace' or 'overlay'."
            )
        if self.timeseries and self.tc_power and self.tc_mode == "overlay":
            raise ValueError(
                "ScatterPlot does not support timeseries=True with "
                "tc_power=True and tc_mode='overlay'; the timeseries "
                "panel can only display one y-series. Pick "
                "tc_mode='replace' or set tc_power=False."
            )

        lhs, rhs, df = self._resolve_xy()
        y_col = lhs[0]
        x_col = rhs[0]

        tc_active = False
        tc_col = None
        if self.tc_power:
            df, tc_col, tc_active = self._ensure_tc_power(df, lhs)

        if self.split_day:
            split_time = self.split_time or util.detect_solar_noon(self.cd.data)
            df = add_am_pm_dim(df, split_time)

        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "index"})

        raw_display, raw_link = self._principal(df, x_col, y_col)
        if tc_active:
            tc_display, tc_link = self._principal(df, x_col, tc_col)
        else:
            tc_display, tc_link = None, None

        if not tc_active or self.tc_mode == "replace":
            display = tc_display if tc_active else raw_display
            link = tc_link if tc_active else raw_link
            layout = hv.Layout([display])
            principal_y = tc_col if tc_active else y_col
        elif self.tc_mode == "add_panel":
            layout = hv.Layout([raw_display, tc_display])
            link = raw_link  # not used: timeseries=True is guarded above
            principal_y = y_col
        else:  # overlay (tc_active=True; timeseries=True is guarded above)
            display = raw_display * tc_display
            layout = hv.Layout([display])
            link = raw_link
            principal_y = y_col

        if self.timeseries:
            return self._attach_timeseries(display, link, df, x_col, principal_y)

        return layout


class ScatterBifiPowerTc(ScatterPlot):
    """
    Two-panel scatter for the ``bifi_power_tc`` preset.

    The ``bifi_power_tc`` regression formula is ``power ~ poa + rpoa``
    where ``power`` is already temperature-corrected. This subclass
    builds one panel per rhs variable (``power vs poa`` and
    ``power vs rpoa``). The ``tc_power`` parameter is ignored here
    because the regression power is already tc-corrected; setting it to
    True emits a ``UserWarning``.

    AM/PM splitting and timeseries pairing are inherited from
    :class:`ScatterPlot`. When ``timeseries=True``, only the first panel
    is paired with a linked timeseries view to keep the layout sane.
    """

    @param.depends(*_VIEW_DEPENDS)
    def view(self):
        """
        Build a two-panel ``hv.Layout`` for the bifi_power_tc preset.

        Returns
        -------
        holoviews.Layout
        """
        self._require_holoviews()
        self._require_cd()

        if self.tc_power:
            warnings.warn(
                "ScatterBifiPowerTc ignores tc_power=True; the regression "
                "power column is already temperature-corrected.",
                stacklevel=2,
            )

        lhs, rhs, df = self._resolve_xy()
        y_col = lhs[0]

        if self.split_day:
            split_time = self.split_time or util.detect_solar_noon(self.cd.data)
            df = add_am_pm_dim(df, split_time)

        df = df.reset_index()
        df = df.rename(columns={df.columns[0]: "index"})

        panel_pairs = [self._principal(df, x_col, y_col) for x_col in rhs]
        displays = [d for d, _ in panel_pairs]
        links = [link_target for _, link_target in panel_pairs]

        if self.timeseries:
            paired = self._attach_timeseries(displays[0], links[0], df, rhs[0], y_col)
            # ``paired`` is a Layout; combine with the remaining panels.
            remaining = displays[1:]
            if remaining:
                return hv.Layout([paired, *remaining])
            return paired

        return hv.Layout(displays)
