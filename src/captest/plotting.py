from pathlib import Path
import copy
import json
import warnings
import itertools
from functools import partial
import importlib

import numpy as np
import pandas as pd
from bokeh.models import NumeralTickFormatter

from .util import tags_by_regex, read_json

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
else:
    warnings.warn("The plotting methods will not work without the holoviews package.")

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
    **kwargs,
):
    """
    Create plotting dashboard.

    NOTE: If a 'plot_defaults.json' file exists in the same directory as the file this
    function is called from called, then the default groups will be read from that file
    instead of using the `default_groups` argument. Delete or manually edit the file to
    change the default groups. Use the `default_groups` or manually edit the file to
    control the order of the plots.

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
    **kwargs : optional
        Pass additional keyword arguments to the holoviews options of the scatter plot
        on the 'Scatter' tab.
    """
    if cd is not None:
        data = cd.data
        cg = cd.column_groups
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
        with open("./plot_defaults.json", "w") as file:
            json.dump(main_ms.value, file)

    plots_to_layout.on_click(set_defaults)

    # setup default groups
    if Path("./plot_defaults.json").exists():
        default_tags = read_json("./plot_defaults.json")
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
