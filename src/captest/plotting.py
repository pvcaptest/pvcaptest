import copy
import warnings
import itertools
import numpy as np
import pandas as pd
import panel as pn
from panel.interact import fixed
import holoviews as hv
from holoviews import opts
import colorcet as cc

from .util import tags_by_regex, append_tags

COMBINE = {
    'poa_ghi': 'irr.*(poa|ghi)$',
    'poa_csky': '(?=.*poa)(?=.*irr)',
    'ghi_csky': '(?=.*ghi)(?=.*irr)',
    'inv_sum_mtr_pwr': ['(?=.*real)(?=.*pwr)(?=.*mtr)', '(?=.*pwr)(?=.*agg)'],
}


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
                warnings.userwarning(
                    'When passing a list of regex. There should be two strings. One for '
                    'identifying groups and one for identifying individual tags (columns).'
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
        # options = list(column_groups.keys())
        options = column_groups.data
        name = 'Groups'
        value = column_groups.data[list(column_groups.keys())[0]]
    else:
        options = []
        for columns in column_groups.values():
            options += columns
        name = 'Columns'
        value = [options[0]]
    return pn.widgets.MultiSelect(
        name=name,
        value=value,
        options=options,
        size=8,
        height=400,
        width=400
    )


def plot_tag(data, tag, width=1500, height=400):
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
        plot = hv.Curve(pd.DataFrame(
            {'no_data': [np.NaN] * data.shape[0]},
            index=data.index
        ))
    plot.opts(
        opts.Curve(
            line_width=1,
            width=width,
            muted_alpha=0,
            height=height,
            tools=['hover']
        ),
        opts.NdOverlay(width=width, height=height, legend_position='right')
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


def plot_tag_groups(data, tags_to_plot):
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
        plot = plot_tag(data, group)
        group_plots.append(plot)
    return hv.Layout(group_plots).cols(1)


def plot(cd=None, cg=None, data=None, combine=COMBINE):
    """
    Create plotting dashboard.

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
    """
    if cd is not None:
        data = cd.data
        cg = cd.column_groups
    # setup custom plot for 'Custom' tab
    groups = msel_from_column_groups(cg)
    tags = msel_from_column_groups({'all_tags': list(data.columns)}, groups=False)
    re_input = pn.widgets.TextInput(name='Input regex to filter columns list')

    def update_ms(event=None):
        if re_input.value == '':
            re_value = '.*'
        else:
            re_value = re_input.value
        options = data.filter(regex=re_value).sort_index(axis=1).columns.to_list()
        tags.param.update(options=options)
    re_input.param.watch(update_ms, 'value')

    custom_plot_name = pn.widgets.TextInput() 
    update = pn.widgets.Button(name='Update')

    custom_plot = pn.Column(
        pn.Row(
            pn.Column(
                pn.Row(custom_plot_name, update),
                groups,
            ),
            pn.WidgetBox(re_input, tags)
        ),
        pn.Row(pn.bind(plot_group_tag_overlay, data, groups, tags))
    )

    # setup group plotter for 'Main' tab
    cg_layout = parse_combine(combine, cg, data)
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
    main_plot = pn.Column(
        pn.Row(main_ms),
        pn.Row(pn.bind(plot_tag_groups, data, main_ms))
    )

    default_tags = [
        cg['irr_poa'],
        cg['irr_ghi'],
    ]
    # layout dashboard
    plotter = pn.Tabs(
        ('Plots', plot_tag_groups(data, default_tags)),
        ('Layout', main_plot),
        ('Overlay', custom_plot),
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
