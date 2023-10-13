import copy
import numpy as np
import pandas as pd
import panel as pn
from panel.interact import fixed
import holoviews as hv
from holoviews import opts
import colorcet as cc

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
    for group in tags_to_plot:
        plot = plot_tag(data, group)
        group_plots.append(plot)
    return hv.Layout(group_plots).cols(1)


def custom_plot_dboard(cd=None, cg=None, data=None):
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
    """
    if cd is not None:
        data = cd.data
        cg = cd.column_groups
    # setup custom plot for 'Custom' tab
    groups = msel_from_column_groups(cg)
    tags = msel_from_column_groups(cg, groups=False)

    custom_plot_name = pn.widgets.TextInput() 
    update = pn.widgets.Button(name='Update')

    custom_plot = pn.Column(
        pn.Row(groups, tags, pn.Column(custom_plot_name, update)),
        pn.Row(pn.bind(plot_group_tag_overlay, data, groups, tags))
    )

    # setup group plotter for 'Main' tab
    cg_layout = copy.deepcopy(cg)
    main_ms = msel_from_column_groups(cg_layout)
    bound_update = pn.bind(
        update_cg_layout,
        main_ms,
        custom_plot_name,
        cg_layout,
        groups,
        tags,
    )
    def ucgl(event):
        bound_update
    update.on_click(ucgl)
    main_plot = pn.Column(
        pn.Row(main_ms),
        pn.Row(pn.bind(plot_tag_groups, data, main_ms))
    )

    # layout dashboard
    plotter = pn.Tabs(
        ('Main', main_plot),
        ('Custom', custom_plot),
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


def update_cg_layout(mselect, name, column_groups, group_tags, column_tags):
    """
    Update the column groups layout.

    Parameters
    ----------
    mselect : pn.widgets.MultiSelect
        The multi-select widget.
    name : str
        The name of the new group.
    column_groups : ColumnGroups
        The column groups object.
    group_tags : list of str
        The tags to plot from the groups selected.
    column_tags : list of str
        The tags to plot from the individually selected columns.
    """
    column_groups_ = copy.deepcopy(column_groups)
    column_groups_ = add_custom_plot(name, column_groups_, group_tags, column_tags)
    mselect.options = column_groups_
