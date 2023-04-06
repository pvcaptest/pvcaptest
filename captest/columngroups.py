import collections

class ColumnGroups(collections.UserDict):
    def __setitem__(self, key, value):
        # key = (key.replace('-', '_')
        # )
        setattr(self, key, value)
        super().__setitem__(key, value)

    def __repr__(self):
        """Print `column_groups` dictionary with nice formatting."""
        output = ''
        for grp_id, col_list in self.data.items():
            output += grp_id + ':\n'
            for col in col_list:
                output += ' ' * 4 + col + '\n'
        return output

# The search strings for types cannot be duplicated across types.
type_defs = collections.OrderedDict([
    ('irr', ['irradiance', 'irr', 'plane of array', 'poa', 'ghi',
              'global', 'glob', 'w/m^2', 'w/m2', 'w/m', 'w/']),
    ('temp', ['temperature', 'temp', 'degrees', 'deg', 'ambient',
               'amb', 'cell temperature', 'TArray']),
    ('wind', ['wind', 'speed']),
    ('pf', ['power factor', 'factor', 'pf']),
    ('op_state', ['operating state', 'state', 'op', 'status']),
    ('real_pwr', ['real power', 'ac power', 'e_grid']),
    ('shade', ['fshdbm', 'shd', 'shade']),
    ('pvsyt_losses', ['IL Pmax', 'IL Pmin', 'IL Vmax', 'IL Vmin']),
    ('index', ['index']),
])

sub_type_defs = collections.OrderedDict([
    ('ghi', ['sun2', 'global horizontal', 'ghi', 'global', 'GlobHor']),
    ('poa', ['sun', 'plane of array', 'poa', 'GlobInc']),
    ('amb', ['TempF', 'ambient', 'amb']),
    ('mod', ['Temp1', 'module', 'mod', 'TArray']),
    ('mtr', ['revenue meter', 'rev meter', 'billing meter', 'meter']),
    ('inv', ['inverter', 'inv']),
])

irr_sensors_defs = {
    'ref_cell': ['reference cell', 'reference', 'ref', 'referance', 'pvel'],
    'pyran': ['pyranometer', 'pyran'],
    'clear_sky': ['csky']
}

def series_type(series, type_defs):
    """
    Assign columns to a category by analyzing the column names.

    The type_defs parameter is a dictionary which defines search strings
    for each key, where the key is a categorical name
    and the search strings are possible related names.  For example an
    irradiance sensor has the key 'irr' with search strings 'irradiance'
    'plane of array', 'poa', etc.

    Parameters
    ----------
    series : pandas series
        Row or column of dataframe passed by pandas.df.apply.
    type_defs : dictionary
        Dictionary with the following structure.  See type_defs
        {'category abbreviation': [category search strings]}

    Returns
    -------
    string
        Returns a string representing the category for the series.
    """
    for key, search_strings in type_defs.items():
        # print('################')
        # print(key)
        for search_str in search_strings:
            # print(search_str)
            if series.name.lower().find(search_str.lower()) == -1:
                continue
            else:
                return key
    return ''

def group_columns(data):
    """
    Create a dict of raw column names paired to categorical column names.

    Uses multiple type_def formatted dictionaries to determine the type,
    sub-type, and equipment type for data series of a dataframe.  The
    determined types are concatenated to a string used as a dictionary key
    with a list of one or more original column names as the paired value.

    Parameters
    ----------
    data : DataFrame
        Data with columns to group.

    Returns
    -------
    cg : ColumnGroups

    Todo
    ----
    type_defs parameter
        Consider refactoring to have a list of type_def dictionaries as an
        input and loop over each dict in the list.
    """
    col_types = data.apply(series_type, args=(type_defs,)).tolist()
    sub_types = data.apply(series_type, args=(sub_type_defs,)).tolist()
    irr_types = data.apply(series_type, args=(irr_sensors_defs,)).tolist()

    col_indices = []
    for typ, sub_typ, irr_typ in zip(col_types, sub_types, irr_types):
        col_indices.append('_'.join([typ, sub_typ, irr_typ]))

    names = []
    for new_name, old_name in zip(col_indices, data.columns.tolist()):
        names.append((new_name, old_name))
    names.sort()
    orig_names_sorted = [name_pair[1] for name_pair in names]

    trans = {}
    col_indices.sort()
    cols = list(set(col_indices))
    cols.sort()
    for name in set(cols):
        start = col_indices.index(name)
        count = col_indices.count(name)
        trans[name] = orig_names_sorted[start:start + count]

    return ColumnGroups(trans)
