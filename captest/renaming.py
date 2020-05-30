"""
A class and functions for effeciently cleaning up column names.

Example syntax to chain methods of ColumnRenamer:

(cr.make_lowercase()
   .clean_terms(cleaners=cleaners['also_energy_names'])
   .drop_parentheses()
   .remove_spaces()
   .strip_chars(chars_to_remove=['_']))


Method to chain functions without class:

rename_dict = rn.remove_spaces(*rn.clean_terms
                               (*rn.shorten_names
                                (*rn.strip_whitespace
                                 (*rn.drop_parentheses
                                  (*rn.remove_prefix
                                   (*rn.make_lowercase(das.data.columns)))))),
                                    cleaners=rn.cleaners['also_energy_names'])
"""


from functools import wraps
import re

abbreviatons = {'temperature': 'temp',
                'irradiance': 'irrad',
                'global horizontal': 'GHI',
                'plane of array': 'POA',
                'pyranometer': 'pyran',
                'celsius': 'degC',
                'fahrenheit': 'degF',
                'module temperature': 'BOM_temp',
                'inverter': 'inv',
                'ambient': 'amb',
                'degrees': 'deg'}

also_energy_names = {'sun': 'POA irrad',
                     'sun2': 'GHI irrad',
                     'temp1': 'BOM temp',
                     'tempf': 'ambient temp'}

undescriptive_terms = ['weather', 'probe']

cleaners = {'abbreviations': abbreviatons,
            'undescriptive_terms': undescriptive_terms,
            'also_energy_names': also_energy_names}


def translate_column_names(func):
    """
    Decorator that maintains a translation dictionary when applying functions to modify column names.

    Decoarted functions should take a list-like of strings representing column names
    and return a list of of strings of modified column names.

    Parameters
    ----------
    columns : list-like
        List or Index of column names. Column names are expected to be strings.

    column_translation : dict, default none
        Dictionary translating the original column names to the new column
        names.

    Returns
    -------
    new_columns : list-like
        New column names.
    column_translation : dict
        Dictionary translating original column names passed as keys and the new
        column names as values.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        column_translation = None

        if len(args) == 1:
            columns = args[0]
        elif len(args) == 2:
            columns = args[0]
            column_translation = args[1]

        new_columns = func(columns, **kwargs)

        if column_translation is None:
            column_translation = {old_col: new_col for old_col, new_col in zip(columns, new_columns)}
        else:
            column_translation = {orig_col: new_col for orig_col, new_col in zip(column_translation.keys(), new_columns)}

        return new_columns, column_translation
    return wrapper


@translate_column_names
def remove_prefix(columns):
    """
    Remove a common prefix from column names.

    Parameters
    ----------
    columns : list-like
        List or Index of column names. Column names are expected to be strings.

    Returns
    -------
    new_columns : list-like
        New column names.
    column_translation : dict
        Dictionary translating original column names passed as keys and the new column names as values.

    """
    characters_in_column_titles = [list(col) for col in columns]
    titles_df = pd.DataFrame(characters_in_column_titles)

    prefix_bool_df = pd.DataFrame(titles_df.apply(lambda x: x.nunique() == 1))
    prefix_bool_df['shift'] = prefix_bool_df.shift(1)
    prefix_bool_df['end_prefix'] = prefix_bool_df.apply(lambda x: x.iloc[0] is False and x.iloc[1] is True, axis=1)

    prefix_end = prefix_bool_df[prefix_bool_df['end_prefix']].index[0]
    new_columns = [col[prefix_end:] for col in columns]

    return new_columns


@translate_column_names
def drop_parentheses(columns):
    new_columns = [re.sub("[\(\[].*?[\)\]]", "", col) for col in columns]
    return new_columns


@translate_column_names
def clean_terms(columns, cleaners=None):
    """
    Removes or replaces terms in column names.

    Common cleaners are defined in the cleaners dictionary.

    Parameters
    ----------
    columns : list-like
        List or Index of column names. Column names are expected to be strings.
    cleaners : list or dict
        If a list the terms passed will be removed from column names.
        If a dict the keys passed will be replaced by the values.
    """
    new_columns = columns.copy()
    if isinstance(cleaners, dict):
        for long_name, short_name in cleaners.items():
            new_columns = [col.replace(long_name, short_name) for col in new_columns]
    if isinstance(cleaners, list):
        for term in cleaners:
            new_columns = [col.replace(term, '') for col in new_columns]

    return new_columns


@translate_column_names
def strip_chars(columns, chars_to_remove=None):
    """
    Removes characters from beginning and end of column names.

    Parameters:
    -----------
    columns : list-like
        List or Index of column names. Column names are expected to be strings.
    chars_to_remove : list, default None
        Default removes white space.
        Pass list of characters to remove.
    """
    new_columns = columns.copy()
    if isinstance(chars_to_remove, list):
        for char in chars_to_remove:
            new_columns = [col.strip(char) for col in new_columns]
    else:
        new_columns = [col.strip() for col in new_columns]
    return new_columns


@translate_column_names
def make_lowercase(columns):
    new_columns = [col.lower() for col in columns]
    return new_columns


@translate_column_names
def remove_spaces(columns):
    new_columns = [col.replace(' ', '_') for col in columns]
    return new_columns


@translate_column_names
def remove_symbols(columns):
    new_columns = [re.sub("[-,\(\)]", "", col) for col in columns]
    return new_columns


@translate_column_names
def remove_duplicate_white_space(columns):
    new_columns = [re.sub(" +", " ", col) for col in columns]
    return new_columns

class ColumnRenamer(object):
    def __init__(self, dataframe):
        super(ColumnRenamer, self).__init__()
        self.orig_dataframe = dataframe
        self.orig_columns = list(dataframe.columns)
        self.new_columns = list(dataframe.columns)
        self.translation = dict()

    def __repr__(self):
        return ' \n'.join(self.new_columns)

    def set_translation(self):
        self.translation = {old_col: new_col for old_col, new_col in zip(self.orig_columns, self.new_columns)}

    def drop_parentheses(self):
        self.new_columns = drop_parentheses(self.new_columns)[0]
        self.set_translation()
        return self

    def make_lowercase(self):
        self.new_columns = make_lowercase(self.new_columns)[0]
        self.set_translation()
        return self

    def clean_terms(self, cleaners=None):
        self.new_columns = clean_terms(self.new_columns, cleaners=cleaners)[0]
        self.set_translation()
        return self

    def remove_prefix(self):
        self.new_columns = remove_prefix(self.new_columns)[0]
        self.set_translation()
        return self

    def strip_chars(self, chars_to_remove=None):
        self.new_columns = strip_chars(self.new_columns, chars_to_remove=chars_to_remove)[0]
        self.set_translation()
        return self

    def remove_spaces(self):
        self.new_columns = remove_spaces(self.new_columns)[0]
        self.set_translation()
        return self

    def remove_symbols(self):
        self.new_columns = remove_symbols(self.new_columns)[0]
        self.set_translation()
        return self

    def get_renamed_df(self):
        return self.orig_dataframe.rename(columns=self.translation)

    def reset(self):
        self.new_columns = self.orig_columns.copy()
