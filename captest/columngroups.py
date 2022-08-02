
class ColumnGroups(object):
    """
    Class with a dictionary of column groups and related functionality.

    Parameters
    ----------
    column_groups : dict

    Todo
    ----
    - Convert to a parametrized class with param library
    - Create a function or method to create instance from excel file
    - Create a function or method to create instance from json
    - Create a function or method to create instance from yaml
    - Create a function or method to create instance from DataFrame columns?
    """
    def __init__(self, column_groups):
        super(ColumnGroups, self).__init__()
        self.column_groups = column_groups
        self.assign_column_groups(column_groups)

    def __call__(self):
        return self.column_groups

    def __str__(self):
        """Print `column_groups` dictionary with nice formatting."""
        output = ''
        for grp_id, col_list in self.column_groups.items():
            output += grp_id + ':\n'
            for col in col_list:
                output += ' ' * 4 + col + '\n'
        return output

    def assign_column_groups(self, column_groups):
        for grp_id, cols in column_groups.items():
            # setattr(self, grp_id.replace('-', '_'), cols)
            setattr(self, grp_id, cols)
