import param

class ColumnGroups(param.Parameterized):
    """
    Parametrized class with a dictionary of column groups and related functionality.

    Todo
    ----
    - Create a function or method to create instance from excel file
    - Create a function or method to create instance from json
    - Create a function or method to create instance from yaml
    - Create a function or method to create instance from DataFrame columns?
    """
    column_groups = param.Dict(
        doc='Column groups dictionary with string group id keys and lists '
            'of column names for values.'
    )

    def __init__(self, column_groups, **params):
        super().__init__(**params)
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
