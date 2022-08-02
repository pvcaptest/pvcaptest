
class ColumnGroups(object):
    """
    Class with a dictionary of column groups and related functionality.

    Parameters
    ----------
    column_groups : dict
    """
    def __init__(self, column_groups):
        super(ColumnGroups, self).__init__()
        self.column_groups = column_groups
        self.assign_column_groups(column_groups)

    def assign_column_groups(self, column_groups):
        for grp_id, cols in column_groups.items():
            setattr(self, grp_id.replace('-', '_'), cols)
