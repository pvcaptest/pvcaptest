from collections import UserDict

"""
Todo
- Create a function to create an instance from excel file
- Create a function to create an instance from json
- Create a function to create an instance from yaml
- Create a function to create an instance from DataFrame columns?
"""

class ColumnGroups(UserDict):
    def __setitem__(self, key, value):
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
