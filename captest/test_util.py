import numpy as np
import pandas as pd
from scipy import stats


def generate_irr_distribution(
    lowest_irr,
    highest_irr,
    rng=np.random.default_rng(82)
):
    """
    Create a list of increasing values similar to POA irradiance data.

    Default parameters result in increasing values where the difference
    between each subsquent value is randomly chosen from the typical range
    of steps for a POA tracker.

    Parameters
    ----------
    lowest_irr : numeric
        Lowest value in the list of values returned.
    highest_irr : numeric
        Highest value in the list of values returned.
    rng : Numpy Random Generator
        Instance of the default Generator.

    Returns
    -------
    irr_values : list
    """
    irr_values = [lowest_irr, ]
    possible_steps = (
        rng.integers(1, high=8, size=10000)
        + rng.random(size=10000)
        - 1
    )
    below_max = True
    while below_max:
        next_val = irr_values[-1] + rng.choice(possible_steps, replace=False)
        if next_val >= highest_irr:
            below_max = False
        else:
            irr_values.append(next_val)
    return irr_values
