import logging
import pandas as pd
import numpy as np
from datetime import timedelta

import theano.tensor as tt

log = logging.getLogger(__name__)

# Define changepoints (we define all vars to surpress the automatic prints)
def get_cps(data_begin, data_end, interval=7, offset=0, **priors_dict):
    """
    Generates and returns change point array.
    
    Parameters
    ----------
    data_begin : dateteime
        First date for possible changepoints
    data_end : datetime
        Last date for possible changepoints
    interval : int, optional
        Interval for the proposed cp in days. default:7
    offset : int, optional
        Offset for the first cp to data_begin
    """
    change_points = []
    count = interval - offset
    default_params = dict(
        pr_sigma_date_transient=1.5,
        pr_sigma_lambda=0.2,  # wiggle compared to previous point
        relative_to_previous=True,
        pr_factor_to_previous=1.0,
        pr_sigma_transient_len=1,
        pr_median_transient_len=4,
        pr_median_lambda=0.125,
    )
    set_missing_priors_with_default(priors_dict, default_params)

    for day in pd.date_range(start=data_begin, end=data_end):
        if count / interval >= 1.0:
            # Add cp

            change_points.append(
                dict(  # one possible change point every sunday
                    pr_mean_date_transient=day, **priors_dict
                )
            )
            count = 1
        else:
            count = count + 1
    return change_points


def set_missing_priors_with_default(priors_dict, default_priors):
    """
        Takes a dict with custom priors and a dict with defaults and sets keys that
        are not given
    """
    for prior_name in priors_dict.keys():
        if prior_name not in default_priors:
            log.warning(f"Prior with name {prior_name} not known")

    for prior_name, value in default_priors.items():
        if prior_name not in priors_dict:
            priors_dict[prior_name] = value
            log.info(f"{prior_name} was set to default value {value}")


def day_to_week_transform(
    arr, arr_begin, arr_end, weeks, end=False, additional_delay=0
):
    """
    Transforms an array with daily observations to one with weekly observations
    Parameters
    ----------
    arr: the array to transform with daily observations
    arr_begin: datetime, day of the first element of arr
    arr_end: datetime, day of the last element of arr
    weeks: array of timedelta days at which the returned variable should have its weeks
    end: whether the days in weeks encode the beginning or the end of the week
    additional_delay: int: additional delay.

    Returns
    -------

    """
    days = pd.date_range(arr_begin, arr_end)
    if end:
        end_week = weeks - timedelta(days=additional_delay)
        begin_week = weeks - timedelta(days=7) - timedelta(days=additional_delay)
    else:
        begin_week = weeks - timedelta(days=1) - timedelta(days=additional_delay)
        end_week = weeks + timedelta(days=6) - timedelta(days=additional_delay)

    i_beg = []
    for d in begin_week:
        i_beg.append(days.get_loc(d))
    i_end = []
    for d in end_week:
        i_end.append(days.get_loc(d))
    cumsum_arr = tt.cumsum(arr, axis=0)
    transformed = cumsum_arr[i_end] - cumsum_arr[i_beg]
    return transformed


def day_to_week_matrix(sim_begin, sim_end, weeks, fill=False, end=False):
    """
    Returns the matrix mapping a day to an week.
    Does more or less the same as pandas resample but we can use it in 
    the model.
    
    Parameters
    ----------
    sim_begin : datetime
    sim_end : datetime
    weeks : array-like, datetimes
        Begining date of week. Normally variants.index
    fill : bool
        Wheater or not to fill the not defined datapoints with ones
    Interval
    [first_week_day,first_week_day+7)
    """
    days = pd.date_range(sim_begin, sim_end)
    m = np.zeros((len(days), len(weeks)))
    for i, d in enumerate(days):
        for j, week_begin in enumerate(weeks):
            week_end = week_begin + timedelta(days=7)
            if end:
                week_end = week_begin + timedelta(days=1)
                week_begin = week_end - timedelta(days=7)
            if d >= week_begin and d < week_end:
                m[i, j] = 1

        if fill:
            if d < weeks[0]:
                m[i, 0] = 1
            if d >= weeks[-1]:
                m[i, -1] = 1
    return m
