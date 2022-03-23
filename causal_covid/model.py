import logging

log = logging.getLogger(__name__)
import argparse
import datetime
import sys
import pymc3 as pm
import theano.tensor as tt
import pickle
import pandas as pd
import numpy as np
import scipy.special
from multiprocessing import cpu_count


sys.path.append("../covid19_inference")
sys.path.append("..")

import data
from .utils import get_cps, day_to_week_matrix, day_to_week_transform
from covid19_inference import Cov19Model
from covid19_inference.model import (
    lambda_t_with_sigmoids,
    kernelized_spread_with_interaction,
    kernelized_spread,
    week_modulation,
    student_t_likelihood,
    delay_cases,
    uncorrelated_prior_E,
)


import covid19_inference


def create_model_multidmensional(
    cases_df, N_population,
):
    """
    Creates the variant model with different compartments

    Parameters
    ----------
    likelihood : str
        Likelihood function for the variant data, possible : ["beta","binom","dirichlet","multinomial"]

    spreading_dynamics : str
        Type of spreading dynamics to use possible : ["SIR","kernelized_spread"]

    variants : pd.DataFrame
        Data array variants

    new_cases : pd.DataFrame
        Data array cases

    Returns
    -------
    model

    """

    # new_cases_obs= data.load_cases(file = "/home/jdehning/repositories/causal_covid/data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/observed_infection_data.csv",
    # )
    new_cases_obs = np.array(cases_df)
    num_age_groups = new_cases_obs.shape[1]

    data_begin = cases_df.index[0] - datetime.timedelta(days=6)
    data_end = cases_df.index[-1]

    # Params for the model
    params = {
        "new_cases_obs": new_cases_obs,
        "data_begin": data_begin,
        "data_end": data_end,
        "fcast_len": 14,
        "diff_data_sim": 14,
        "N_population": N_population,
    }

    pr_delay = 10

    pr_median_lambda = 1.0

    with Cov19Model(**params) as this_model:

        # Get base reproduction number/spreading rate
        R_t_log = lambda_t_with_sigmoids(
            change_points_list=get_cps(
                this_model.data_begin,
                this_model.sim_end,
                interval=14,
                pr_median_transient_len=6,
                pr_sigma_transient_len=2,
            ),
            pr_median_lambda_0=pr_median_lambda,
            name_lambda_t="base_R_t",
        )

        E_begin = uncorrelated_prior_E(n_data_points_used=2) / 7

        C = np.array([[1, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])

        # Put the lambdas together unknown and known into one tensor (shape: t,v)
        new_cases = kernelized_spread_with_interaction(
            R_t_log=R_t_log,
            interaction_matrix=C,
            num_groups=this_model.sim_shape[-1],
            pr_new_E_begin=E_begin,
        )  # has shape (num_days, num_age_groups)

        # Transform to weekly cases and add a delay of 6 days
        weekly_cases = day_to_week_transform(
            new_cases,
            arr_begin=this_model.sim_begin,
            arr_end=this_model.sim_end,
            weeks=cases_df.index,
            end=True,
            additional_delay=6,
        )

        weekly_cases = pm.Deterministic("weekly_cases", weekly_cases)

        sigma_obs = pm.HalfCauchy("sigma_obs", beta=300, shape=num_age_groups,)
        sigma = (
            tt.abs_(weekly_cases + 1) ** 0.5 * sigma_obs
        )  # offset and tt.abs to avoid nans
        data_obs = this_model.new_cases_obs
        pm.StudentT(
            name="_new_cases_studentT",
            nu=4,
            mu=weekly_cases[~np.isnan(data_obs)],
            sigma=sigma[~np.isnan(data_obs)],
            observed=data_obs[~np.isnan(data_obs)],
        )

        return this_model


def create_model_single_dimension(cases_df, N_population):
    new_cases_obs = np.array(cases_df)
    data_begin = cases_df.index[0] - datetime.timedelta(days=6)
    data_end = cases_df.index[-1]

    # Params for the model
    params = {
        "new_cases_obs": new_cases_obs,
        "data_begin": data_begin,
        "data_end": data_end,
        "fcast_len": 14,
        "diff_data_sim": 14,
        "N_population": N_population,
    }

    pr_delay = 10

    pr_median_lambda = 1.0

    with Cov19Model(**params) as this_model:

        # Get base reproduction number/spreading rate
        R_t_log = lambda_t_with_sigmoids(
            change_points_list=get_cps(
                this_model.data_begin,
                this_model.sim_end,
                interval=14,
                pr_median_transient_len=6,
                pr_sigma_transient_len=2,
            ),
            pr_median_lambda_0=pr_median_lambda,
            name_lambda_t="base_R_t",
        )

        E_begin = uncorrelated_prior_E(n_data_points_used=2) / 7

        # Put the lambdas together unknown and known into one tensor (shape: t,v)
        new_cases = kernelized_spread(
            lambda_t_log=R_t_log, pr_new_E_begin=E_begin,
        )  # has shape (num_days, num_age_groups)

        # Transform to weekly cases and add a delay of 6 days
        weekly_cases = day_to_week_transform(
            new_cases,
            arr_begin=this_model.sim_begin,
            arr_end=this_model.sim_end,
            weeks=cases_df.index,
            end=True,
            additional_delay=6,
        )

        weekly_cases = pm.Deterministic("weekly_cases", weekly_cases)

        sigma_obs = pm.HalfCauchy("sigma_obs", beta=100)
        sigma = (
            tt.abs_(weekly_cases + 1) ** 0.5 * sigma_obs
        )  # offset and tt.abs to avoid nans
        data_obs = this_model.new_cases_obs
        pm.StudentT(
            name="_new_cases_studentT",
            nu=4,
            mu=weekly_cases[~np.isnan(data_obs)],
            sigma=sigma[~np.isnan(data_obs)],
            observed=data_obs[~np.isnan(data_obs)],
        )

        return this_model


def create_model_single_dimension_infectiability(
    cases_df, infectiability_df, N_population
):
    new_cases_obs = np.squeeze(np.array(cases_df))
    assert new_cases_obs.ndim == 1
    data_begin = cases_df.index[0] - datetime.timedelta(days=6)
    data_end = cases_df.index[-1]

    diff_data_sim = (cases_df.index[0] - infectiability_df.index[0]).days
    if diff_data_sim < 10 + 6:
        raise RuntimeError("Not enough days before the begin of data")

    # Params for the model
    params = {
        "new_cases_obs": new_cases_obs,
        "data_begin": data_begin,
        "data_end": data_end,
        "fcast_len": 0,
        "diff_data_sim": diff_data_sim,
        "N_population": N_population,
    }

    pr_delay = 10

    pr_median_lambda = 2.0

    with Cov19Model(**params) as this_model:

        # Infectiability
        infectiability_log = np.log(np.squeeze(np.array(infectiability_df)))
        assert infectiability_log.ndim == 1
        infectiability_log = day_to_week_matrix(
            this_model.sim_begin, this_model.sim_end, infectiability_df.index, end=True
        ).dot(infectiability_log)
        # Get base reproduction number/spreading rate
        R_t_log_base = lambda_t_with_sigmoids(
            change_points_list=get_cps(
                this_model.data_begin,
                this_model.sim_end,
                interval=14,
                pr_median_transient_len=6,
                pr_sigma_transient_len=2,
            ),
            pr_median_lambda_0=pr_median_lambda,
            name_lambda_t="base_R_t",
        )
        R_t_log_eff = R_t_log_base + infectiability_log

        pm.Deterministic("eff_R_t", tt.exp(R_t_log_eff))

        E_begin = uncorrelated_prior_E(n_data_points_used=2) / 7

        # Put the lambdas together unknown and known into one tensor (shape: t,v)
        new_cases = kernelized_spread(
            lambda_t_log=R_t_log_eff, pr_new_E_begin=E_begin,
        )  # has shape (num_days, num_age_groups)

        # Transform to weekly cases and add a delay of 6 days
        weekly_cases = day_to_week_transform(
            new_cases,
            arr_begin=this_model.sim_begin,
            arr_end=this_model.sim_end,
            weeks=cases_df.index,
            end=True,
            additional_delay=6,
        )

        weekly_cases = pm.Deterministic("weekly_cases", weekly_cases)

        sigma_obs = pm.HalfCauchy("sigma_obs", beta=100)
        sigma = (
            tt.abs_(weekly_cases + 1) ** 0.5 * sigma_obs
        )  # offset and tt.abs to avoid nans
        data_obs = this_model.new_cases_obs
        pm.StudentT(
            name="_new_cases_studentT",
            nu=4,
            mu=weekly_cases[~np.isnan(data_obs)],
            sigma=sigma[~np.isnan(data_obs)],
            observed=data_obs[~np.isnan(data_obs)],
        )

        return this_model
