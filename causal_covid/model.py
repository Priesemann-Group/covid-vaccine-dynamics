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


sys.path.append("..")

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
from causal_covid.data import load_contact_matrix_mistri


import covid19_inference


def create_model_multidimensional(
    cases_df, infectiability_df, N_population, C_mat_param, influx_inci
):
    """
    Creates the variant model with different compartments

    Parameters
    ----------
    cases_df : pandas dataframe
        Columns: Age-groups, lines: time with as index the date.

    infectiability_df : pandas dataframe
        Columns: Age-groups, lines: time with as index the date, same size as cases_df

    N_population : array
        The number of people in each age-group

    Returns
    -------
    model

    """

    # This model works on the incidences instead on cases
    new_cases_obs = np.array(cases_df)
    num_age_groups = new_cases_obs.shape[1]

    assert new_cases_obs.ndim == 2
    data_begin = cases_df.index[0]
    data_end = cases_df.index[-1] + datetime.timedelta(days=6)

    # Params for the model
    params = {
        "new_cases_obs": new_cases_obs,
        "data_begin": data_begin,
        "data_end": data_end,
        "fcast_len": 0,
        "diff_data_sim": 14,
        "N_population": N_population,
    }

    pr_median_lambda = 1.0



    with Cov19Model(**params) as this_model:

        infectiability_log = np.log(np.array(infectiability_df))
        assert infectiability_log.ndim == 2

        # prepend 2 weeks
        infectiability_log = np.concatenate(
            [[infectiability_log[0]] * 2, infectiability_log]
        )

        weeks = [
            data_begin - datetime.timedelta(days=14),
            data_begin - datetime.timedelta(days=7),
        ] + list(infectiability_df.index)

        infectiability_log = day_to_week_matrix(
            this_model.sim_begin, this_model.sim_end, weeks
        ).dot(infectiability_log)

        # This assures that the infectiability can be changed in other scenarios
        # But won't change the inference as the normal distribution is very small
        infectiability_log = (
            pm.Normal("infectiability_log_diff", 0, 1, shape=infectiability_log.shape)
        ) * 1e-6 + infectiability_log

        # Get base reproduction number/spreading rate
        R_t_log_base = lambda_t_with_sigmoids(
            change_points_list=get_cps(
                this_model.data_begin,
                this_model.sim_end,
                interval=21,
                pr_median_transient_len=9,
                pr_sigma_transient_len=3,
                pr_sigma_date_transient=4,
            ),
            pr_median_lambda_0=pr_median_lambda,
            name_lambda_t="base_R_t",
        )

        R_t_log_eff = R_t_log_base + infectiability_log

        pm.Deterministic("eff_R_t", tt.exp(R_t_log_eff))

        n_data_points_used = 2
        num_new_E_ref = (
            np.nansum(this_model.new_cases_obs[:n_data_points_used], axis=0)
            / n_data_points_used
            / 7
        )
        diff_E_begin = pm.Normal(
            f"diff_E_begin_log", mu=0, sigma=2, shape=num_age_groups
        )
        sigma_E_beg = pm.HalfNormal(f"sigma_E_begin", 1, shape=num_age_groups)
        diff_E_beg_L2 = pm.Normal(
            f"diff_E_begin_log_L2", mu=0, sigma=1, shape=(11, num_age_groups),
        )
        diff_E_begin_L2_log = diff_E_begin + diff_E_beg_L2 * sigma_E_beg
        new_E_begin = num_new_E_ref * tt.exp(diff_E_begin_L2_log)

        pm.Deterministic("E_begin", new_E_begin)

        # C = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
        # C = np.array([[1.0, 0.1, 0.1], [0.1, 1, 0.1], [0.1, 0.1, 1]])

        rho = N_population / np.sum(N_population)

        #rho_ext = np.concatenate([[rho[0] / 2], rho])
        #rho_ext[1] = rho[0] / 2
        if C_mat_param in ("original", "half-school", "quarter-school"):
            C = load_contact_matrix_mistri(C_mat_param, N_population)
        else:
            C_mat_param = float(C_mat_param) / 100.0
            assert 0 <= C_mat_param <= 1
            C1 = (1 - C_mat_param) * np.identity(num_age_groups )
            C2 = C_mat_param * (rho[:, None] @ np.ones((1, num_age_groups)))
            #normalize_groups1 = np.identity(num_age_groups + 1)[:, 1:]
            #normalize_groups1[0, 0] = 0.5
            #normalize_groups1[1, 0] = 0.5
            #normalize_groups2 = np.identity(num_age_groups + 1)[1:, :]
            #normalize_groups2[0, 0] = 1
            C = C1 + C2

        influx = influx_inci * tt.ones(this_model.sim_shape) * this_model.N_population/1e6

        # Put the lambdas together unknown and known into one tensor (shape: t,v)
        new_E = kernelized_spread_with_interaction(
            R_t_log=R_t_log_eff,
            interaction_matrix=C,
            influx=influx,
            num_groups=this_model.sim_shape[-1],
            pr_new_E_begin=new_E_begin,
            pr_sigma_median_incubation=None,
            name_new_I_t = None,
            name_S_t=None,
            name_new_E_t=None,
        )  # has shape (num_days, num_age_groups)

        # Transform to weekly cases and add a delay of 6 days
        weekly_cases = day_to_week_transform(
            new_E,
            arr_begin=this_model.sim_begin,
            arr_end=this_model.sim_end,
            weeks=cases_df.index,
            end=False,
            additional_delay=6,
        )

        pm.Deterministic("weekly_cases", weekly_cases)

        sigma_obs = pm.HalfCauchy("sigma_obs", beta=1, shape=num_age_groups,)
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
    new_cases_obs = np.squeeze(np.array(cases_df))
    assert new_cases_obs.ndim == 1
    data_begin = cases_df.index[0]
    data_end = cases_df.index[-1] + datetime.timedelta(days=6)

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
            end=False,
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
    data_begin = cases_df.index[0]
    data_end = cases_df.index[-1] + datetime.timedelta(days=6)

    # Params for the model
    params = {
        "new_cases_obs": new_cases_obs,
        "data_begin": data_begin,
        "data_end": data_end,
        "fcast_len": 0,
        "diff_data_sim": 14,
        "N_population": N_population,
    }

    pr_delay = 10

    pr_median_lambda = 1.0

    with Cov19Model(**params) as this_model:

        # infectiability
        infectiability_log = np.log(np.squeeze(np.array(infectiability_df)))
        assert infectiability_log.ndim == 1

        # prepend 2 weeks
        infectiability_log = np.concatenate(
            [[infectiability_log[0]] * 2, infectiability_log]
        )
        weeks = [
            data_begin - datetime.timedelta(days=14),
            data_begin - datetime.timedelta(days=7),
        ] + list(infectiability_df.index)

        infectiability_log = day_to_week_matrix(
            this_model.sim_begin, this_model.sim_end, weeks
        ).dot(infectiability_log)

        # This assures that the infectiability can be changed in other scenarios
        # But won't change the inference as the normal distribution is very small
        infectiability_log = (
            pm.Normal("infectiability_log_diff", 0, 1, shape=infectiability_log.shape)
        ) * 1e-6 + infectiability_log

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
            end=False,
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


