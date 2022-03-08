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
from utils import get_cps, day_to_week_matrix
from covid19_inference import Cov19Model
from covid19_inference.model import (
    lambda_t_with_sigmoids,
    kernelized_spread_gender,
    week_modulation,
    student_t_likelihood,
    delay_cases,
    uncorrelated_prior_E,
)




import covid19_inference

def create_model(
    cases_df=None,
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

    #new_cases_obs= data.load_cases(file = "/home/jdehning/repositories/causal_covid/data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/observed_infection_data.csv",
                                  # )
    new_cases_obs = np.array(cases_df)
    num_age_groups = new_cases_obs.shape[0]
    data_begin=cases_df.index[0]

    # Params for the model
    params = {
        "new_cases_obs": new_cases_obs,
        "data_begin": data_begin,
        "fcast_len": 42,
        "diff_data_sim": 16,
        "N_population": [10_0000, 10_000, 10_0000],  # population chile
    }



    pr_delay = 10

    pr_median_lambda = 1.0

    with Cov19Model(**params) as this_model:

        # Get base reproduction number/spreading rate
        lambda_t_log = lambda_t_with_sigmoids(
            change_points_list=get_cps(
                this_model.data_begin,
                this_model.sim_end,
                interval=14,
                pr_median_transient_len=6,
                pr_sigma_transient_len=2,
            ),
            pr_median_lambda_0=pr_median_lambda,
            name_lambda_t="base_lambda_t",
        )

        week_mapping = day_to_week_matrix(
            this_model.sim_begin, this_model.sim_end, cases_df.index,
        )

        E_begin = uncorrelated_prior_E()

        C = np.array([[1,0.1,0.1], [0.1, 1, 0.1],[0.1,0.1,1]])

        # Put the lambdas together unknown and known into one tensor (shape: t,v)
        new_cases = kernelized_spread_gender(
            lambda_t_log=lambda_t_log,
            gender_interaction_matrix = C,
            pr_new_E_begin=E_begin,
        )

        # Delay the cases by a lognormal reporting delay and add them as a trace variable
        new_cases = delay_cases(
            cases=new_cases,
            name_cases="delayed_cases",
            pr_mean_of_median=pr_delay,
            pr_median_of_width=0.3,
            seperate_on_axes=False,
            num_seperated_axes=num_age_groups,
        )

        """

        # Modulate the inferred cases by a abs(sin(x)) function, to account for weekend effects
        # Also adds the "new_cases" variable to the trace that has all model features.
        new_cases_modulated = week_modulation(cases=new_cases, name_cases="new_cases")
        """
        # Define the likelihood, uses the new_cases_obs set as model parameter
        student_t_likelihood(cases=week_mapping @ new_cases)

        return this_model