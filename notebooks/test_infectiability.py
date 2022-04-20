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
import matplotlib.pyplot as plt



sys.path.append("..")

import data
from causal_covid.data import load_cases, load_infectiability
from causal_covid.utils import get_cps, day_to_week_matrix
from causal_covid.model import (
    create_model_single_dimension,
    create_model_multidimensional,
    create_model_single_dimension_infectiability,
)

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

begin = datetime.datetime(2021, 8, 1)
end = datetime.datetime(2021, 10, 1)
file = "../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/observed_infection_data.csv"
cases_df = load_cases(file, begin, end, num_age_groups=1)

vaccination_file = "./../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/observed_vaccination_data.csv"
waning_file = "./../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/vaccine_efficacy_waning_data.csv"
population_file = "./../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/population_data.csv"
observed_U2_file = "./../data/2022-02-02_14-19-24_observed_vac_policy/vaccination_policy/U_2.npy"
observed_U3_file = "./../data/2022-02-02_14-19-24_observed_vac_policy/vaccination_policy/u_3.npy"

diff_data_sim = 14+7 # plus 7 days because data begin 6 days earlier as the reported index at the end of the week
begin_infectiability = begin - datetime.timedelta(days=diff_data_sim)

infectiability_df = load_infectiability(vaccination_file, population_file, observed_U2_file, observed_U3_file, waning_file, begin_infectiability, end, num_age_groups=1)

model = create_model_single_dimension_infectiability(cases_df, infectiability_df, N_population=10 ** 7)
# model2 = create_model_multidimensional(cases_df, [10**8, 10**8, 10**8])
trace = pm.sample(model=model, return_inferencedata=True)

vaccination_file = "./../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/observed_vaccination_data.csv"
load_infectiability(vaccination_file, population_file, observed_U2_file, observed_U3_file, waning_file, begin_infectiability, end, num_age_groups=1)


## Plotting

import matplotlib.dates as mdates


def format_date_axis(ax):
    """
    Formats axis with dates
    """
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4, byweekday=mdates.SU))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1, byweekday=mdates.SU))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


f, axes = plt.subplots(2, 1, figsize=(5, 4), gridspec_kw=dict(height_ratios=(0.3, 1)))
t = [
    cases_df.index[-1] - datetime.timedelta(days=i)
    for i in range(len(cases_df) * 7, -1, -1)
]
ax = axes[0]
ax.axhline(1, ls="--", color="gray", alpha=0.5)
ax.fill_between(
    t,
    *np.percentile(trace.posterior.base_R_t[..., 14:-13], axis=(0, 1), q=(12.5, 87.5)),
    alpha=0.3,
    color="tab:blue",
    label="model (75% & 95% CI)"
)
ax.fill_between(
    t,
    *np.percentile(trace.posterior.base_R_t[..., 14:-13], axis=(0, 1), q=(2.5, 97.5)),
    alpha=0.3,
    color="tab:blue"
)
ax.set_xlim(min(t), max(t))
ax.set_ylabel("Effective $R_t$")
format_date_axis(ax)

ax = axes[1]

ax.fill_between(
    cases_df.index,
    *np.percentile(trace.posterior.weekly_cases, axis=(0, 1), q=(12.5, 87.5)),
    alpha=0.3,
    color="tab:blue",
    label="model (75% & 95% CI)"
)
plt.fill_between(
    cases_df.index,
    *np.percentile(trace.posterior.weekly_cases, axis=(0, 1), q=(2.5, 97.5)),
    alpha=0.3,
    color="tab:blue"
)
plt.plot(cases_df.index, np.array(cases_df), "d", color="k", label="data")
plt.xlabel("2021")
plt.ylabel("Weekly cases")
ax.set_xlim(min(t), max(t))
ax.set_ylim(0)
format_date_axis(ax)
plt.legend()
plt.tight_layout()

plt.show()
