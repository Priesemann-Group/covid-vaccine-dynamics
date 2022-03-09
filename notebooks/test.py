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


sys.path.append("../covid19_inference")
sys.path.append("..")

import data
from causal_covid.data import load_cases
from causal_covid.utils import get_cps, day_to_week_matrix
from causal_covid.model import create_model_single_dimension, create_model_multidmensional

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
file = "/home/jdehning/repositories/causal_covid/data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/observed_infection_data.csv"
cases_df = load_cases(file, begin, end)
cases_df = cases_df.sum(axis=1)

model = create_model_single_dimension(cases_df, N_population=10**7)
#model2 = create_model_multidmensional(cases_df, [10**8, 10**8, 10**8])
trace = pm.sample(model=model, return_inferencedata=True)

plt.fill_between(np.arange(9), *np.percentile(trace.posterior.weekly_cases, axis=(0,1),q=(25,75)), alpha=0.3, color="tab:blue", label="model")
plt.fill_between(np.arange(9), *np.percentile(trace.posterior.weekly_cases, axis=(0,1),q=(2.5,97.5)), alpha=0.3, color="tab:blue")
plt.plot(np.sum(np.array(cases_df), axis=1), "d", label='data')
plt.xlabel("week number")
plt.ylabel("incidence")
plt.legend()

plt.show()