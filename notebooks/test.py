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
from causal_covid.data import load_cases
from causal_covid.utils import get_cps, day_to_week_matrix
from covid19_inference import Cov19Model
from covid19_inference.model import (
    lambda_t_with_sigmoids,
    kernelized_spread_gender,
    week_modulation,
    student_t_likelihood,
    delay_cases,
    uncorrelated_prior_E,
)

begin = datetime.datetime(2021,8,1)
end = datetime.datetime(2021,10,1)
file = "/home/jdehning/repositories/causal_covid/data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/observed_infection_data.csv"
cases_df = load_cases(file, begin, end)



