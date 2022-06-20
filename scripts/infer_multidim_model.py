import logging

log = logging.getLogger(__name__)

import datetime
import sys
import pickle
import argparse
import os


import numpy as np


sys.path.append("..")

from causal_covid.data import load_cases, load_infectiability, load_population
from causal_covid.model import create_model_multidimensional
from causal_covid import params
import covid19_inference as cov19


def str2datetime(string):
    return datetime.datetime.strptime(string, "%Y-%m-%d")


def dict_2_string(dictionary):
    """
    Creates a string from a dictionary
    """

    f_str = ""
    for arg in dictionary:
        if arg in ["log", "dir"]:
            continue
        f_str += f"-{arg}={args.__dict__[arg]}"
    return f_str


parser = argparse.ArgumentParser(description="Run model")

parser.add_argument(
    "-b",
    "--begin",
    type=str,
    help="Use which Sunday as first data point (YYYY-MM-DD)",
    default="",
)

parser.add_argument(
    "-e",
    "--end",
    type=str,
    help="Use which Sunday as last data point (YYYY-MM-DD)",
    default="",
)

parser.add_argument(
    "-C", "--C_mat", type=str, help="Parameter of the contact-matrix", default="",
)

parser.add_argument(
    "-v", "--V1_eff", type=str, help="Efficacy of 1st shot (in percent)", default="",
)

parser.add_argument(
    "-w",
    "--V2_eff",
    type=str,
    help="Efficacy of the 2nd shot (in percent)",
    default="",
)

parser.add_argument(
    "-z",
    "--V3_eff",
    type=str,
    help="Efficacy of the 3rd shot (in percent)",
    default="",
)

parser.add_argument(
    "-i", "--influx", type=str, help="Influx per day in cases per million", default="",
)


parser.add_argument(
    "-d", "--draws", type=int, help="Number of draws", default="",
)

if __name__ == "__main__":
    args = parser.parse_args()

    log.info(f"Script started: {datetime.datetime.now()}")
    log.info(f"Args: {args.__dict__}")

    begin = str2datetime(args.begin)
    end = str2datetime(args.end)

    cases_df = load_cases(params.cases_file, begin, end, num_age_groups=9)
    cases_df = cases_df

    diff_data_sim = 14
    begin_infectiability = begin

    infectiability_df = load_infectiability(
        params.vaccination_file,
        params.population_file,
        params.observed_U2_file,
        params.observed_U3_file,
        params.waning_file,
        begin_infectiability,
        end,
        V1_eff=float(args.V1_eff),
        V2_eff=float(args.V2_eff),
        V3_eff=float(args.V3_eff),
        num_age_groups=9,
    )
    infectiability_df = infectiability_df

    population_df = load_population(params.population_file, num_age_groups=9)
    population = np.squeeze(np.array(population_df))


    model = create_model_multidimensional(
        cases_df,
        infectiability_df,
        N_population=population,
        C_mat_param=args.C_mat,
        influx_inci=float(args.influx),
    )

    draws = tune = args.draws

    multitrace, trace = cov19.robust_sample(
        model,
        tune=tune,
        draws=draws,
        burnin_draws=draws // 6,
        burnin_draws_2nd=draws // 3,
        burnin_chains=16,
        burnin_chains_2nd=8,
        final_chains=2,
        sample_kwargs={"cores": 16},
        max_treedepth=10,
        target_accept=0.85,
    )

    input_args_dict = dict(**args.__dict__)
    save_name = f"run{dict_2_string(input_args_dict)}"

    file_path = os.path.join(params.traces_dir, save_name)

    with open(file_path + ".pkl", "wb") as f:
        pickle.dump([cases_df, infectiability_df, model, trace], f)
