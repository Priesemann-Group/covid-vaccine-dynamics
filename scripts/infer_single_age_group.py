import logging

log = logging.getLogger(__name__)

import datetime
import sys
import pickle
import argparse
import os


import pymc3 as pm

sys.path.append("../covid19_inference")
sys.path.append("..")

from causal_covid.data import load_cases, load_infectiability, load_population
from causal_covid.model import create_model_single_dimension_infectiability
from causal_covid import params


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
    "-a", "--age_group", type=str, help="Age-group to use", default="",
)

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

if __name__ == "__main__":
    args = parser.parse_args()

    log.info(f"Script started: {datetime.datetime.now()}")
    log.info(f"Args: {args.__dict__}")

    begin = str2datetime(args.begin)
    end = str2datetime(args.end)

    cases_df = load_cases(params.cases_file, begin, end, num_age_groups=9)
    cases_df = cases_df[args.age_group]

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
        num_age_groups=9,
    )
    infectiability_df = infectiability_df[args.age_group]

    population = load_population(params.population_file, num_age_groups=9)
    population = float(population[args.age_group])

    model = create_model_single_dimension_infectiability(
        cases_df, infectiability_df, N_population=population
    )
    trace = pm.sample(
        model=model, draws=500, tune=500, return_inferencedata=True, cores=2, chains=2
    )

    input_args_dict = dict(**args.__dict__)
    save_name = f"run{dict_2_string(input_args_dict)}"

    file_path = os.path.join(params.traces_dir, save_name)

    with open(file_path + ".pkl", "wb") as f:
        pickle.dump([cases_df, infectiability_df, model, trace], f)
