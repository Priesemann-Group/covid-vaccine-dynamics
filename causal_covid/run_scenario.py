import pickle
import os
import sys

import numpy as np
import pymc3 as pm
import pandas as pd


import params
from causal_covid.data import load_infectiability
from causal_covid.utils import day_to_week_matrix
from causal_covid.plotting import plot_R_and_cases



def single_dimensional(
    U2,
    u3,
    begin_str="2021-07-01",
    end_str="2021-12-01",
    plotting=False,
    save_dir=None,
    save_name="scenario",
):
    age_groups = [
        "0-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70-79",
        "80-89",
        "90+",
    ]

    if save_dir is None:
        save_dir=os.path.join(os.path.dirname(__file__), "../data/results_scenarios/")

    weekly_cases_list = []

    if plotting:
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(8, 30))
        outer_gs = mpl.gridspec.GridSpec(len(age_groups), 2, wspace=0.3, hspace=0.5)

    for i, age_group in enumerate(age_groups):

        # Load trace and model of the age-group:
        file_path = os.path.join(
            params.traces_dir,
            f"run-age_group={age_group}-begin={begin_str}-end={end_str}.pkl",
        )
        with open(file_path, "rb") as f:
            loaded_stuff = pickle.load(f)
        cases_df, infectiability_df, model, trace = loaded_stuff

        # Load the vaccination file of the scenario and calculat the changed infectability:
        infectiability_scenario_df = load_infectiability(
            params.vaccination_file,
            params.population_file,
            U2,
            u3,
            params.waning_file,
            infectiability_df.index[0],
            infectiability_df.index[-1],
            num_age_groups=9,
        )
        infectiability_scenario_df = infectiability_scenario_df[age_group]

        # Change the infactability in the loaded trace to afterwards the calculate the new dynamics:

        infectability_original = np.log(np.squeeze(np.array(infectiability_df)))
        infectability_scenario = np.log(
            np.squeeze(np.array(infectiability_scenario_df))
        )
        trace_for_scenario = trace.copy()
        # This requires some calculations, as the infectability was originally
        # modelled as a distribution with a very small standard deviation
        infectiability_diff_new = day_to_week_matrix(
            model.sim_begin, model.sim_end, infectiability_df.index, end=True
        ).dot((infectability_scenario - infectability_original) * 1e6)
        shape_to_have = trace_for_scenario.posterior["infectiability_log_diff"].shape
        trace_for_scenario.posterior["infectiability_log_diff"].values = (
            np.ones(shape_to_have) * infectiability_diff_new
        )

        # Sample the new dynamics :

        # (pm.fast_sample_posterior_predictive can be replaced by
        # pm.sample_posterior_predictive if some problem should occur)
        predictive = pm.fast_sample_posterior_predictive(
            trace=trace_for_scenario,
            model=model,
            var_names=["weekly_cases", "base_R_t", "eff_R_t"],
        )
        weekly_cases_list.append(np.median(predictive["weekly_cases"], axis=0))

        # Plotting:
        if plotting:
            axes_original = plot_R_and_cases(fig, outer_gs[i, 0], cases_df, trace.posterior)
            axes_scenario = plot_R_and_cases(fig, outer_gs[i, 1], cases_df, predictive)
            axes_original[0].set_title(f"Age-group {age_group}, Observed")
            axes_scenario[0].set_title(f"Age-group {age_group}, Scenario")
    if plotting:
        save_path = os.path.join(save_dir, f"{save_name}_{begin_str}--{end_str}")
        plt.savefig(save_path + ".pdf", bbox_inches="tight")

    median_cases = pd.DataFrame(
        index=cases_df.index, data=np.array(weekly_cases_list).T, columns=age_groups,
    )

    return median_cases
