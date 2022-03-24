import pickle
import os
import sys

sys.path.append("../covid19_inference")
sys.path.append("..")

import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import params
from causal_covid.data import load_cases, load_infectiability
from causal_covid.utils import day_to_week_matrix
from causal_covid.plotting import plot_R_and_cases


scenario_U2_file = (
    "./../data/2022-02-09_16-39-19_young_to_old_cap/vaccination_policy/U_2.npy"
)
scenario_U3_file = (
    "./../data/2022-02-09_16-39-19_young_to_old_cap/vaccination_policy/u_3.npy"
)
save_dir = "../data/results_scenarios/"
save_name = "scenario_young_to_old_cap"

begin_str = "2021-07-01"
end_str = "2021-12-01"
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


weekly_cases_list = []

fig = plt.figure(figsize=(8, 30))
outer_gs = mpl.gridspec.GridSpec(len(age_groups), 2, wspace=0.3, hspace=0.5)


for i, age_group in enumerate(age_groups):

    # Load trace and model of the age-group:

    file_path = os.path.join(
        params.traces_dir,
        f"run-age_group={age_group}" f"-begin={begin_str}" f"-end={end_str}.pkl",
    )
    with open(file_path, "rb") as f:
        loaded_stuff = pickle.load(f)
    cases_df, infectiability_df, model, trace = loaded_stuff

    # Load the vaccination file of the scenario and calculat the changed infectability:

    infectiability_scenario_df = load_infectiability(
        params.vaccination_file,
        params.population_file,
        scenario_U2_file,
        scenario_U3_file,
        params.waning_file,
        infectiability_df.index[0],
        infectiability_df.index[-1],
        num_age_groups=9,
    )
    infectiability_scenario_df = infectiability_scenario_df[age_group]

    # Change the infactability in the loaded trace to afterwards the calculate the new dynamics:

    infectability_original = np.log(np.squeeze(np.array(infectiability_df)))
    infectability_scenario = np.log(np.squeeze(np.array(infectiability_scenario_df)))
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

    axes_original = plot_R_and_cases(fig, outer_gs[i, 0], cases_df, trace.posterior)
    axes_scenario = plot_R_and_cases(fig, outer_gs[i, 1], cases_df, predictive)
    axes_original[0].set_title(f"Age-group {age_group}, Observed")
    axes_scenario[0].set_title(f"Age-group {age_group}, Scenario")

save_path = os.path.join(save_dir, f"{save_name}_{begin_str}--{end_str}")

plt.savefig(save_path + ".pdf", bbox_inches="tight")

to_save = pd.DataFrame(
    index=cases_df.index, data=np.array(weekly_cases_list).T, columns=age_groups,
)

to_save.to_csv(save_path + ".csv")
