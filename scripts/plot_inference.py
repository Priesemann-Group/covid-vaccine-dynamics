import pickle
import os
import sys
import datetime


sys.path.append("..")

import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from causal_covid import params
from causal_covid.plotting import  format_date_axis



begin_str = "2020-12-20"
end_str = "2021-12-19"
draws=500
save_name="inference"

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

save_dir =  "../data/results_inference/"

weekly_cases_list = []


fig = plt.figure(figsize=(8, 20))
outer_gs = mpl.gridspec.GridSpec(len(age_groups), 1, wspace=0.3, hspace=0.5)

for i, age_group in enumerate(age_groups[:]):

    # Load trace and model of the age-group:
    file_path = os.path.join(
        params.traces_dir,
        f"run-age_group={age_group}-begin={begin_str}-end={end_str}-draws={draws}.pkl",
    )
    with open(file_path, "rb") as f:
        loaded_stuff = pickle.load(f)
    cases_df, infectiability_df, model, trace = loaded_stuff


    dict_variable = trace.posterior

    inner = mpl.gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[i], hspace=0.2, wspace=0.3, width_ratios=(1, 1)
    )
    axes = []
    for j in range(2):
        ax = fig.add_subplot(inner[j])
        axes.append(ax)

    t = [
        cases_df.index[0] + datetime.timedelta(days=i)
        for i in range(len(cases_df) * 7)
    ]

    base_R_t = np.array(dict_variable["base_R_t"][..., 14 : len(t) + 14]).reshape(
        (-1, len(t))
    )
    eff_R_t = np.array(dict_variable["eff_R_t"][..., 14 : len(t) + 14]).reshape(
        (-1, len(t))
    )
    weekly_cases = np.array(dict_variable["weekly_cases"]).reshape(
        (-1, len(cases_df.index))
    )

    ax = axes[0]
    ax.axhline(1, ls="--", color="gray", alpha=0.5)
    ax.fill_between(
        t,
        *np.percentile(base_R_t, axis=(0,), q=(12.5, 87.5)),
        alpha=0.3,
        color="tab:blue",
        label="base $R_t$ (75% & 95% CI)"
    )
    ax.fill_between(
        t,
        *np.percentile(base_R_t, axis=(0,), q=(2.5, 97.5)),
        alpha=0.3,
        color="tab:blue"
    )
    ax.fill_between(
        t,
        *np.percentile(eff_R_t, axis=(0,), q=(12.5, 87.5)),
        alpha=0.3,
        color="tab:orange",
        label="eff. $R_t$ (75% & 95% CI)"
    )
    ax.fill_between(
        t,
        *np.percentile(eff_R_t, axis=(0,), q=(2.5, 97.5)),
        alpha=0.3,
        color="tab:orange"
    )

    ax.set_ylim(0.5, 4)
    ax.set_xlim(min(t), max(t))
    ax.set_ylabel("$R_t$")
    ax.set_xlabel("2021")
    ax.legend()
    format_date_axis(ax)

    ax = axes[1]
    ax.fill_between(
        cases_df.index,
        *np.percentile(weekly_cases/model.N_population*1e6, axis=(0,), q=(12.5, 87.5)),
        alpha=0.3,
        color="tab:blue",
        label="model (75% & 95% CI)"
    )
    plt.fill_between(
        cases_df.index,
        *np.percentile(weekly_cases/model.N_population*1e6, axis=(0,), q=(2.5, 97.5)),
        alpha=0.3,
        color="tab:blue"
    )
    ax.plot(cases_df.index, np.array(cases_df)/model.N_population*1e6, "d", color="k", label="data")
    ax.set_xlabel("2021")
    ax.set_ylabel("Weekly incidence\n(cases per million)")
    ax.set_xlim(min(t), max(t))
    ax.set_ylim(0, 10_000)
    format_date_axis(ax)
    ax.legend()

save_path = os.path.join(save_dir, f"{save_name}_{begin_str}--{end_str}")
plt.savefig(save_path + ".pdf", bbox_inches="tight")


save_dir =  "../data/results_inference/"

weekly_cases_list = []


fig, ax = plt.subplots(figsize=(5, 3))

for i, age_group in enumerate(age_groups[:]):

    # Load trace and model of the age-group:
    file_path = os.path.join(
        params.traces_dir,
        f"run-age_group={age_group}-begin={begin_str}-end={end_str}-draws={draws}.pkl",
    )
    with open(file_path, "rb") as f:
        loaded_stuff = pickle.load(f)
    cases_df, infectiability_df, model, trace = loaded_stuff


    dict_variable = trace.posterior

    t = [
        cases_df.index[0] + datetime.timedelta(days=i)
        for i in range(len(cases_df) * 7)
    ]

    eff_R_t = np.array(dict_variable["eff_R_t"][..., 14 : len(t) + 14]).reshape(
        (-1, len(t))
    )


    ax.axhline(1, ls="--", color="gray", alpha=0.5)
    ax.plot(
        t,
        np.median(eff_R_t, axis=(0,)),
        label=f"Age group {age_group}"
    )

ax.set_ylim(0.5, 2)
ax.set_xlim(min(t), max(t))
ax.set_ylabel("Effectiv $R_t$")
ax.legend()
format_date_axis(ax)
ax.set_xlabel("2021")

ax.legend()

save_path = os.path.join(save_dir, f"{save_name}_{begin_str}--{end_str}_eff-R")
plt.savefig(save_path + ".pdf", bbox_inches="tight")