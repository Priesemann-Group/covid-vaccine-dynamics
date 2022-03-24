import logging

log = logging.getLogger(__name__)
import datetime
import sys
import pymc3 as pm
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt


sys.path.append("../covid19_inference")
sys.path.append("..")

from causal_covid.data import load_cases, load_population
from causal_covid.model import (
    create_model_single_dimension,
    create_model_multidmensional,
)


begin = datetime.datetime(2021, 8, 1)
end = datetime.datetime(2021, 9, 15)
file = "../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/observed_infection_data.csv"
file_pop = (
    "../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/population_data.csv"
)

cases_df = load_cases(file, begin, end, num_age_groups=3)
population_df = load_population(file_pop, num_age_groups=3)
incidence_df = pd.DataFrame(
    index=cases_df.index,
    data=np.array(cases_df) / np.array(population_df) * 1e6,
    columns=cases_df.columns,
)


# model = create_model_single_dimension(cases_df, N_population=10**7)
model = create_model_multidmensional(incidence_df, np.squeeze(np.array(population_df)))
trace = pm.sample(model=model, return_inferencedata=True, tune=800, draws=300)


## Plotting

import matplotlib.dates as mdates

def format_date_axis(ax):
    """
    Formats axis with dates
    """
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4, byweekday=mdates.SU))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1, byweekday=mdates.SU))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


colors = ["tab:blue", "tab:orange", "tab:green"]

f, axes = plt.subplots(2, 1, figsize=(5, 4), gridspec_kw=dict(height_ratios=(0.3, 1)))
t = [
    cases_df.index[-1] - datetime.timedelta(days=i)
    for i in range(len(cases_df) * 7, -1, -1)
]
ax = axes[0]
ax.axhline(1, ls="--", color="gray", alpha=0.5)
for a in range(cases_df.shape[1]):
    ax.fill_between(
        t,
        *np.percentile(
            trace.posterior.base_R_t[..., 14:-13, a], axis=(0, 1), q=(12.5, 87.5)
        ),
        alpha=0.3,
        color=colors[a],
        # label="model (75% & 95% CI)"
    )
    ax.fill_between(
        t,
        *np.percentile(
            trace.posterior.base_R_t[..., 14:-13, a], axis=(0, 1), q=(2.5, 97.5)
        ),
        alpha=0.3,
        color=colors[a],
    )
ax.set_xlim(min(t), max(t))
ax.set_ylabel("Effective $R_t$")
format_date_axis(ax)

ax = axes[1]

for a in range(cases_df.shape[1]):
    ax.fill_between(
        cases_df.index,
        *np.percentile(
            trace.posterior.weekly_cases[..., a], axis=(0, 1), q=(12.5, 87.5)
        ),
        alpha=0.3,
        color=colors[a],
        label=f"model (75% & 95% CI), {cases_df.columns[a]}",
    )
    plt.fill_between(
        cases_df.index,
        *np.percentile(
            trace.posterior.weekly_cases[..., a], axis=(0, 1), q=(2.5, 97.5)
        ),
        alpha=0.3,
        color=colors[a],
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
