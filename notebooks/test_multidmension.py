import logging

log = logging.getLogger(__name__)
import datetime
import sys
import pymc3 as pm
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt



sys.path.append("..")

from causal_covid.data import load_cases, load_population
from causal_covid.model import (
    create_model_single_dimension,
    create_model_multidimensional,
)

begin = datetime.datetime(2020, 12, 20)
end = datetime.datetime(2021, 5, 1)
file = "../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/observed_infection_data.csv"
file_pop = (
    "../data/2022-02-09_16-39-19_young_to_old_cap/scenario_export/population_data.csv"
)

cases_df = load_cases(file, begin, end, num_age_groups=9)
population_df = load_population(file_pop, num_age_groups=9)


# model = create_model_single_dimension(cases_df, N_population=10**7)
model = create_model_multidimensional(cases_df, np.squeeze(np.array(population_df)))
trace = pm.sample(model=model, return_inferencedata=True, tune=500, draws=300)

from causal_covid.plotting import format_date_axis
import matplotlib as mpl
import matplotlib.pyplot as plt

age_groups = list(cases_df.columns)
num_age_groups = len(age_groups)

save_dir = "../data/results_inference/"

weekly_cases_list = []


fig = plt.figure(figsize=(8, 2.5 * num_age_groups))
outer_gs = mpl.gridspec.GridSpec(len(age_groups), 1, wspace=0.3, hspace=0.5)

for i, age_group in enumerate(age_groups[:]):

    dict_variable = trace.posterior

    inner = mpl.gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[i], hspace=0.2, wspace=0.3, width_ratios=(1, 1)
    )
    axes = []
    for j in range(2):
        ax = fig.add_subplot(inner[j])
        axes.append(ax)

    t = [
        cases_df.index[0] + datetime.timedelta(days=i) for i in range(len(cases_df) * 7)
    ]

    base_R_t = np.array(dict_variable["base_R_t"][..., 14 : len(t) + 14, i]).reshape(
        (-1, len(t))
    )
    # eff_R_t = np.array(dict_variable["eff_R_t"][..., 14 : len(t) + 14, i]).reshape(
    #    (-1, len(t))
    # )
    weekly_cases = np.array(dict_variable["weekly_cases"][..., i]).reshape(
        (-1, len(cases_df.index))
    )

    ax = axes[0]
    ax.axhline(1, ls="--", color="gray", alpha=0.5)
    ax.fill_between(
        t,
        *np.percentile(base_R_t, axis=(0,), q=(12.5, 87.5)),
        alpha=0.3,
        color="tab:blue",
        label="base $R_t$ (75% & 95% CI)",
    )
    ax.fill_between(
        t,
        *np.percentile(base_R_t, axis=(0,), q=(2.5, 97.5)),
        alpha=0.3,
        color="tab:blue",
    )
    # ax.fill_between(
    #    t,
    #    *np.percentile(eff_R_t, axis=(0,), q=(12.5, 87.5)),
    #    alpha=0.3,
    #    color="tab:orange",
    #    label="eff. $R_t$ (75% & 95% CI)"
    # )
    # ax.fill_between(
    #    t,
    #    *np.percentile(eff_R_t, axis=(0,), q=(2.5, 97.5)),
    #    alpha=0.3,
    #    color="tab:orange"
    # )

    ax.set_ylim(0.5, 4)
    ax.set_xlim(min(t), max(t))
    ax.set_ylabel("$R_t$")
    ax.set_xlabel("2021")
    ax.legend()
    format_date_axis(ax)

    ax = axes[1]
    ax.fill_between(
        cases_df.index,
        *np.percentile(
            weekly_cases / model.N_population[i] * 1e6, axis=(0,), q=(12.5, 87.5)
        ),
        alpha=0.3,
        color="tab:blue",
        label="model (75% & 95% CI)",
    )
    ax.fill_between(
        cases_df.index,
        *np.percentile(
            weekly_cases / model.N_population[i] * 1e6, axis=(0,), q=(2.5, 97.5)
        ),
        alpha=0.3,
        color="tab:blue",
    )
    ax.plot(
        cases_df.index,
        np.array(cases_df)[..., i] / model.N_population[i] * 1e6,
        "d",
        color="k",
        label="data",
    )
    ax.set_xlabel("2021")
    ax.set_ylabel("Weekly incidence\n(cases per million)")
    ax.set_xlim(min(t), max(t))
    ax.set_ylim(0, 10_000)
    format_date_axis(ax)
    ax.legend()
plt.show()

save_path = os.path.join(save_dir, f"{save_name}_{begin_str}--{end_str}")
plt.savefig(save_path + ".pdf", bbox_inches="tight")


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


import theano.tensor as tt
import theano
import covid19_inference.model.utility as ut
from causal_covid.utils import day_to_week_transform

num_groups = num_age_groups

N = model.N_population  # shape: [num_groups]

assert len(N) == num_groups
assert model.sim_shape[-1] == num_groups
assert len(model.sim_shape) == 2

# new_E_begin = tt.as_tensor_variable(np.median(trace.posterior.I_begin, axis=(0,1)))/7
# R_t = tt.as_tensor_variable(np.median(trace.posterior.base_R_t, axis=(0,1)))
new_E_begin = tt.as_tensor_variable(trace.posterior.I_begin[-1, -1]) / 7
R_t = tt.as_tensor_variable(trace.posterior.base_R_t[-1, -1])

interaction_matrix = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])  # , dtype='float32')

# shape: num_groups
S_begin = N - pm.math.sum(new_E_begin, axis=(0,))

# shape: num_groups, [independent dimension]
new_I_0 = tt.zeros(model.sim_shape[1:])


median_incubation = 4
sigma_incubation = 0.4

# Choose transition rates (E to I) according to incubation period distribution

x = np.arange(1, 11)[:, None]

beta = ut.tt_lognormal(x, tt.log(median_incubation), sigma_incubation)

# Runs kernelized spread model:
def next_day(
    R_t,
    S_t,
    nE1,
    nE2,
    nE3,
    nE4,
    nE5,
    nE6,
    nE7,
    nE8,
    nE9,
    nE10,
    _,
    beta,
    N,
    interaction_matrix,
):
    new_I_t = (
        beta[0] * nE1
        + beta[1] * nE2
        + beta[2] * nE3
        + beta[3] * nE4
        + beta[4] * nE5
        + beta[5] * nE6
        + beta[6] * nE7
        + beta[7] * nE8
        + beta[8] * nE9
        + beta[9] * nE10
    )

    # The reproduction number is assumed to have a symmetric effect, hence the sqrt
    new_E_t = tt.sqrt(R_t) / N * new_I_t * S_t

    # Interaction between gender groups (groups,groups)@(groups, [evtl. other dimension])
    new_E_t = tt.sqrt(R_t) * tt.dot(interaction_matrix, new_E_t)

    # Update suceptible compartement
    S_t = S_t - new_E_t.sum(axis=0)
    S_t = tt.clip(S_t, -1, N)
    return S_t, new_E_t, new_I_t


# theano scan returns two tuples, first one containing a time series of
# what we give in outputs_info : S, E's, new_I
outputs, _ = theano.scan(
    fn=next_day,
    sequences=[R_t],
    outputs_info=[
        S_begin,  # shape: countries
        dict(
            initial=new_E_begin, taps=[-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]
        ),  # shape time, groups, independent dimension
        new_I_0,  # shape groups, independent dimension
    ],
    non_sequences=[beta, N, interaction_matrix],
)
S_t, new_E_t, new_I_t = outputs
# new_E_t = new_E_t.eval()


fig = plt.figure(figsize=(8, 2.5 * num_age_groups))
outer_gs = mpl.gridspec.GridSpec(len(age_groups), 1, wspace=0.3, hspace=0.5)

for i, age_group in enumerate(age_groups[:]):

    dict_variable = trace.posterior

    inner = mpl.gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[i], hspace=0.2, wspace=0.3, width_ratios=(1, 1)
    )
    axes = []
    for j in range(2):
        ax = fig.add_subplot(inner[j])
        axes.append(ax)

    t = [
        cases_df.index[0] + datetime.timedelta(days=i) for i in range(len(cases_df) * 7)
    ]

    base_R_t = np.array(dict_variable["base_R_t"][..., 14 : len(t) + 14, i]).reshape(
        (-1, len(t))
    )
    # eff_R_t = np.array(dict_variable["eff_R_t"][..., 14 : len(t) + 14, i]).reshape(
    #    (-1, len(t))
    # )
    # daily_cases = new_E_t[14 : len(t) + 14, i]
    weekly_cases = day_to_week_transform(
        new_E_t,
        arr_begin=model.sim_begin,
        arr_end=model.sim_end,
        weeks=cases_df.index,
        end=False,
        additional_delay=6,
    ).eval()[..., i]

    ax = axes[0]
    ax.axhline(1, ls="--", color="gray", alpha=0.5)
    ax.fill_between(
        t,
        *np.percentile(base_R_t, axis=(0,), q=(12.5, 87.5)),
        alpha=0.3,
        color="tab:blue",
        label="base $R_t$ (75% & 95% CI)",
    )
    ax.fill_between(
        t,
        *np.percentile(base_R_t, axis=(0,), q=(2.5, 97.5)),
        alpha=0.3,
        color="tab:blue",
    )
    # ax.fill_between(
    #    t,
    #    *np.percentile(eff_R_t, axis=(0,), q=(12.5, 87.5)),
    #    alpha=0.3,
    #    color="tab:orange",
    #    label="eff. $R_t$ (75% & 95% CI)"
    # )
    # ax.fill_between(
    #    t,
    #    *np.percentile(eff_R_t, axis=(0,), q=(2.5, 97.5)),
    #    alpha=0.3,
    #    color="tab:orange"
    # )

    ax.set_ylim(0.5, 4)
    ax.set_xlim(min(t), max(t))
    ax.set_ylabel("$R_t$")
    ax.set_xlabel("2021")
    ax.legend()
    format_date_axis(ax)

    ax = axes[1]

    # plt.plot(t, daily_cases/model.N_population[i]*1e6*7, color="tab:blue")
    plt.plot(
        cases_df.index, weekly_cases / model.N_population[i] * 1e6, color="tab:blue"
    )

    ax.plot(
        cases_df.index,
        np.array(cases_df)[..., i] / model.N_population[i] * 1e6,
        "d",
        color="k",
        label="data",
    )
    ax.set_xlabel("2021")
    ax.set_ylabel("Weekly incidence\n(cases per million)")
    ax.set_xlim(min(t), max(t))
    ax.set_ylim(0, 10_000)
    format_date_axis(ax)
    ax.legend()
plt.show()
