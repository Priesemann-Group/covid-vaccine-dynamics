import logging
import os
import pickle

log = logging.getLogger(__name__)
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import scipy.special
import scipy.ndimage

from causal_covid import params

# Matplotlib config
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["font.family"] = "sans-serif"
# mpl.rcParams['axes.linewidth'] = 0.3
mpl.rcParams["axes.labelcolor"] = "black"
mpl.rcParams["axes.edgecolor"] = "black"
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["axes.labelsize"] = 8
mpl.rcParams["axes.titlesize"] = 8
mpl.rcParams["legend.fontsize"] = 6
mpl.rcParams["figure.dpi"] = 200  # this primarily affects the size on screen

# Save figure as pdf and png
save_kwargs = {"transparent": True, "dpi": 300, "bbox_inches": "tight"}


def format_date_axis(ax):
    """
    Formats axis with dates
    """
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=8, byweekday=mdates.SU))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1, byweekday=mdates.SU))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))


def plot_R_and_cases(fig, outer_gs, cases_df, dict_variable):

    inner = mpl.gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs, hspace=0.2, height_ratios=(0.6, 1)
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

    ax.set_xlim(min(t), max(t))
    ax.set_ylabel("$R_t$")
    ax.legend()
    format_date_axis(ax)
    ax.xaxis.set_ticklabels([])

    ax = axes[1]
    ax.fill_between(
        cases_df.index,
        *np.percentile(weekly_cases, axis=(0,), q=(12.5, 87.5)),
        alpha=0.3,
        color="tab:blue",
        label="model (75% & 95% CI)"
    )
    plt.fill_between(
        cases_df.index,
        *np.percentile(weekly_cases, axis=(0,), q=(2.5, 97.5)),
        alpha=0.3,
        color="tab:blue"
    )
    ax.plot(cases_df.index, np.array(cases_df), "d", color="k", label="data")
    ax.set_xlabel("2021")
    ax.set_ylabel("Weekly cases")
    ax.set_xlim(min(t), max(t))
    ax.set_ylim(0)
    format_date_axis(ax)
    ax.legend()

    return axes


def plot_R_and_cases_multidim(i_age, fig, outer_gs, cases_df, dict_variable, population):

    inner = mpl.gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer_gs, hspace=0.2, height_ratios=(0.6, 1)
    )
    axes = []
    for j in range(2):
        ax = fig.add_subplot(inner[j])
        axes.append(ax)

    t = [
        cases_df.index[0] + datetime.timedelta(days=i)
        for i in range(len(cases_df) * 7)
    ]

    base_R_t = np.array(dict_variable["base_R_t"][..., 14 : len(t) + 14, i_age]).reshape(
        (-1, len(t))
    )
    eff_R_t = np.array(dict_variable["eff_R_t"][..., 14 : len(t) + 14, i_age]).reshape(
        (-1, len(t))
    )
    weekly_cases = np.array(dict_variable["weekly_cases"])[..., i_age].reshape(
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

    ax.set_xlim(min(t), max(t))
    ax.set_ylabel("$R_t$")
    ax.legend()
    format_date_axis(ax)
    ax.xaxis.set_ticklabels([])

    ax = axes[1]
    ax.fill_between(
        cases_df.index,
        *np.percentile(weekly_cases/population[i_age]*1e6, axis=(0,), q=(12.5, 87.5)),
        alpha=0.3,
        color="tab:blue",
        label="model (75% & 95% CI)"
    )
    plt.fill_between(
        cases_df.index,
        *np.percentile(weekly_cases/population[i_age]*1e6, axis=(0,), q=(2.5, 97.5)),
        alpha=0.3,
        color="tab:blue"
    )
    ax.plot(cases_df.index, np.array(cases_df)[...,i_age]/population[i_age]*1e6, "d", color="k", label="data")
    ax.set_xlabel("2021")
    ax.set_ylabel("Weekly incidence")
    ax.set_xlim(min(t), max(t))
    ax.set_ylim(0)
    format_date_axis(ax)
    ax.legend()

    return axes


def plot_inference(
    begin_str="2020-12-20",
    end_str="2021-12-19",
    C_mat_param=71,
    V1_eff=50,
    V2_eff=70,
    V3_eff=95,
    influx=1,
    draws=500,
    save_dir=None):

    file_path = os.path.join(
            params.traces_dir,
            f"run-begin={begin_str}-end={end_str}-C_mat={C_mat_param}-"
            f"V1_eff={V1_eff}-V2_eff={V2_eff}-V3_eff={V3_eff}-influx={influx}-draws={draws}.pkl",
        )
    with open(file_path, "rb") as f:
            loaded_stuff = pickle.load(f)
    cases_df, infectiability_df, model, trace = loaded_stuff
    t = [
            cases_df.index[0] + datetime.timedelta(days=i)
            for i in range(len(cases_df) * 7)
        ]
    cmap = mpl.cm.get_cmap("viridis")
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
    f, axes = plt.subplots(2, 2, figsize=(8,4), gridspec_kw = dict(height_ratios=(0.3,1)))
    ax = axes[1][0]
    for i_age, age in enumerate(age_groups):
        eff_R_t = np.array(trace.posterior["base_R_t"][..., 14 : len(t) + 14, i_age]).reshape((-1, len(t)))
        ax.plot(t, np.median(eff_R_t, axis=0), color=cmap(1-i_age/9), label=f"age {age}")
        ax.set_xlim(min(t), max(t))
        ax.set_ylabel("base $R_t$")
    format_date_axis(ax)
    ax.grid(True)
    ax = axes[1][1]
    for i_age, age in enumerate(age_groups):
        eff_R_t = np.array(trace.posterior["eff_R_t"][..., 14 : len(t) + 14, i_age]).reshape((-1, len(t)))
        ax.plot(t, np.median(eff_R_t, axis=0), color=cmap(1-i_age/9), label=f"age {age}")
        ax.set_xlim(min(t), max(t))
        ax.set_ylabel("effective $R_t$")
    ax.grid(True)
    ax.legend()
    format_date_axis(ax)

    if "influx_weekly" in trace.posterior.keys():
        ax = axes[0][0]
        k_influx=0.2
        scale_influx = (
                influx / scipy.special.gamma(1 + 1 / k_influx)
        )
        for i_age, age in enumerate(age_groups):
            influx_arr = np.array(trace.posterior["influx_weekly"][..., 2:, i_age]).reshape((-1, len(cases_df.index)))*scale_influx
            ax.plot(cases_df.index, np.median(influx_arr, axis=0), color=cmap(1-i_age/9), label=f"age {age}")
        ax.set_xlim(min(t), max(t))
        ax.set_ylabel("influx (cases/$10^6$)")
        format_date_axis(ax)

        ax = axes[0][1]
        influx_arr = np.array(trace.posterior["influx_weekly"][..., 2:, :]).reshape((-1, len(cases_df.index), len(age_groups)))*scale_influx

        mean_influx = np.sum(influx_arr*model.N_population, axis=-1)/np.sum(model.N_population)
        mean_influx = scipy.ndimage.gaussian_filter1d(mean_influx, sigma=1, mode='nearest')

        ax.plot(
            cases_df.index,
            np.median(mean_influx, axis=0),
            alpha=1,
            color="tab:blue",
        )
        ax.fill_between(
            cases_df.index,
            *np.percentile(mean_influx, axis=(0,), q=(25, 75)),
            alpha=0.3,
            color="tab:blue",
            label="50% & 95% CI"
        )
        ax.fill_between(
            cases_df.index,
            *np.percentile(mean_influx, axis=(0,), q=(2.5, 97.5)),
            alpha=0.2,
            color="tab:blue",
        )
        ax.set_ylabel("mean influx")
        ax.set_xlim(min(t), max(t))
        format_date_axis(ax)
        ax.legend()
    else:
        ax = axes[0][1]
        ax.plot(cases_df.index, influx*np.ones(len(cases_df.index)), color="tab:blue")
        ax.set_ylim(0, influx*1.5)
        ax.set_ylabel("mean influx")
        ax.set_xlim(min(t), max(t))
        format_date_axis(ax)



    if save_dir is None:
        save_dir = os.path.join(os.path.dirname(__file__), "../data/results_inference/")

    save_path = os.path.join(
        save_dir,
        f"Inference_{begin_str}--{end_str}-C_mat={C_mat_param}-"
        f"V1_eff={V1_eff}-V2_eff={V2_eff}-V3_eff={V3_eff}-influx={influx}",
    )

    plt.savefig(save_path + ".pdf")
    plt.show()