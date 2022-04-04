import logging

log = logging.getLogger(__name__)
import datetime
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

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
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4, byweekday=mdates.SU))
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
