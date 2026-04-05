import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter
import mplhep as hep

hep.style.use(hep.style.ATLAS)
ATLAS_LABEL_TEXT = "Internal Simulation"


def _rel_err_func(a, b):
    if  np.isnan(a).any() or np.isnan(b).any():
        print("Warning: NaN values detected in input arrays.")
    if (np.any(b == 0)):
        print("Warning: Zero values detected in denominator array.")
    mask = ~np.isnan(a) & ~np.isnan(b) & (b != 0)
    return (a[mask] - b[mask]) / b[mask]

def _err_bounds(x, rel_err=0.2, offset=5.0):
    delta = rel_err * np.abs(x) + offset
    upper = x + delta
    lower = x - delta
    return upper, lower

def _rmse(pred, truth):
	mask = np.isfinite(pred) & np.isfinite(truth)
	if np.sum(mask) != len(pred):
		print(f"Warning: {len(pred) - np.sum(mask)} invalid entries found")
	pred = pred[mask]
	truth = truth[mask]
	return np.sqrt(np.mean((pred - truth) ** 2))


def plot_1d_hist(
    pred,
    truth,
    name,
    bins_edges=np.linspace(-200, 200, 51),
    unit="GeV",
    color="black",
    savepath=None,
    ratio_ylim=(0.5, 1.5),
):
    fig, (ax, rax) = plt.subplots(
        2,
        1,
        figsize=(7, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [3.5, 1.0], "hspace": 0.1},
    )

    # Top panel: truth and prediction histograms
    pred_counts, _ = np.histogram(pred, bins=bins_edges)
    truth_counts, _ = np.histogram(truth, bins=bins_edges)

    ax.hist(pred, bins=bins_edges, linewidth=2, color="red", histtype="step", label="Pred")
    ax.hist(truth, bins=bins_edges, linewidth=2, color="blue", histtype="step", label="True")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylabel("Events", loc="top")

    txt = hep.atlas.label(ATLAS_LABEL_TEXT, data=True, loc=2, rlabel="", ax=ax)
    txt[0].set_color(color)
    txt[1].set_color(color)
    ax.set_ylim(0, 1.15 * max(pred_counts.max(), truth_counts.max(), 1))
    ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax.tick_params(axis="y", which="major", pad=12)

    # Bottom panel: Pred/True ratio
    pred_counts, _ = np.histogram(pred, bins=bins_edges)
    truth_counts, _ = np.histogram(truth, bins=bins_edges)
    pred_counts = pred_counts.astype(float)
    truth_counts = truth_counts.astype(float)
    bin_centers = 0.5 * (bins_edges[1:] + bins_edges[:-1])

    valid = (pred_counts > 0) & (truth_counts > 0)
    ratio = np.full_like(pred_counts, np.nan, dtype=float)
    ratio_err = np.full_like(pred_counts, np.nan, dtype=float)

    ratio[valid] = pred_counts[valid] / truth_counts[valid]
    # Propagated Poisson uncertainty for r = N_pred / N_true:
    # sigma_r = r * sqrt(1/N_pred + 1/N_true)
    ratio_err[valid] = ratio[valid] * np.sqrt(1.0 / pred_counts[valid] + 1.0 / truth_counts[valid])

    # Reference uncertainty band from denominator (truth) statistics around unity
    truth_rel_err = np.full_like(truth_counts, np.nan, dtype=float)
    truth_mask = truth_counts > 0
    truth_rel_err[truth_mask] = 1.0 / np.sqrt(truth_counts[truth_mask])
    band_low = 1.0 - truth_rel_err
    band_high = 1.0 + truth_rel_err
    rax.fill_between(bin_centers, band_low, band_high, step="mid", color="gray", alpha=0.25, linewidth=0)

    y_min, y_max = ratio_ylim
    span = y_max - y_min

    in_view = valid & (ratio >= y_min) & (ratio <= y_max)
    overflow_high = valid & (ratio > y_max)
    overflow_low = valid & (ratio < y_min)

    rax.axhline(1.0, color="gray", linestyle="--", linewidth=1.3)
    rax.errorbar(
        bin_centers[in_view],
        ratio[in_view],
        yerr=ratio_err[in_view],
        fmt="o",
        color="black",
        markersize=3,
        linewidth=1,
        capsize=0,
    )

    # Draw arrows at the panel edge for overflow points
    y_top = y_max 
    y_top_from = y_max - 0.15 * span
    y_bot = y_min 
    y_bot_from = y_min + 0.15 * span

    for x in bin_centers[overflow_high]:
        rax.annotate(
            "",
            xy=(x, y_top),
            xytext=(x, y_top_from),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5),
            clip_on=False,
        )

    for x in bin_centers[overflow_low]:
        rax.annotate(
            "",
            xy=(x, y_bot),
            xytext=(x, y_bot_from),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=0.5),
            clip_on=False,
        )

    rax.set_xlabel(name + " [" + unit + "]", loc="right")
    rax.set_ylabel("Pred/True")
    rax.set_ylim(*ratio_ylim)
    rax.grid(axis="y", linestyle="--", alpha=0.35)
    rax.tick_params(axis="both", which="major", pad=10)

    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()

def plot_2d_hist(pred, truth, name, bins_edges=np.linspace(-200, 200, 51), log=False, unit="GeV", color="black", vmax=1e3, offset=0.5, savepath=None):
    err = 0.2
    cor_mask = np.abs(_rel_err_func(pred, truth)) <= err # set 20% relative error cut
    fig, ax = plt.subplots()
    if log:
        norm = LogNorm(vmin=1, vmax=vmax)
        ax.hist2d(pred, truth, bins=[bins_edges, bins_edges], cmap="viridis", norm=norm)
    else:
        ax.hist2d(pred, truth, bins=[bins_edges, bins_edges], cmap="viridis", vmin=1, vmax=vmax)

    # Plot error guides aligned with bin edges
    # upper, lower = _err_bounds(bins_edges, rel_err=err, offset=offset)
    # plt.plot(upper, bins_edges, color="gainsboro", linestyle="-", label=r"$\pm 20\% offset$")
    ax.plot(bins_edges, bins_edges, color="gainsboro", linestyle="--")
    # plt.plot(lower, bins_edges, color="gainsboro", linestyle="-")

    ax.set_xlabel(f"Pred [{unit}]")
    ax.set_ylabel(f"True [{unit}]")
    ax.set_title(f"{name}" + f" (RMSE: {_rmse(pred, truth):.2f})", loc="right")
    print(f"Rel err < 20%: {100*np.sum(cor_mask)/len(truth):.2f} %")

    txt = hep.atlas.label(ATLAS_LABEL_TEXT, data=True, loc=2, rlabel="", ax=ax)
    txt[0].set_color(color)
    txt[1].set_color(color)
    ax.tick_params(axis="both", which="major", pad=10)

    ticks = np.linspace(bins_edges[0], bins_edges[-1], 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    x_min, x_max = bins_edges[0], bins_edges[-1]
    y_min, y_max = bins_edges[0], bins_edges[-1]
    x_margin = 0.02 * (x_max - x_min)
    ax.set_xlim(x_min, x_max + x_margin)
    ax.set_ylim(y_min, y_max)

    fig.colorbar(ax.collections[0], ax=ax, label="Events")
    ax.set_aspect("equal", adjustable="box")  # Make plot square
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()
    # if log:
    #     plt.hist2d(pred[cor_mask], truth[cor_mask], bins=[bins_edges, bins_edges], cmap="viridis", norm=norm)
    # else:
    #     plt.hist2d(pred[cor_mask], truth[cor_mask], bins=[bins_edges, bins_edges], cmap="viridis", vmin=1, vmax=vmax)
    # # Plot error guides aligned with bin edges
    # upper, lower = _err_bounds(bins_edges, rel_err=err)
    # plt.plot(upper, bins_edges, color="red", linestyle="-", label=r"$\pm 20\% offset$")
    # plt.plot(lower, bins_edges, color="red", linestyle="-")
    # plt.xlabel(f"Pred [{unit}]")
    # plt.ylabel(f"True [{unit}]")
    # plt.title(f"{name} (Rel err < 20%)"+f" with _RMSE: {_rmse(pred[cor_mask], truth[cor_mask]):.2f}", loc="right")
    # txt = hep.atlas.label(ATLAS_LABEL_TEXT, data=True, loc=2, rlabel="")
    # txt[0].set_color(color)
    # txt[1].set_color(color)
    # plt.tick_params(axis="both", which="major", pad=10) 
    # plt.colorbar(label="Events")
    # plt.gca().set_aspect("equal", adjustable="box")  # Make plot square
    # # if savepath is not None:
    # #     plt.savefig(savepath, bbox_inches="tight")
    
def plot_2d_res_hist(pred, truth, name_pos, name_neg, bins_edges=np.linspace(-200, 200, 51), log=False, unit="GeV", color="black", vmax=5e3, savepath=None):
    fig, ax = plt.subplots()
    if log:
        norm = LogNorm(vmin=1, vmax=vmax)
        ax.hist2d(pred, truth, bins=[bins_edges, bins_edges], cmap="viridis", norm=norm)
    else:
        ax.hist2d(pred, truth, bins=[bins_edges, bins_edges], cmap="viridis", vmin=1, vmax=vmax)
    ax.set_xlabel(rf"$\Delta_\text{{res}}${name_pos} [{unit}]")
    ax.set_ylabel(rf"$\Delta_\text{{res}}${name_neg} [{unit}]")
    txt = hep.atlas.label(ATLAS_LABEL_TEXT, data=True, loc=0, rlabel="", ax=ax)
    txt[0].set_color(color)
    txt[1].set_color(color)
    ax.tick_params(axis="both", which="major", pad=10)
    ticks = np.linspace(bins_edges[0], bins_edges[-1], 5)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    fig.colorbar(ax.collections[0], ax=ax, label="Events")
    ax.set_aspect("equal", adjustable="box")  # Make plot square
    if savepath is not None:
        fig.savefig(savepath, bbox_inches="tight")
    plt.show()