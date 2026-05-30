import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import config
from shared_utils import ensure_dir, figure_dir, load_result


plt.rcParams["font.size"] = 15


def pick_method(metric_obj, method_idx):
    arr = np.asarray(metric_obj)
    if arr.ndim == 0:
        return float(arr)
    return float(arr[method_idx])


def safe_improve(std_curve, base_std, orac_std):
    denom = base_std - orac_std
    if np.abs(denom) < 1e-12:
        return np.full_like(std_curve, np.nan, dtype=float)
    return (1.0 - (std_curve - orac_std) / denom) * 100.0


def make_single_lambda_plot(dtype_name):
    method_info = [("GLCP", 0), ("SCC", 1)]
    method_name2 = {"GLCP": "GLCP", "SCC": "CQR"}
    res = load_result(dtype_name, config.LAMBDA_N, config.LAMBDA_M)
    lbds = np.asarray(res["meta"]["lbds"], dtype=float)
    alpha = float(res["meta"]["alpha"])
    n = int(res["meta"]["n"])
    target_cov = 1.0 - alpha
    low_bound = target_cov - 0.01
    up_bound = target_cov + 1.0 / (n + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 6))
    legend_handles = None
    legend_labels = None
    improve_handle = None
    star_handle = None

    for row, (method_name, method_idx) in enumerate(method_info):
        title = f"{config.dtype_label(dtype_name)} - {method_name2[method_name]}"
        ax_mar = axes[row, 0]
        ax_std = axes[row, 1]

        mar_curve = np.asarray(res["StCP"][0])[method_idx]
        std_curve = np.asarray(res["StCP"][2])[method_idx]
        sel_mar = pick_method(res["StCP-sel"][0], method_idx)
        base_std = pick_method(res["base"][2], method_idx)
        orac_std = pick_method(res["oracle"][2], method_idx)
        sel_std = pick_method(res["StCP-sel"][2], method_idx)

        ax_mar.axhline(target_cov, color="gray", linestyle=":", linewidth=1.0)
        ax_mar.axhline(low_bound, color="gray", linestyle="-.", linewidth=2.0, label="Mar-bound")
        ax_mar.axhline(up_bound, color="gray", linestyle="-.", linewidth=2.0)
        ax_mar.axhline(sel_mar, color="tab:red", linestyle="-.", linewidth=2.0, label="StCP-Sel")
        ax_mar.plot(lbds, mar_curve, color="tab:blue", marker="o", linewidth=2.0, label="StCP")
        ax_mar.set_ylabel("Marginal")
        ax_mar.set_title(title)
        ax_mar.set_xlabel(r"$\lambda$")
        ax_mar.set_ylim((0.8, 1.0))

        improve = safe_improve(std_curve, base_std, orac_std)
        ax_std_r = ax_std.twinx()
        improve_shade = ax_std_r.fill_between(lbds, 0, improve, alpha=0.25, color="tab:green")
        ax_std_r.plot(lbds, improve, color="tab:green", linestyle="-", linewidth=0.5)
        ax_std_r.plot(lbds, np.zeros_like(lbds), color="tab:green", linestyle="-", linewidth=0.5)
        sel_std = float(res["StCP-sel"][2][method_idx])
        target_lbds = np.interp(sel_std, std_curve[::-1], lbds[::-1])
        target_improve = np.interp(target_lbds, lbds, improve)
        star_scatter = ax_std_r.scatter(
            [target_lbds],
            [target_improve],
            color="tab:red",
            marker="*",
            s=150,
            zorder=5,
            label="Improve-Sel %",
        )
        ax_std_r.set_ylabel("Improve %")

        ax_std.axhline(base_std, color="tab:grey", linestyle="--", linewidth=1.8, label="Base")
        ax_std.axhline(sel_std, color="tab:red", linestyle="-.", linewidth=2.0, label="StCP-Sel")
        ax_std.plot(lbds, std_curve, color="tab:blue", marker="o", linewidth=2.0, label="StCP")
        ax_std.set_title(title)
        ax_std.set_ylabel("Std")
        ax_std.set_xlabel(r"$\lambda$")
        up = max(base_std, sel_std, np.max(std_curve))
        low = min(base_std, sel_std, np.min(std_curve))
        ax_std.set_ylim((low - 0.2 * (up - low), up + 0.2 * (up - low)))

        if improve_handle is None:
            improve_handle = improve_shade
        if star_handle is None:
            star_handle = star_scatter
        if legend_handles is None:
            legend_handles, legend_labels = ax_mar.get_legend_handles_labels()

    if improve_handle is not None:
        legend_handles = legend_handles + [improve_handle, star_handle]
        legend_labels = legend_labels + ["Improve %", "Improve-Sel %"]
    fig.legend(legend_handles, legend_labels, loc="upper center", ncol=min(8, len(legend_handles)), frameon=False)
    plt.tight_layout()
    plt.subplots_adjust(top=0.86)
    return fig


if __name__ == "__main__":
    config.validate_config()
    repo_dir = Path(__file__).resolve().parent.parent
    mytex_fig_dir = repo_dir / "MyTex" / "fig"
    ensure_dir(figure_dir())
    ensure_dir(mytex_fig_dir)

    for dtype_name in config.DISPLAY_DTYPES:
        fig = make_single_lambda_plot(dtype_name)
        out_name = f"lambda_sensitivity_{dtype_name}.pdf"
        out_path = os.path.join(figure_dir(), out_name)
        fig.savefig(out_path, format="pdf")
        fig.savefig(mytex_fig_dir / out_name, format="pdf")
        plt.close(fig)
        print(f"[Done] saved: {out_path}")
