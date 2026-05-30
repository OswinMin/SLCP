from pathlib import Path

import numpy as np

import config
from shared_utils import load_result


METHOD_KEYS = ["base", "SDCP", "PPI", "StCP", "StCP-sel", "oracle", "NOAL"]
METHOD_LABELS = ["base", "SDCP", "PPI", "ours", "ours-sel", "oracle", "DP"]
MODEL_LABELS = ["GLCP", "CQR"]


def select_stcp_row(res, method_idx, tol=0.01):
    mar = np.asarray(res["StCP"][0])[method_idx]
    size = np.asarray(res["StCP"][1])[method_idx]
    std = np.asarray(res["StCP"][2])[method_idx]
    mis = np.asarray(res["StCP"][3])[method_idx]
    alpha = float(res["meta"]["alpha"])
    n = int(res["meta"]["n"])
    target = 1.0 - alpha
    low = target - tol
    up = target + 1.0 / (n + 1)
    mask = (mar >= low) & (mar <= up)
    if np.any(mask):
        idx_pool = np.where(mask)[0]
        idx = int(idx_pool[np.argmin(std[idx_pool])])
    else:
        idx = int(np.argmin(np.abs(mar - target)))
    return np.array([mar[idx], size[idx], std[idx], mis[idx]], dtype=float)


def collect_result_row(res, method_idx):
    rows = []
    for key in METHOD_KEYS:
        if key == "StCP":
            rows.append(select_stcp_row(res, method_idx))
        else:
            values = res[key]
            rows.append(
                np.array(
                    [
                        np.asarray(values[0])[method_idx],
                        np.asarray(values[1])[method_idx],
                        np.asarray(values[2])[method_idx],
                        np.asarray(values[3])[method_idx],
                    ],
                    dtype=float,
                )
            )
    return np.vstack(rows)


def coverage_marks(mar, tol=0.01, n=30):
    target = 0.9
    marks = []
    for value in mar:
        if value < target - tol:
            marks.append("^-")
        elif value > target + 1.0 / (n + 1):
            marks.append("^+")
        else:
            marks.append("")
    return marks


def fmt_marginal(value, mark=""):
    return f"$\\mathit{{{value:.3f}}}{mark}$" if mark else f"${value:.3f}$"


def fmt_metric(value, mark="", bold=False, digits=2):
    core = f"{value:.{digits}f}"
    if bold:
        core = f"\\textbf{{{core}}}"
    if mark:
        return f"$\\mathit{{{core}}}{mark}$"
    return f"${core}$"


def valid_baseline_indices(marks):
    baseline_idx = [0, 1, 2]
    valid_idx = [idx for idx in baseline_idx if not marks[idx]]
    return valid_idx if valid_idx else baseline_idx


def fmt_std_block(std, marks):
    out = []
    baseline_idx = valid_baseline_indices(marks)
    baseline_min = np.min(std[baseline_idx])
    base_std = std[0]
    for i, value in enumerate(std):
        if i in [3, 4]:
            improve = 0.0 if abs(base_std) < 1e-12 else (base_std - value) / base_std * 100.0
            bold = (not marks[i]) and (value <= baseline_min + 1e-12)
            value_core = f"\\textbf{{{value:.2f}}}" if bold else f"{value:.2f}"
            if marks[i]:
                out.append(f"$\\mathit{{{value_core}}}{marks[i]}\\,({improve:.1f}\\%)$")
            else:
                out.append(f"${value_core}\\,({improve:.1f}\\%)$")
        else:
            out.append(fmt_metric(value, marks[i], bold=False, digits=2))
    return out


def fmt_size_block(size, marks):
    baseline_idx = valid_baseline_indices(marks)
    baseline_best = np.min(size[baseline_idx])
    out = []
    for i, value in enumerate(size):
        bold = (i != 5) and (not marks[i]) and value <= baseline_best + 1e-12
        out.append(fmt_metric(value, marks[i], bold=bold, digits=2))
    return out


def fmt_mis_block(mis):
    return [f"${value:.3f}$" for value in mis]


def render_longtable(output_path, caption, label, param_symbol, values):
    left_count = 4
    method_count = 7
    total_columns = left_count + method_count
    col_spec = "c" * total_columns

    lines = []
    lines.append("\\begingroup\n")
    lines.append("\\setlength{\\LTleft}{0pt}\n")
    lines.append("\\setlength{\\LTright}{0pt}\n")
    lines.append("\\fontsize{6.0}{7.2}\\selectfont\n")
    lines.append("\\setlength{\\tabcolsep}{2.2pt}\n")
    lines.append("\\renewcommand{\\arraystretch}{0.95}\n")
    lines.append(f"\\begin{{longtable}}{{{col_spec}}}\n")
    lines.append(f"\\caption{{{caption}}}\\label{{{label}}}\\\\\n")
    lines.append("\\toprule\n")
    lines.append("&&&& \\multicolumn{7}{c}{Method}\\\\\n")
    lines.append("\\cmidrule(lr){5-11}\n")
    lines.append(
        f"& Setting & ${param_symbol}$ & Model & "
        + " & ".join(METHOD_LABELS)
        + " \\\\\n"
    )
    lines.append("\\midrule\n")
    lines.append("\\endfirsthead\n")
    lines.append("\\multicolumn{11}{c}{\\tablename\\ \\thetable{} -- continued from previous page}\\\\\n")
    lines.append("\\toprule\n")
    lines.append("&&&& \\multicolumn{7}{c}{Method}\\\\\n")
    lines.append("\\cmidrule(lr){5-11}\n")
    lines.append(
        f"& Setting & ${param_symbol}$ & Model & "
        + " & ".join(METHOD_LABELS)
        + " \\\\\n"
    )
    lines.append("\\midrule\n")
    lines.append("\\endhead\n")
    lines.append("\\midrule\n")
    lines.append("\\multicolumn{11}{r}{Continued on next page}\\\\\n")
    lines.append("\\midrule\n")
    lines.append("\\endfoot\n")
    lines.append("\\bottomrule\n")
    lines.append("\\endlastfoot\n")

    for metric_name in ["Std", "Marginal", "Size", "Miscoverage"]:
        first_row = True
        for dtype in config.DISPLAY_DTYPES:
            for value in values:
                if param_symbol == "n":
                    res = load_result(dtype, int(value), int(config.FIXED_M))
                    n_for_marks = int(value)
                else:
                    res = load_result(dtype, int(config.FIXED_N), int(value))
                    n_for_marks = int(config.FIXED_N)

                row_glcp = collect_result_row(res, 0)
                row_cqr = collect_result_row(res, 1)
                marks_glcp = coverage_marks(row_glcp[:, 0], n=n_for_marks)
                marks_cqr = coverage_marks(row_cqr[:, 0], n=n_for_marks)

                if metric_name == "Std":
                    glcp_cells = fmt_std_block(row_glcp[:, 2], marks_glcp)
                    cqr_cells = fmt_std_block(row_cqr[:, 2], marks_cqr)
                elif metric_name == "Marginal":
                    glcp_cells = [fmt_marginal(v, m) for v, m in zip(row_glcp[:, 0], marks_glcp)]
                    cqr_cells = [fmt_marginal(v, m) for v, m in zip(row_cqr[:, 0], marks_cqr)]
                elif metric_name == "Size":
                    glcp_cells = fmt_size_block(row_glcp[:, 1], marks_glcp)
                    cqr_cells = fmt_size_block(row_cqr[:, 1], marks_cqr)
                else:
                    glcp_cells = fmt_mis_block(row_glcp[:, 3])
                    cqr_cells = fmt_mis_block(row_cqr[:, 3])

                prefix = metric_name if first_row else ""
                first_row = False
                lines.append(
                    f"{prefix} & {config.dtype_label(dtype)} & ${int(value)}$ & {MODEL_LABELS[0]} & "
                    + " & ".join(glcp_cells)
                    + " \\\\\n"
                )
                lines.append(
                    f" &  &  & {MODEL_LABELS[1]} & "
                    + " & ".join(cqr_cells)
                    + " \\\\\n"
                )
        if metric_name != "Miscoverage":
            lines.append("\\midrule\n")

    lines.append("\\end{longtable}\n")
    lines.append("\\endgroup\n")
    Path(output_path).write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    config.validate_config()
    sim_dir = Path(__file__).resolve().parent
    mytex_fig_dir = sim_dir.parent / "MyTex" / "fig"
    mytex_fig_dir.mkdir(parents=True, exist_ok=True)

    render_longtable(
        mytex_fig_dir / "sim_table_n.tex",
        "Simulation results with $m=500$ and $n\\in\\{30,100,500\\}$ across three representative sigma settings. In the Marginal block, ``$-$'' indicates marginal coverage below $1-\\alpha-0.01$, and ``$+$'' indicates marginal coverage above $1-\\alpha+1/(n+1)$.",
        "tab:sim-vary-n",
        "n",
        config.DISPLAY_N_VALUES,
    )
    render_longtable(
        mytex_fig_dir / "sim_table_m.tex",
        "Simulation results with $n=30$ and $m\\in\\{30,100,500\\}$ across three representative sigma settings. In the Marginal block, ``$-$'' indicates marginal coverage below $1-\\alpha-0.01$, and ``$+$'' indicates marginal coverage above $1-\\alpha+1/(n+1)$.",
        "tab:sim-vary-m",
        "m",
        config.DISPLAY_M_VALUES,
    )
    print(f"[Done] saved: {mytex_fig_dir / 'sim_table_n.tex'}")
    print(f"[Done] saved: {mytex_fig_dir / 'sim_table_m.tex'}")
