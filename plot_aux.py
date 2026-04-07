import os

import matplotlib
if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
    matplotlib.use('Agg')
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
import sys

import glob


# This function is vibe coded, pretty sure there is a cleaner way to do continuous hue
def _param_to_colour(base_colour, t, lighten_frac=0.2):
    """
    Map a normalised parameter t in [0, 1] to a colour that goes from
    a lighter version of base_colour (t=0) to black (t=1).
    Keeps the algorithm-to-colour mapping while varying shade by parameter.
    lighten_frac: how much to mix base with white for the "low" end (0 = no lighten, 1 = white).
    """
    rgb = np.array(mpl.colors.to_rgb(base_colour))
    white = np.array([1.0, 1.0, 1.0])
    light_rgb = (1 - lighten_frac) * rgb + lighten_frac * white
    black = np.array([0.0, 0.0, 0.0])
    # t=0 -> rgb, t=1 -> black
    blend = (1 - t) * rgb + t * black
    blend = np.clip(blend, 0, 1)
    return mpl.colors.to_hex(blend)


def make_lineplot_df(
    df, 
    x_axis_key, 
    y_axis_key, 
    hue_key=None, 
    style_key=None, 
    title=None, 
    x_axis_lab=None, 
    y_axis_lab=None, 
    legend_lab=None,
    x_log_scale=False,
    y_log_scale=False,
    y_scale_transform_forward=None,
    y_scale_transform_inverse=None,
    filename=None, 
    vertical_lines=None,
    palette=None,
    dashes=None,
    markers=None,
    markevery=1,
    y_axis_range=None,
    alpha=1.0,
    use_legend=True,
    font_scale=1.2,
    figsize=(10.24, 7.68),
    dpi=200,
    legend_loc=None,
    horizontal_lines=None):
    """
    General helper for plotting lineplots in our style.
    """

    # Default matplotlib output is often ~640x480 (dpi=100). Increase dpi so
    # saved images are higher resolution without changing plot physical size.
    plt.figure(figsize=figsize, dpi=dpi)
    sns.set_theme(style="darkgrid",font_scale=font_scale)

    # params = {
    #     'axes.labelsize': 48,
    #     'axes.titlesize': 48, 
    #     # 'text.fontsize': 20, 
    #     'legend.fontsize': 48, 
    #     # 'xtick.labelsize': 48, 
    #     # 'ytick.labelsize': 48,
    # }
    # mpl.rcParams.update(params)
    # # mpl.rcParams['font.size'] = font_size
    # # mpl.rcParams.update({'font.size': font_size})

    if y_scale_transform_forward is not None and y_scale_transform_inverse is not None:
        plt.yscale("function", functions=(y_scale_transform_forward, y_scale_transform_inverse))

    if palette is None:
        palette = "deep"
    if dashes is None:
        dashes = False
    if markers is None:
        markers = False
    if legend_loc is None:
        legend_loc = "lower right"

    sns.lineplot(
        data=df, 
        x=x_axis_key, 
        y=y_axis_key, 
        hue=hue_key, 
        style=style_key, 
        palette=palette, 
        dashes=dashes, 
        markers=markers,
        markevery=markevery,
        markersize=14,
        mec=None,
        alpha=alpha)

    if title is not None:
        plt.title(title)
    if x_axis_lab is not None:
        plt.xlabel(x_axis_lab)
    if y_axis_lab is not None:
        plt.ylabel(y_axis_lab)
    if horizontal_lines is not None:
        for y in horizontal_lines:
            plt.axhline(y=y, color='k', linestyle='--')
    if vertical_lines is not None:
        for x in vertical_lines:
            plt.axvline(x=x, color='k', linestyle='--')
    if legend_lab is not None:
        plt.legend(loc=legend_loc, title=legend_lab)
    if y_axis_range is not None:
        plt.gca().set_ylim(y_axis_range)
    if not use_legend:
        plt.gca().get_legend().remove()
    if x_log_scale:
        plt.xscale('log')
    if y_log_scale:
        plt.yscale('log')

    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    plt.close()

def read_hpopt_file_to_df(filename):
    df = pd.read_csv(filename)
    df = df.rename(columns={"eval(mc_estimate_expected_utility)": "V"})
    return df

def subdf(df, key, val):
    subdf = df[df[key] == val]
    return subdf

def read_eval_file_to_df(filename,num_trials_scale):
    """
    Reads the eval file from 'filename'
    Appends results in this eval file to the arrays: alg_ids, run_idxs, runtimes, num_trialss, 

    Manually reads the first two lines to get params that the algorithm was run with (usually the algorithm name + params)
    At the moment we just care about the algorithm name

    Then uses pandas to read 2nd half, which is a csv file
    And adds an additional collumn with the algorithm name
    """

    alg_id = None
    temp = None
    bias = None
    heuristic_value = None
    entropy_coeff = None
    with open(filename) as f:
        
        # first two lines are for human readability
        f.readline()
        f.readline()

        # next two lines are xpr level params
        _xpr_param_ids = f.readline().strip().split(",")
        _xpr_param_vals = f.readline().strip().split(",")
        
        # next three lines are for human readability
        f.readline()
        f.readline()
        f.readline()
        
        # Then next two lines are the values that we care about 
        param_ids = f.readline().strip().split(",")
        param_vals = f.readline().strip().split(",")

        # Get params for alg
        for param_id, val in zip(param_ids,param_vals):
            if param_id == "alg_id":
                alg_id = val
            elif param_id == "temp":
                temp = float(val)
            elif param_id == "bias":
                bias = float(val)
            elif param_id == "heuristic_value":
                heuristic_value = float(val)
            elif param_id == "entropy_coeff":
                entropy_coeff = float(val)
            elif param_id == "normalise_entropy_before_adding" and val == "0":
                alg_id = "dents(no_norm)"

    df = pd.read_csv(filepath_or_buffer=filename, header=12, index_col=False, skip_blank_lines=False)
    df["alg_id"] = alg_id
    df["temp"] = temp
    df["bias"] = bias
    if heuristic_value is not None:
        df["heuristic_value"] = heuristic_value
    if entropy_coeff is not None:
        df["entropy_coeff"] = entropy_coeff
    df["num_trials"] /= num_trials_scale

    df = df.rename(columns={
        "alg_id": "alg_id",
        "run_idx": "run_idx",
        "runtime": "runtime",
        "num_trials": "num_trials",
        "eval": "mc_val",
        "eval_std": "mc_std",
    }, errors="raise")
    
    return df

def get_env_id(filename):
    """
    Environment name is part of the filename
    filenames are of the form results/<expr_id>/<env_id>/<alg_id>/<alg_param_1>/.../<alg_param_N>/eval.txt
    """
    return filename.split("/")[2]


def read_eval_files_to_df(filenames,num_trials_scale):
    """
    Reads dataframes for each eval file in 'filenames' and concatinates them all into one big dataframe

    Adds the env_id for each environment to each dataframe, and if the environment was a tree, then add additional 
    columns for the params of the environment
    """
    dfs = []
    for filename in filenames:
        df = read_eval_file_to_df(filename,num_trials_scale)
        df["env_id"] = get_env_id(filename)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def make_eval_plot(
    filenames, 
    plot_filename, 
    hue_key=None, 
    title=None, 
    x_axis_key=None,
    x_axis_lab=None, 
    y_axis_key=None,
    y_axis_lab=None, 
    legend_lab=None, 
    x_axis_truncate=None,
    y_scale_transform_forward=None,
    y_scale_transform_inverse=None,
    y_axis_range=None,
    add_markers=False,
    markevery=1,
    use_legend=True,
    alpha=1.0,
    num_trials_scale=1,
    horizontal_lines=None,
    continuous_hue=False,
    continuous_hue_key_is_logarithmic=False,
    add_dashes=False,
    legend_loc=None,
    font_scale=1.2,
    path_len_plot=False,
    max_path_len=100,
    ):
    """
    Makes an eval plot using the data in the given filenames
    Can make a non eum plot by specifying y_axis_key.
    If continuous_hue is set (e.g. "heuristic_value"), line colour varies smoothly
    from a lighter version of the algorithm colour (low param) to black (high param).
    If continuous_hue_key_is_logarithmic is True, the parameter is normalised on a log scale for the palette.
    """

    if (len(filenames) == 0):
        print(f"Skipping plot {plot_filename} because no files found")
        return

    # Default params
    if hue_key is None:
        hue_key = "alg_id"
    if x_axis_key is None:
        x_axis_key = "num_trials"
    if x_axis_lab is None:
        x_axis_lab = "Num Trials"
    if y_axis_key is None:
        y_axis_key = "mc_val" if not path_len_plot else "path_len"
    if y_axis_lab is None:
        y_axis_lab = "Monte Carlo Value Estimate" if not path_len_plot else "Path Length"
    if title is None:
        title = y_axis_lab + " vs " + x_axis_lab
    if legend_lab is None and use_legend:
        legend_lab = "Algorithm"

    # Update x_axis lable if applying scaling
    if num_trials_scale > 1:
        x_axis_lab += " (x{scale})".format(scale=num_trials_scale)

    # Read in data + make algorithm names more pretty
    bts_str = "BTS"
    dents_str = "DENTS"
    uct_str = "UCT"
    max_uct_str = "MAX-UCT"
    hmcts_str = "HMCTS"
    ments_str = "MENTS"
    rents_str = "RENTS"
    tents_str = "TENTS"


    df = read_eval_files_to_df(filenames, num_trials_scale)
    df["alg_id"] = df["alg_id"].map({
        "bts": bts_str,
        "dents": dents_str,
        "dents(no_norm)": dents_str + "(no_norm)",
        "uct": uct_str,
        "maxuct": max_uct_str,
        "hmcts": hmcts_str,
        "ments": ments_str,
        "rents": rents_str,
        "tents": tents_str,
    })

    df["path_len"] = np.log(df["mc_val"]) / np.log(0.99)
    df.loc[df["mc_val"] == 0.0, "path_len"] = max_path_len

    # Get the set of alg ids working with
    alg_id_set = set(df["alg_id"])
    
    # Define line styles - (colour) palette
    # N.B. palette can be a colourmap: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
    # Currently using dict for mapping using colours from: https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
    palette = {}
    for alg_id in alg_id_set:
        if bts_str in alg_id:
            palette[alg_id] = "tab:blue"
        if dents_str in alg_id:
            palette[alg_id] = "tab:orange"
        if dents_str + "(no_norm)" in alg_id:
            palette[alg_id] = plt.cm.tab20(3) # light orange
        if uct_str in alg_id:
            palette[alg_id] = "tab:green"
        if max_uct_str in alg_id:
            palette[alg_id] = "tab:red" #plt.cm.tab20(12) # light green
        if hmcts_str in alg_id:
            palette[alg_id] = "tab:purple" #"tab:red"
        if ments_str in alg_id:
            palette[alg_id] = "tab:brown" #"tab:purple"
        if rents_str in alg_id:
            palette[alg_id] = "tab:pink" #"tab:brown"
        if tents_str in alg_id:
            palette[alg_id] = "tab:gray" #"tab:pink"

    # Sort hue key lexicographically so legend order is deterministic
    sorted_hue_vals = sorted(df[hue_key].unique())
    df[hue_key] = pd.Categorical(df[hue_key], categories=sorted_hue_vals, ordered=True)

    # Define line styles - dashes
    # When add_dashes is True, algorithms with multiple bias/temp values get
    # separate lines distinguished by dash pattern (same base colour).
    dashes = None
    if add_dashes:
        algs_with_multi_bias = set()
        algs_with_multi_temp = set()
        for aid in df["alg_id"].unique():
            adf = df[df["alg_id"] == aid]
            if "bias" in adf.columns and adf["bias"].dropna().nunique() > 1:
                algs_with_multi_bias.add(aid)
            if "temp" in adf.columns and adf["temp"].dropna().nunique() > 1:
                algs_with_multi_temp.add(aid)

        def _make_alg_label(row):
            base = row["alg_id"]
            if base in algs_with_multi_bias and pd.notna(row.get("bias")):
                return f"{base} (b={row['bias']:g})"
            if base in algs_with_multi_temp and pd.notna(row.get("temp")):
                return f"{base} (t={row['temp']:g})"
            return base

        df["alg_label"] = df.apply(_make_alg_label, axis=1)
        hue_key = "alg_label"

        sorted_alg_ids = sorted(palette.keys(), key=len, reverse=True)
        new_palette = {}
        for label in df["alg_label"].unique():
            for aid in sorted_alg_ids:
                if label.startswith(aid):
                    new_palette[label] = palette[aid]
                    break
        palette = new_palette

        dash_patterns = [
            "",
            (8, 4),
            (2, 4),
            # (4, 2, 1, 2),
            # (6, 2, 1, 2, 1, 2),
            # (8, 2),
            # (2, 4),
        ]
        multi_param_algs = algs_with_multi_bias | algs_with_multi_temp

        unique_labels = sorted(df["alg_label"].unique())
        df["alg_label"] = pd.Categorical(df["alg_label"], categories=unique_labels, ordered=True)

        dashes = {}
        dash_counters = {}
        for label in unique_labels:
            is_multi = any(label.startswith(a) for a in multi_param_algs)
            if is_multi:
                base = next(a for a in multi_param_algs if label.startswith(a))
                idx = dash_counters.get(base, 0)
                dashes[label] = dash_patterns[idx % len(dash_patterns)]
                dash_counters[base] = idx + 1
            else:
                dashes[label] = ""

    # Define line styles - markers (currently unused, but dont want del setup)
    # N.B. see following for valid values: https://matplotlib.org/stable/api/markers_api.html
    markers = None
    if add_markers:
        pass
    # if add_markers:
    #     markers = {}
    #     for alg_id in alg_set:
    #         markers[alg_id] = ""
    #         if "UCT" in alg_id:
    #             markers[alg_id] = 5
    #         if "MENTS" in alg_id:
    #             markers[alg_id] = 7
    #         if "DENTS" in alg_id:
    #             markers[alg_id] = 6

    # Truncate x_axis if want
    if x_axis_truncate is not None:
        df = df[df[x_axis_key] <= x_axis_truncate]

    # Continuous hue: one line per value of hue_key, colour from light(alg) to black
    if continuous_hue:
        # check only one alg id is in the data
        if len(df["alg_id"].unique()) > 1:
            raise ValueError("Continuous hue is only supported for single-alg data")
        
        # create a new palette for the continuous hue
        alg_base_colour = palette[df["alg_id"].unique()[0]]
        param_vals = df[hue_key].dropna().unique()
        pmin, pmax = float(np.nanmin(param_vals)), float(np.nanmax(param_vals))
        if continuous_hue_key_is_logarithmic and pmin > 0 and pmax > 0:
            log_min, log_max = np.log(pmin), np.log(pmax)
            log_span = (log_max - log_min) if log_max > log_min else 1.0
            palette = {v: _param_to_colour(alg_base_colour, (np.log(float(v)) - log_min) / log_span) for v in param_vals}
        else:
            span = (pmax - pmin) if pmax > pmin else 1.0
            palette = {v: _param_to_colour(alg_base_colour, (float(v) - pmin) / span) for v in param_vals}

    # Call our actual make plot function
    make_lineplot_df(
        df=df, 
        x_axis_key=x_axis_key, 
        y_axis_key=y_axis_key, 
        title=title,
        hue_key=hue_key,
        palette=palette,
        style_key=hue_key,
        dashes=dashes,
        x_axis_lab=x_axis_lab,
        y_axis_lab=y_axis_lab,
        y_scale_transform_forward=y_scale_transform_forward,
        y_scale_transform_inverse=y_scale_transform_inverse,
        legend_lab=legend_lab,
        filename=plot_filename,
        y_axis_range=y_axis_range,
        markers=markers,
        markevery=markevery,
        use_legend=use_legend,
        legend_loc=legend_loc,
        alpha=alpha,
        font_scale=font_scale,
        horizontal_lines=horizontal_lines)

def make_param_sens_plot(
    filenames, 
    plot_filename=None,
    plot_filename_frmt_str=None, 
    hue_key=None, 
    title=None, 
    x_axis_key=None,
    x_axis_lab=None, 
    y_axis_key=None,
    y_axis_lab=None, 
    legend_lab=None, 
    x_axis_truncate=None,
    y_scale_transform_forward=None,
    y_scale_transform_inverse=None,
    y_axis_range=None,
    add_markers=False,
    markevery=1,
    use_legend=True,
    alpha=1.0,
    num_trials_scale=1,
    horizontal_lines=None,
    legend_loc=None,
    seperate_plots=True,
    fl_sparse_reward_transform=False,
    font_scale=1.2):
    """
    Makes an plot to compare  
    """

    if (len(filenames) == 0):
        print(f"Skipping plot {plot_filename} because no files found")
        return

    if seperate_plots and plot_filename_frmt_str is None:
        raise ValueError("plot_filename_frmt_str must be provided if seperate_plots is True")
    if not seperate_plots and plot_filename is None:
        raise ValueError("plot_filename must be provided if seperate_plots is False")

    # SCALE_DIFF: Default params
    if hue_key is None:
        hue_key = "alg_id"
    if y_axis_key is None:
        y_axis_key = "mc_val"
    if y_axis_lab is None:
        y_axis_lab = "Monte Carlo Value Estimate"
    if legend_lab is None and use_legend:
        legend_lab = "Algorithm"


    # Read in data + make algorithm names more pretty
    bts_str = "BTS"
    dents_str = "DENTS"
    uct_str = "UCT"
    max_uct_str = "MAX-UCT"
    hmcts_str = "HMCTS"
    ments_str = "MENTS"
    rents_str = "RENTS"
    tents_str = "TENTS"


    df = read_eval_files_to_df(filenames, num_trials_scale)
    df["alg_id"] = df["alg_id"].map({
        "bts": bts_str,
        "dents": dents_str,
        "dents(no_norm)": dents_str + "(no_norm)",
        "uct": uct_str,
        "maxuct": max_uct_str,
        "hmcts": hmcts_str,
        "ments": ments_str,
        "rents": rents_str,
        "tents": tents_str,
    })

    # PARAM_SENS_DIFF: only care about the final results after the search (assumes all experiments run for same search time)
    experiment_num_trials = df["num_trials"].max()
    df = df[df["num_trials"] == experiment_num_trials]

    # pd.set_option('display.max_rows', None)
    # mdf = df[df["alg_id"] == ments_str]
    # mdf = mdf[mdf["temp"] > 0.1]
    # mdf = mdf[mdf["temp"] < 0.6]
    # print(mdf)

    # Get the set of alg ids working with
    alg_id_set = set(df["alg_id"])
    
    # Define line styles - (colour) palette
    # N.B. palette can be a colourmap: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
    # Currently using dict for mapping using colours from: https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
    palette = {}
    for alg_id in alg_id_set:
        if bts_str in alg_id:
            palette[alg_id] = "tab:blue"
        if dents_str in alg_id:
            palette[alg_id] = "tab:orange"
        if dents_str + "(no_norm)" in alg_id:
            palette[alg_id] = plt.cm.tab20(3) # light orange
        if uct_str in alg_id:
            palette[alg_id] = "tab:green"
        if max_uct_str in alg_id:
            palette[alg_id] = "tab:red" 
        if hmcts_str in alg_id:
            palette[alg_id] = "tab:purple" #"tab:red"
        if ments_str in alg_id:
            palette[alg_id] = "tab:brown" #"tab:purple"
        if rents_str in alg_id:
            palette[alg_id] = "tab:pink" #"tab:brown"
        if tents_str in alg_id:
            palette[alg_id] = "tab:gray" #"tab:pink"

    # Sort hue key lexicographically so legend order is deterministic
    sorted_hue_vals = sorted(df[hue_key].unique())
    df[hue_key] = pd.Categorical(df[hue_key], categories=sorted_hue_vals, ordered=True)

    # Truncate x_axis if want
    if x_axis_truncate is not None:
        df = df[df[x_axis_key] <= x_axis_truncate]

    algs_in_data = set(df["alg_id"])

    if fl_sparse_reward_transform:
        df.loc[df["mc_val"] == 0.0, "mc_val"] = pow(0.99, 36)
        df["mc_val"] = -np.log(df["mc_val"]) / np.log(0.99)
    
    if not seperate_plots:
        if x_axis_key is None:
            x_axis_key = "temp_or_bias"
        if x_axis_lab is None:
            x_axis_lab = "Temperature/Bias"
        df["temp_or_bias"] = df["temp"].fillna(df["bias"])
        local_title = title
        if local_title is None:
            local_title = y_axis_lab + " vs " + x_axis_lab
        make_lineplot_df(
            df=df, 
            x_axis_key=x_axis_key, 
            y_axis_key=y_axis_key, 
            title=local_title,
            hue_key=hue_key,
            palette=palette,
            style_key=hue_key,
            # dashes=dashes,
            x_axis_lab=x_axis_lab,
            y_axis_lab=y_axis_lab,
            x_log_scale=True,
            y_scale_transform_forward=y_scale_transform_forward,
            y_scale_transform_inverse=y_scale_transform_inverse,
            legend_lab=legend_lab,
            filename=plot_filename,
            y_axis_range=y_axis_range,
            # markers=markers,
            # markevery=markevery,
            use_legend=use_legend,
            legend_loc=legend_loc,
            alpha=alpha,
            font_scale=font_scale,
            horizontal_lines=horizontal_lines)
        return

    for alg_id in [uct_str]:
        if alg_id not in algs_in_data:
            continue
        local_x_axis_key = x_axis_key if x_axis_key is not None else "bias"
        local_x_axis_lab = x_axis_lab if x_axis_lab is not None else "Bias"
        bias_df = df[df["alg_id"] == alg_id]
        local_title = title
        if local_title is None:
            local_title = y_axis_lab + " vs " + local_x_axis_lab
        filename = plot_filename_frmt_str.format(alg_id=alg_id)
        make_lineplot_df(
            df=bias_df, 
            x_axis_key=local_x_axis_key, 
            y_axis_key=y_axis_key, 
            title=local_title,
            hue_key=hue_key,
            palette=palette,
            style_key=hue_key,
            # dashes=dashes,
            x_axis_lab=local_x_axis_lab,
            y_axis_lab=y_axis_lab,
            x_log_scale=True,
            y_scale_transform_forward=y_scale_transform_forward,
            y_scale_transform_inverse=y_scale_transform_inverse,
            legend_lab=legend_lab,
            filename=filename,
            y_axis_range=y_axis_range,
            # markers=markers,
            # markevery=markevery,
            use_legend=use_legend,
            legend_loc=legend_loc,
            alpha=alpha,
            font_scale=font_scale,
            horizontal_lines=horizontal_lines)
        
    for alg_id in [ments_str,bts_str,dents_str]:
        if alg_id not in algs_in_data:
            continue
        local_x_axis_key = x_axis_key if x_axis_key is not None else "temp"
        local_x_axis_lab = x_axis_lab if x_axis_lab is not None else "Temperature"
        temp_df = df[df["alg_id"] == alg_id]
        local_title = title
        if local_title is None:
            local_title = y_axis_lab + " vs " + local_x_axis_lab
        filename = plot_filename_frmt_str.format(alg_id=alg_id)
        make_lineplot_df(
            df=temp_df, 
            x_axis_key=local_x_axis_key, 
            y_axis_key=y_axis_key, 
            title=local_title,
            hue_key=hue_key,
            palette=palette,
            style_key=hue_key,
            # dashes=dashes,
            x_axis_lab=local_x_axis_lab,
            y_axis_lab=y_axis_lab,
            x_log_scale=True,
            y_scale_transform_forward=y_scale_transform_forward,
            y_scale_transform_inverse=y_scale_transform_inverse,
            legend_lab=legend_lab,
            filename=filename,
            y_axis_range=y_axis_range,
            # markers=markers,
            # markevery=markevery,
            use_legend=use_legend,
            legend_loc=legend_loc,
            alpha=alpha,
            font_scale=font_scale,
            horizontal_lines=horizontal_lines)


    # Call our actual make plot function


    
def negative_log_transform(x):
    # EPS = 1e-10
    # if x < EPS: x = EPS
    return np.log(x)

def y_scale_piecwise_linear_forward(x, min_y=0.0, mid_y=0.65, scaled_y=0.1, max_y=1.0):
    """
    data is min_y -> mid_y -> max_y
    want it to be plotted at min_y -> scaled_y -> max_y

    y = x.copy()
    y[y<0.6] = 0.25 * y[y<0.6]/0.6
    y[y>=0.6] = 0.25 + 0.75 * (y[y>=0.6] - 0.6) / 0.4
    return y
    """
    y = x.copy()
    y[y<mid_y] = min_y + (scaled_y - min_y) * (y[y<mid_y] - min_y) / (mid_y-min_y)
    y[y>=mid_y] = scaled_y + (max_y - scaled_y) * (y[y>=mid_y] - mid_y) / (max_y-mid_y)
    return y
    
def y_scale_piecwise_linear_inverse(x, min_y=0.0, mid_y=0.65, scaled_y=0.1, max_y=1.0):
    """
    data is min_y -> scaled_y -> max_y
    want it to be plotted at min_y -> mid_y -> max_y

    y = x.copy()
    y[y<0.25] = 0.6 * y[y<0.25] / 0.25
    y[y>=0.25] = 0.6 + 0.4 * (y[y>=0.25] - 0.25) / 0.75
    return y

    y = x.copy()
    y[y<scaled_y] = min_y + (mid_y - min_y) * (y[y<scaled_y] - min_y) / (scaled_y-min_y)
    y[y>=scaled_y] = mid_y + (max_y - mid_y) * (y[y>=scaled_y] - scaled_y) / (max_y-scaled_y)
    return y
    """
    return y_scale_piecwise_linear_forward(x, min_y=min_y, mid_y=scaled_y, scaled_y=mid_y, max_y=max_y)

def get_piecwise_linear_forward_transform(min_y,mid_y,scaled_y,max_y):
    return lambda x: y_scale_piecwise_linear_forward(x,min_y,mid_y,scaled_y,max_y)

def get_piecwise_linear_inverse_transform(min_y,mid_y,scaled_y,max_y):
    return lambda x: y_scale_piecwise_linear_inverse(x,min_y,mid_y,scaled_y,max_y)







if __name__ == "__main__":
    if not os.path.exists("plots"):
        os.makedirs("plots")

    # ------------------------------------------------------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------------------------------------------------------
    
    if "010" in sys.argv or "all" in sys.argv or "debug" in sys.argv:
        print("Plotting: ", "010")
        filenames = glob.glob("aux_eval_logs/010_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/010_debug_fl_dense.png",
            # legend_loc="lower left",
        )
    
    if "011" in sys.argv or "all" in sys.argv or "debug" in sys.argv:
        print("Plotting: ", "011")
        filenames = glob.glob("aux_eval_logs/011_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/011_debug_fl_sparse.png",
            # legend_loc="lower left",
        )
    
    if "012" in sys.argv or "all" in sys.argv or "debug" in sys.argv:
        print("Plotting: ", "012")
        filenames = glob.glob("aux_eval_logs/012_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/012_debug_fl_slippy.png",
            # legend_loc="lower left",
        )
    
    if "013" in sys.argv or "all" in sys.argv or "debug" in sys.argv:
        print("Plotting: ", "013")
        filenames = glob.glob("aux_eval_logs/013_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/013_debug_sailing.png",
            # legend_loc="lower left",
        )
    
    if "040" in sys.argv or "all" in sys.argv or "debug" in sys.argv:
        print("Plotting: ", "040")
        filenames = glob.glob("aux_eval_logs/040_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/040_debug_fl_sparse_big.png",
            # legend_loc="lower left",
        )

    if "999" in sys.argv or "all" in sys.argv or "debug" in sys.argv:
        print("Plotting: ", "999")
        filenames = glob.glob("aux_eval_logs/999_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/999_debug.png",
            # legend_loc="lower left",
        )

    # ------------------------------------------------------------------------------------------------------------------
    # FL Dense No Hole (Ch4.1)
    # ------------------------------------------------------------------------------------------------------------------
    if "100" in sys.argv or "all" in sys.argv or "intro" in sys.argv:
        print("Plotting: ", "100")
        # filenames = glob.glob("results/100_supp_dchain_temp_vary_1740880779/**/eval.txt", recursive=True)
        filenames = glob.glob("aux_eval_logs/100_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/100_intro_ssp_gridworld_eval_vs_biastemp.png",
            seperate_plots=False,
            legend_loc="lower left",
            font_scale=1.7,
            title="Reward Along Recommended Path vs Bias/Temperature",
            x_axis_lab="Bias/Temperature",
            y_axis_lab="Reward Along Recommended Path",
        )

    if "101" in sys.argv or "all" in sys.argv or "intro" in sys.argv:
        print("Plotting: ", "101")
        filenames = glob.glob("aux_eval_logs/101_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/101_intro_ssp_gridworld_temp=1_eval_vs_numtrials.png",
            add_dashes=True,
            font_scale=1.7,
            title="Reward Along Recommended Path vs Num Trials",
            x_axis_lab="Num Trials",
            y_axis_lab="Reward Along Recommended Path",
            # legend_loc="lower left",
        )

    # ------------------------------------------------------------------------------------------------------------------
    # DChain/EntropyTrap vs temp plots
    # ------------------------------------------------------------------------------------------------------------------
    if "110" in sys.argv or "all" in sys.argv or "dchain" in sys.argv:
        print("Plotting: ", "110")
        filenames = glob.glob("aux_eval_logs/110_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/110_dchain_10_vs_tempbias.png",
            seperate_plots=False,
            legend_loc="lower left",
        )
        # make_param_sens_plot(
        #     filenames=filenames,
        #     plot_filename_frmt_str="plots/100_dchain_10_vs_tempbias_{alg_id}.png",
        #     seperate_plots=True,
        #     legend_loc="lower left",
        # )

    if "111" in sys.argv or "all" in sys.argv or "dchain" in sys.argv:
        print("Plotting: ", "111")
        filenames = glob.glob("aux_eval_logs/111_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/111_mod_dchain_10_vs_tempbias.png",
            seperate_plots=False,
            legend_loc="lower left",
        )

    if "112" in sys.argv or "all" in sys.argv or "dchain" in sys.argv:
        print("Plotting: ", "112")
        filenames = glob.glob("aux_eval_logs/112_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/112_entropy_trap_10_vs_tempbias_5k_trials.png",
            seperate_plots=False,
            legend_loc="lower left",
        )

    if "113" in sys.argv or "all" in sys.argv or "dchain" in sys.argv:
        print("Plotting: ", "113")
        filenames = glob.glob("aux_eval_logs/113_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/113_entropy_trap_15_vs_tempbias_500k_trials.png",
            seperate_plots=False,
            legend_loc="lower left",
        )


    # # ------------------------------------------------------------------------------------------------------------------
    # # Eval - Frozen Lake Sparse
    # # ------------------------------------------------------------------------------------------------------------------
    # if "400" in sys.argv or "all" in sys.argv or "fl_sparse" in sys.argv:
    #     print("Plotting: ", "400")
    #     filenames = glob.glob("aux_eval_logs/400_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/400_FL_sparse_8x8.png",
    #         # legend_loc="lower left",
    #     )

    # if "401" in sys.argv or "all" in sys.argv or "fl_sparse" in sys.argv:
    #     print("Plotting: ", "401")
    #     filenames = glob.glob("aux_eval_logs/401_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/401_FL_sparse_8x12.png",
    #         # legend_loc="lower left",
    #     )

    # if "402" in sys.argv or "all" in sys.argv or "fl_sparse" in sys.argv:
    #     print("Plotting: ", "402")
    #     filenames = glob.glob("aux_eval_logs/402_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/402_FL_sparse_12x12.png",
    #         # legend_loc="lower left",
    #     )


    # # ------------------------------------------------------------------------------------------------------------------
    # # Eval - Frozen Lake Sparse Big
    # # ------------------------------------------------------------------------------------------------------------------
    # if "410" in sys.argv or "all" in sys.argv or "fl_sparse_big" in sys.argv:
    #     print("Plotting: ", "410")
    #     filenames = glob.glob("aux_eval_logs/410_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/410_FL_sparse_big_12x12+graph.png",
    #         # legend_loc="lower left",
    #     )

    # if "411" in sys.argv or "all" in sys.argv or "fl_sparse_big" in sys.argv:
    #     print("Plotting: ", "411")
    #     filenames = glob.glob("aux_eval_logs/411_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/411_FL_sparse_big_8x16+graph.png",
    #         # legend_loc="lower left",
    #     )

    # if "412" in sys.argv or "all" in sys.argv or "fl_sparse_big" in sys.argv:
    #     print("Plotting: ", "412")
    #     filenames = glob.glob("aux_eval_logs/412_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/412_FL_sparse_big_16x16+graph.png",
    #         # legend_loc="lower left",
    #     )


    # ------------------------------------------------------------------------------------------------------------------
    # Eval - Frozen Lake Sparse Big + heuristic
    # ------------------------------------------------------------------------------------------------------------------
    if "420" in sys.argv or "all" in sys.argv or "fl_sparse_big_heuristic" in sys.argv:
        print("Plotting: ", "420")
        filenames = glob.glob("aux_eval_logs/420_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/420_FL_sparse_8x8+graph+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/420_FL_sparse_8x8_PATH+graph+heuristic_weight_local=1.png",
            # legend_loc="lower left",
            path_len_plot=True,
            max_path_len=100,
        )

    # if "421" in sys.argv or "all" in sys.argv or "fl_sparse_big_heuristic" in sys.argv:
    #     print("Plotting: ", "421")
    #     filenames = glob.glob("aux_eval_logs/421_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/421_FL_sparse_8x12+graph+heuristic_weight_local=1.png",
    #         # legend_loc="lower left",
    #     )

    if "422" in sys.argv or "all" in sys.argv or "fl_sparse_big_heuristic" in sys.argv:
        print("Plotting: ", "422")
        filenames = glob.glob("aux_eval_logs/422_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/422_FL_sparse_8x16+graph+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/422_FL_sparse_8x16_PATH+graph+heuristic_weight_local=1.png",
            # legend_loc="lower left",
            path_len_plot=True,
            max_path_len=100,
        )

    # if "423" in sys.argv or "all" in sys.argv or "fl_sparse_big_heuristic" in sys.argv:
    #     print("Plotting: ", "423")
    #     filenames = glob.glob("aux_eval_logs/423_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/423_FL_sparse_8x20+graph+heuristic_weight_local=1.png",
    #         # legend_loc="lower left",
    #     )
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/423_FL_sparse_8x20_PATH+graph+heuristic_weight_local=1.png",
    #         # legend_loc="lower left",
    #         path_len_plot=True,
    #     )

    if "424" in sys.argv or "all" in sys.argv or "fl_sparse_big_heuristic" in sys.argv:
        print("Plotting: ", "424")
        filenames = glob.glob("aux_eval_logs/424_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/424_FL_sparse_8x24+graph+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/424_FL_sparse_8x24_PATH+graph+heuristic_weight_local=1.png",
            # legend_loc="lower left",
            path_len_plot=True,
            max_path_len=100,
        )

    if "425" in sys.argv or "all" in sys.argv or "fl_sparse_big_heuristic" in sys.argv:
        print("Plotting: ", "425")
        filenames = glob.glob("aux_eval_logs/425_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/425_FL_sparse_8x32+graph+heuristic_weight_local=1.png",
            # legend_loc="lower left",
            max_path_len=150,
        )
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/425_FL_sparse_8x32_PATH+graph+heuristic_weight_local=1.png",
            # legend_loc="lower left",
            path_len_plot=True,
            max_path_len=150,
        )

    if "425a" in sys.argv or "all" in sys.argv or "fl_sparse_big_heuristic" in sys.argv:
        print("Plotting: ", "425a")
        filenames = glob.glob("aux_eval_logs/425a_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/425a_FL_sparse_8x32+graph+heuristic_weight_local=1.png",
            # legend_loc="lower left",
            max_path_len=150,
        )
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/425a_FL_sparse_8x32_PATH+graph+heuristic_weight_local=1.png",
            # legend_loc="lower left",
            path_len_plot=True,
            max_path_len=150,
        )


    # # ------------------------------------------------------------------------------------------------------------------
    # # Eval - Frozen Lake Slippy
    # # ------------------------------------------------------------------------------------------------------------------
    # if "430" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
    #     print("Plotting: ", "430")
    #     filenames = glob.glob("aux_eval_logs/430_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/430_FL_slippy_4x4.png",
    #         # legend_loc="lower left",
    #     )

    # if "431" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
    #     print("Plotting: ", "431")
    #     filenames = glob.glob("aux_eval_logs/431_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/431_FL_slippy_5x5.png",
    #         # legend_loc="lower left",
    #     )

    # if "432" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
    #     print("Plotting: ", "432")
    #     filenames = glob.glob("aux_eval_logs/432_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/432_FL_slippy_6x6.png",
    #         # legend_loc="lower left",
    #     )

    # if "433" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
    #     print("Plotting: ", "433")
    #     filenames = glob.glob("aux_eval_logs/433_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/433_FL_slippy_8x8.png",
    #         # legend_loc="lower left",
    #     )

    # if "434" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
    #     print("Plotting: ", "434")
    #     filenames = glob.glob("aux_eval_logs/434_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/434_FL_slippy_4x8.png",
    #         # legend_loc="lower left",
    #     )

    # if "435" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
    #     print("Plotting: ", "435")
    #     filenames = glob.glob("aux_eval_logs/435_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/435_FL_slippy_4x12.png",
    #         # legend_loc="lower left",
    #     )

    # if "436" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
    #     print("Plotting: ", "436")
    #     filenames = glob.glob("aux_eval_logs/436_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/436_FL_slippy_8x12.png",
    #         # legend_loc="lower left",
    #     )

    # if "437" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
    #     print("Plotting: ", "437")
    #     filenames = glob.glob("aux_eval_logs/437_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/437_FL_slippy_12x12.png",
    #         # legend_loc="lower left",
    #     )


    # # ------------------------------------------------------------------------------------------------------------------
    # # Eval - Sailing North
    # # ------------------------------------------------------------------------------------------------------------------
    # if "440" in sys.argv or "all" in sys.argv or "sailing_north" in sys.argv:
    #     print("Plotting: ", "440")
    #     filenames = glob.glob("aux_eval_logs/440_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/440_sailing_north_8x8.png",
    #         # legend_loc="lower left",
    #     )

    # if "441" in sys.argv or "all" in sys.argv or "sailing_north" in sys.argv:
    #     print("Plotting: ", "441")
    #     filenames = glob.glob("aux_eval_logs/441_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/441_sailing_north_8x16.png",
    #         # legend_loc="lower left",
    #     )

    # if "442" in sys.argv or "all" in sys.argv or "sailing_north" in sys.argv:
    #     print("Plotting: ", "442")
    #     filenames = glob.glob("aux_eval_logs/442_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/442_sailing_north_16x16.png",
    #         # legend_loc="lower left",
    #     )


    # # ------------------------------------------------------------------------------------------------------------------
    # # Eval - Sailing South East
    # # ------------------------------------------------------------------------------------------------------------------
    # if "450" in sys.argv or "all" in sys.argv or "sailing_south_east" in sys.argv:
    #     print("Plotting: ", "450")
    #     filenames = glob.glob("aux_eval_logs/450_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/450_sailing_south_east_8x8.png",
    #         # legend_loc="lower left",
    #     )

    # if "451" in sys.argv or "all" in sys.argv or "sailing_south_east" in sys.argv:
    #     print("Plotting: ", "441")
    #     filenames = glob.glob("aux_eval_logs/451_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/451_sailing_south_east_8x16.png",
    #         # legend_loc="lower left",
    #     )

    # if "452" in sys.argv or "all" in sys.argv or "sailing_south_east" in sys.argv:
    #     print("Plotting: ", "452")
    #     filenames = glob.glob("aux_eval_logs/452_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/452_sailing_south_east_16x16.png",
    #         # legend_loc="lower left",
    #     )


    # ------------------------------------------------------------------------------------------------------------------
    # Eval - Frozen Lake Sparse + mcts mode
    # ------------------------------------------------------------------------------------------------------------------
    if "460" in sys.argv or "all" in sys.argv or "fl_sparse_mcts" in sys.argv:
        print("Plotting: ", "460")
        filenames = glob.glob("aux_eval_logs/460_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/460_FL_sparse_8x8+mcts.png",
            # legend_loc="lower left",
        )
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/460_FL_sparse_8x8_PATH+mcts.png",
            # legend_loc="lower left",
            path_len_plot=True,
            max_path_len=100,
        )

    if "462" in sys.argv or "all" in sys.argv or "fl_sparse_mcts" in sys.argv:
        print("Plotting: ", "462")
        filenames = glob.glob("aux_eval_logs/462_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/462_FL_sparse_8x16+mcts.png",
            # legend_loc="lower left",
        )
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/462_FL_sparse_8x16_PATH+mcts.png",
            # legend_loc="lower left",
            path_len_plot=True,
            max_path_len=100,
        )

    if "464" in sys.argv or "all" in sys.argv or "fl_sparse_mcts" in sys.argv:
        print("Plotting: ", "464")
        filenames = glob.glob("aux_eval_logs/464_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/464_FL_sparse_8x24+mcts.png",
            # legend_loc="lower left",
        )
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/464_FL_sparse_8x24_PATH+mcts.png",
            # legend_loc="lower left",
            path_len_plot=True,
            max_path_len=100,
        )

    if "465" in sys.argv or "all" in sys.argv or "fl_sparse_mcts" in sys.argv:
        print("Plotting: ", "465")
        filenames = glob.glob("aux_eval_logs/465_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/465_FL_sparse_8x32+mcts.png",
            # legend_loc="lower left",
        )
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/465_FL_sparse_8x32_PATH+mcts.png",
            # legend_loc="lower left",
            path_len_plot=True,
            max_path_len=150,
        )


    # # ------------------------------------------------------------------------------------------------------------------
    # # Eval - Frozen Lake Sparse + mcts mode
    # # ------------------------------------------------------------------------------------------------------------------
    # if "470" in sys.argv or "all" in sys.argv or "fl_sparse_mcts_graph" in sys.argv:
    #     print("Plotting: ", "470")
    #     filenames = glob.glob("aux_eval_logs/470_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/470_FL_sparse_8x8+mcts+graph.png",
    #         # legend_loc="lower left",
    #     )

    # if "471" in sys.argv or "all" in sys.argv or "fl_sparse_mcts_graph" in sys.argv:
    #     print("Plotting: ", "471")
    #     filenames = glob.glob("aux_eval_logs/471_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/471_FL_sparse_8x12+mcts+graph.png",
    #         # legend_loc="lower left",
    #     )


    # ------------------------------------------------------------------------------------------------------------------
    # Eval - Frozen Lake Slippy + heuristic weight local = 1
    # ------------------------------------------------------------------------------------------------------------------
    if "480" in sys.argv or "all" in sys.argv or "fl_slippy_heuristic" in sys.argv:
        print("Plotting: ", "480")
        filenames = glob.glob("aux_eval_logs/480_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/480_FL_slippy_4x4+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )

    if "481" in sys.argv or "all" in sys.argv or "fl_slippy_heuristic" in sys.argv:
        print("Plotting: ", "481")
        filenames = glob.glob("aux_eval_logs/481_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/481_FL_slippy_4x8+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )

    if "482" in sys.argv or "all" in sys.argv or "fl_slippy_heuristic" in sys.argv:
        print("Plotting: ", "482")
        filenames = glob.glob("aux_eval_logs/482_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/482_FL_slippy_4x12+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )

    if "483" in sys.argv or "all" in sys.argv or "fl_slippy_heuristic" in sys.argv:
        print("Plotting: ", "483")
        filenames = glob.glob("aux_eval_logs/483_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/483_FL_slippy_4x16+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )

    # if "484" in sys.argv or "all" in sys.argv or "fl_slippy_heuristic" in sys.argv:
    #     print("Plotting: ", "484")
    #     filenames = glob.glob("aux_eval_logs/484_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/484_FL_slippy_4x16+heuristic_weight_local=1.png",
    #         # legend_loc="lower left",
    #     )

    # if "485" in sys.argv or "all" in sys.argv or "fl_slippy_heuristic" in sys.argv:
    #     print("Plotting: ", "485")
    #     filenames = glob.glob("aux_eval_logs/485_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/485_FL_slippy_8x20+heuristic_weight_local=1.png",
    #         # legend_loc="lower left",
    #     )

    # if "486" in sys.argv or "all" in sys.argv or "fl_slippy_heuristic" in sys.argv:
    #     print("Plotting: ", "486")
    #     filenames = glob.glob("aux_eval_logs/486_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/486_FL_slippy_8x12+heuristic_weight_local=1.png",
    #         # legend_loc="lower left",
    #     )

    # if "487" in sys.argv or "all" in sys.argv or "fl_slippy_heuristic" in sys.argv:
    #     print("Plotting: ", "487")
    #     filenames = glob.glob("aux_eval_logs/487_*/**/eval_log.txt", recursive=True)
    #     make_eval_plot(
    #         filenames=filenames,
    #         plot_filename="plots/487_FL_slippy_12x12+heuristic_weight_local=1.png",
    #         # legend_loc="lower left",
    #     )


    # ------------------------------------------------------------------------------------------------------------------
    # Eval - Sailing North + heuristic weight local = 1
    # ------------------------------------------------------------------------------------------------------------------
    if "490" in sys.argv or "all" in sys.argv or "sailing_north_heuristic" in sys.argv:
        print("Plotting: ", "490")
        filenames = glob.glob("aux_eval_logs/490_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/490_sailing_north_8x8+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )

    if "491" in sys.argv or "all" in sys.argv or "sailing_north_heuristic" in sys.argv:
        print("Plotting: ", "491")
        filenames = glob.glob("aux_eval_logs/491_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/491_sailing_north_8x16+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )

    if "492" in sys.argv or "all" in sys.argv or "sailing_north_heuristic" in sys.argv:
        print("Plotting: ", "492")
        filenames = glob.glob("aux_eval_logs/492_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/492_sailing_north_16x16+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Eval - Sailing South East
    # ------------------------------------------------------------------------------------------------------------------
    if "500" in sys.argv or "all" in sys.argv or "sailing_south_east_heuristic" in sys.argv:
        print("Plotting: ", "500")
        filenames = glob.glob("aux_eval_logs/500_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/500_sailing_south_east_8x8+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )

    if "501" in sys.argv or "all" in sys.argv or "sailing_south_east_heuristic" in sys.argv:
        print("Plotting: ", "501")
        filenames = glob.glob("aux_eval_logs/501_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/501_sailing_south_east_8x16+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )

    if "502" in sys.argv or "all" in sys.argv or "sailing_south_east_heuristic" in sys.argv:
        print("Plotting: ", "502")
        filenames = glob.glob("aux_eval_logs/502_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/502_sailing_south_east_16x16+heuristic_weight_local=1.png",
            # legend_loc="lower left",
        )


    # # ------------------------------------------------------------------------------------------------------------------
    # # Hpopt/aux - temperature vs performance - fl dense
    # # ------------------------------------------------------------------------------------------------------------------
    # if "600" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
    #     print("Plotting: ", "600")
    #     filenames = glob.glob("aux_eval_logs/600_*/**/eval_log.txt", recursive=True)
    #     make_param_sens_plot(
    #         filenames=filenames,
    #         plot_filename="plots/600_fl_dense_8x8_eps=0.0.png",
    #         legend_loc="lower left",
    #         seperate_plots=False,
    #         horizontal_lines=[-15],
    #     )

    # if "601" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
    #     print("Plotting: ", "601")
    #     filenames = glob.glob("aux_eval_logs/601_*/**/eval_log.txt", recursive=True)
    #     make_param_sens_plot(
    #         filenames=filenames,
    #         plot_filename="plots/601_fl_dense_8x12_eps=0.0.png",
    #         legend_loc="lower left",
    #         seperate_plots=False,
    #         horizontal_lines=[-19],
    #     )

    # if "602" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
    #     print("Plotting: ", "602")
    #     filenames = glob.glob("aux_eval_logs/602_*/**/eval_log.txt", recursive=True)
    #     make_param_sens_plot(
    #         filenames=filenames,
    #         plot_filename="plots/602_fl_dense_12x12_eps=0.0.png",
    #         legend_loc="lower left",
    #         seperate_plots=False,
    #         horizontal_lines=[-23],
    #     )


    # ------------------------------------------------------------------------------------------------------------------
    # Hpopt/aux - temperature vs performance - fl sparse
    # ------------------------------------------------------------------------------------------------------------------
    if "700" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "700")
        filenames = glob.glob("aux_eval_logs/700_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/700_fl_sparse_8x8_eps=0.0.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.99**15],
        )

    if "701" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "701")
        filenames = glob.glob("aux_eval_logs/701_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/701_fl_sparse_8x8_eps=0.0_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.99**19],
        )

    if "710" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "710")
        filenames = glob.glob("aux_eval_logs/710_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/710_fl_sparse_8x8_eps=1.0.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.99**15],
        )

    if "711" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "711")
        filenames = glob.glob("aux_eval_logs/711_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/711_fl_sparse_8x8_eps=1.0_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.99**19],
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Hpopt/aux - temperature vs performance - fl sparse
    # ------------------------------------------------------------------------------------------------------------------

    if "750" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "750")
        filenames = glob.glob("aux_eval_logs/750_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/750_fl_sparse_8x16_eps=0.0_heuristic_weight_local=1.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.99**23],
        )

    if "751" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "751")
        filenames = glob.glob("aux_eval_logs/751_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/751_fl_sparse_8x16_eps=0.0_heuristic_weight_local=1_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.99**23],
        )

    if "760" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "760")
        filenames = glob.glob("aux_eval_logs/760_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/760_fl_sparse_8x16_eps=1.0_heuristic_weight_local=1.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.99**23],
        )

    if "761" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "761")
        filenames = glob.glob("aux_eval_logs/761_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/761_fl_sparse_8x16_eps=1.0_heuristic_weight_local=1_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.99**23],
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Hpopt/aux - temperature vs performance - fl slippy
    # ------------------------------------------------------------------------------------------------------------------
    if "800" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "800")
        filenames = glob.glob("aux_eval_logs/800_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/800_fl_slippy_4x4_eps=0.0.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.7442],
        )

    if "801" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "801")
        filenames = glob.glob("aux_eval_logs/801_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/801_fl_slippy_4x4_eps=0.0_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.7442],
        )

    if "810" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "810")
        filenames = glob.glob("aux_eval_logs/810_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/810_fl_slippy_4x4_eps=1.0.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.7442],
        )

    if "811" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "811")
        filenames = glob.glob("aux_eval_logs/811_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/811_fl_slippy_4x4_eps=1.0_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.7442],
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Hpopt/aux - temperature vs performance - fl slippy
    # ------------------------------------------------------------------------------------------------------------------

    if "850" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "850")
        filenames = glob.glob("aux_eval_logs/850_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/850_fl_slippy_6x6_eps=0.0_heuristic_weight_local=1.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.8165],
        )

    if "851" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "851")
        filenames = glob.glob("aux_eval_logs/851_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/851_fl_slippy_6x6_eps=0.0_heuristic_weight_local=1_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.8165],
        )

    if "860" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "860")
        filenames = glob.glob("aux_eval_logs/860_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/860_fl_slippy_6x6_eps=1.0_heuristic_weight_local=1.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.8165],
        )

    if "861" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "861")
        filenames = glob.glob("aux_eval_logs/861_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/861_fl_slippy_6x6_eps=1.0_heuristic_weight_local=1_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
            horizontal_lines=[0.8165],
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Hpopt/aux - temperature vs performance - sailing
    # ------------------------------------------------------------------------------------------------------------------
    if "900" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "900")
        filenames = glob.glob("aux_eval_logs/900_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/900_sailing_north_8x8_eps=0.0.png",
            legend_loc="lower left",
            seperate_plots=False,
        )

    if "901" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "901")
        filenames = glob.glob("aux_eval_logs/901_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/901_sailing_north_8x8_eps=0.0_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
        )

    if "910" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "910")
        filenames = glob.glob("aux_eval_logs/910_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/910_sailing_north_8x8_eps=1.0.png",
            legend_loc="lower left",
            seperate_plots=False,
        )

    if "911" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "911")
        filenames = glob.glob("aux_eval_logs/911_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/911_sailing_north_8x8_eps=1.0_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Hpopt/aux - temperature vs performance - sailing
    # ------------------------------------------------------------------------------------------------------------------
    if "950" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "950")
        filenames = glob.glob("aux_eval_logs/950_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/950_sailing_south_east_8x16_eps=0.0.png",
            legend_loc="lower left",
            seperate_plots=False,
        )

    if "951" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "951")
        filenames = glob.glob("aux_eval_logs/951_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/951_sailing_south_east_8x16_eps=0.0_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
        )

    if "960" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "960")
        filenames = glob.glob("aux_eval_logs/960_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/960_sailing_south_east_8x16_eps=1.0.png",
            legend_loc="lower left",
            seperate_plots=False,
        )

    if "961" in sys.argv or "all" in sys.argv or "temp" in sys.argv:
        print("Plotting: ", "961")
        filenames = glob.glob("aux_eval_logs/961_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename="plots/961_sailing_south_east_8x16_eps=1.0_decay_temp=0.01.png",
            legend_loc="lower left",
            seperate_plots=False,
        )


                           
                 






    
    ## Example of plotting per algorithm and not combining into one plot
    # if "111" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
    #     filenames = glob.glob("aux_eval_logs/111_uct_on_fl_sparse_len_1742131975/**/eval_log.txt", recursive=True)
    #     make_param_sens_plot(
    #         filenames=filenames,
    #         plot_filename_frmt_str="plots/111={alg_id}.png",
    #     )