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
    TODO: this file needs updating w.r.t. current log files (manual) 

    Reads the eval file from 'filename'
    Appends results in this eval file to the arrays: alg_ids, replicates, search_times, num_trialss, 

    Manually reads the first two lines to get params that the algorithm was run with (usually the algorithm name + params)
    At the moment we just care about the algorithm name

    Then uses pandas to read 2nd half, which is a csv file
    And adds an additional collumn with the algorithm name
    """

    alg_id = None
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

        # At the moment we just care about which algorithm this was, rather than the params it was run with
        for param_id, val in zip(param_ids,param_vals):
            if param_id == "alg_id":
                alg_id = val
    
    df = pd.read_csv(filepath_or_buffer=filename, header=12, index_col=False, skip_blank_lines=False)
    df["alg_id"] = alg_id
    df["num_trials"] /= num_trials_scale

    df = df.rename(columns={
        "alg_id": "alg_id",

        "run_idx": "run_idx",
        "search_budget_consumed": "search_time",
        "num_trials": "num_trials",
        "num_backups": "num_backups",

        "ctx_mean": "utility",
        "ctx_std_dev": "utility_std",
        "reweighted_ctx_mean": "reweighted_utility",
        "reweighted_ctx_std_dev": "reweighted_utility_std",
        "normalised_ctx_mean": "norm_utility",
        "normalised_ctx_std_dev": "norm_utility_std",

        "hypervolume": "hypervolume",
        "additive_eps_metric": "additive_eps_metric",
        "sparsity_metric": "sparsity_metric",
        "normalised_hypervolume": "normalised_hypervolume",
        "normalised_additive_eps_metric": "normalised_additive_eps_metric",
        "normalised_sparsity_metric": "normalised_sparsity_metric",
    }, errors="raise")
    
    return df

def get_env_id(filename):
    """
    Environment name is part of the filename
    filenames are of the form results/<expr_id>/<env_id>/<alg_id>/<alg_param_1>/.../<alg_param_N>/eval.txt
    """
    return filename.split("/")[2]

def is_tree_env(filename):
    """
    Checks if a filename corresponds to a run on trees
    filenames are of the form results/<expr_id>/<env_id>/<alg_id>/<alg_param_1>/.../<alg_param_N>/eval.txt
    on tree envs the params are delimited with 
    """
    return "|" in filename

def get_tree_env_params(filename):
    """
    filenames are of the form results/<expr_id>/<env_id>/<alg_id>/<alg_param_1>/.../<alg_param_N>/eval.txt
    In this case, the <env_id> is of the form <tree_env_id>|<num_steps>|<reward_dim>|<num_actions>
    Returns the 4 parts of the env_id
    """
    env_id = get_env_id(filename)
    tree_env_id, num_steps_str, reward_dim_str, num_actions_str = env_id.split("|")
    return tree_env_id, int(num_steps_str), int(reward_dim_str), int(num_actions_str)


def read_eval_files_to_df(filenames,num_trials_scale):
    """
    Reads dataframes for each eval file in 'filenames' and concatinates them all into one big dataframe

    Adds the env_id for each environment to each dataframe, and if the environment was a tree, then add additional 
    columns for the params of the environment
    """
    dfs = []
    for filename in filenames:
        df = read_eval_file_to_df(filename,num_trials_scale)
        env_id = None
        if is_tree_env(filename):
            env_id, num_steps, reward_dim, num_actions = get_tree_env_params(filename)
            df["num_steps"] = num_steps
            df["reward_dim"] = reward_dim
            df["num_actions"] = num_actions
        else:
            env_id = get_env_id(filename)
        df["env_id"] = env_id

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
        x_axis_key = "search_time"
    if x_axis_lab is None:
        x_axis_lab = "Search Time"
    if y_axis_key is None:
        y_axis_key = "utility"
    if y_axis_lab is None:
        y_axis_lab = "Expected Utility Metric"
    if title is None:
        title = y_axis_lab + " vs " + x_axis_lab
    if legend_lab is None and use_legend:
        legend_lab = "Algorithm"

    # Update x_axis lable if applying scaling
    if num_trials_scale > 1:
        x_axis_lab += " (x{scale})".format(scale=num_trials_scale)

    # Read in data + make algorithm names more pretty
    chvi_str = "CHVI"
    chvi_ordered_str = "CHVI(t=0->H)"
    chvi_reversed_str = "CHVI(t=H->0)"
    czt_str = "CZT"
    czt_doubling_str = "CZT(Doubling)"
    ch_uct_str = "CH-UCT"
    ch_czt_str = "CH-CZT"
    ch_czt_doubling_str = "CH-CZT(Doubling)"
    ch_bts_str = "CH-BTS"
    ch_hvuct_str = "CH-HVUCT"
    ch_pareto_str = "CH-PARETO"
    ch_cheby_str = "CH-CHEBY"
    ch_standard_cheby_str = "CH-CHEBY(Standard)"
    sm_bts_str = "SM-BTS"
    sm_dents_str = "SM-DENTS"


    df = read_eval_files_to_df(filenames, num_trials_scale)
    df["alg_id"] = df["alg_id"].map({
        "chvi": chvi_str,
        "chvi_ordered": chvi_ordered_str,
        "chvi_reversed": chvi_reversed_str,
        "czt": czt_str,
        "czt_doubling": czt_doubling_str,
        "ch_uct": ch_uct_str,
        "ch_czt": ch_czt_str,
        "ch_czt_doubling": ch_czt_doubling_str,
        "ch_bts": ch_bts_str,
        "ch_hvuct": ch_hvuct_str,
        "ch_pareto": ch_pareto_str,
        "ch_cheby": ch_cheby_str,
        "ch_standard_cheby": ch_standard_cheby_str,
        "sm_bts": sm_bts_str,
        "sm_dents": sm_dents_str,
    })

    # Get the set of alg ids working with
    alg_id_set = set(df["alg_id"])
    
    # Define line styles - (colour) palette
    # N.B. palette can be a colourmap: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
    # Currently using dict for mapping using colours from: https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
    palette = {}
    for alg_id in alg_id_set:
        if chvi_str in alg_id:
            palette[alg_id] = "#7f7f7f"  # tab:gray
        if chvi_ordered_str in alg_id:
            palette[alg_id] = "#9467bd"  # tab:purple
        if chvi_reversed_str in alg_id:
            palette[alg_id] = "#e377c2"  # tab:pink

        if czt_str in alg_id:
            palette[alg_id] = "#2ca02c"  # tab:green
        if czt_doubling_str in alg_id:
            palette[alg_id] = "#d62728"  # tab:red
        if ch_czt_str in alg_id:
            palette[alg_id] = "#98df8a"  # light green
        if ch_czt_doubling_str in alg_id:
            palette[alg_id] = "#ff9896"  # light red

        
        if ch_hvuct_str in alg_id:
            palette[alg_id] = "#c5b0d5"  # light purple
        if ch_pareto_str in alg_id:
            palette[alg_id] = "#c49c94"  # light brown
        if ch_cheby_str in alg_id:
            palette[alg_id] = "#f7b6d2"  # light pink
        if ch_standard_cheby_str in alg_id:
            palette[alg_id] = "#dbdb8d"  # light olive
            
        if ch_uct_str in alg_id:
            palette[alg_id] = "#ffbb78"  # light orange
            

        if ch_bts_str in alg_id:
            palette[alg_id] = "#aec7e8"  # light blue
        if sm_bts_str in alg_id:
            palette[alg_id] = "#1f77b4"  # tab:blue
        if sm_dents_str in alg_id:
            palette[alg_id] = "#ff7f0e"  # tab:orange

    # Currently unused tab20 colours
    # palette[alg_id] = "#8c564b"  # tab:brown
    # palette[alg_id] = "#c7c7c7"  # light gray
    # palette[alg_id] = "#bcbd22"  # tab:olive
    # palette[alg_id] = "#17becf"  # tab:cyan
    # palette[alg_id] = "#9edae5"  # light cyan

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
    
def make_num_trials_plot(
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
    num_trials_scale=1):
    """
    Essentially an overload for make_eum_plot, but plotting the number of trials instead
    """
    # Default (differing) params
    if y_axis_key is None:
        y_axis_key = "num_trials"
    if y_axis_lab is None:
        y_axis_lab = "Num Trials"

    # Forward function call
    make_eum_plot(
        filenames=filenames,
        plot_filename=plot_filename,
        hue_key=hue_key,
        title=title,
        x_axis_key=x_axis_key,
        x_axis_lab=x_axis_lab,
        y_axis_key=y_axis_key,
        y_axis_lab=y_axis_lab,
        legend_lab=legend_lab,
        x_axis_truncate=x_axis_truncate,
        y_scale_transform_forward=y_scale_transform_forward,
        y_scale_transform_inverse=y_scale_transform_inverse,
        y_axis_range=y_axis_range,
        add_markers=add_markers,
        markevery=markevery,
        use_legend=use_legend,
        alpha=alpha,
        num_trials_scale=num_trials_scale,
    )

def make_eum_scalability_plot(
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
    num_trials_scale=1):
    """
    Makes an scalability plot using the data in the given filenames
    Adapted from make_eum_plot (changes marked with SCALE_DIFF comment)
    Takes results from a bunch of runs over varying environments and plots performance scaling
    Can make a non eum plot by specifying y_axis_key
    """

    # SCALE_DIFF: Default params
    if hue_key is None:
        hue_key = "alg_id"
    if x_axis_key is None:
        raise Exception("Running make_eum_scalability_plot without providing x_axis_key argument")
    if x_axis_lab is None:
        raise Exception("Running make_eum_scalability_plot without providing x_axis_lab argument")
    if y_axis_key is None:
        y_axis_key = "utility"
    if y_axis_lab is None:
        y_axis_lab = "Expected Utility Metric"
    if title is None:
        title = y_axis_lab + " vs " + x_axis_lab
    if legend_lab is None and use_legend:
        legend_lab = "Algorithm"

    # Update x_axis lable if applying scaling
    if num_trials_scale > 1:
        x_axis_lab += " (x{scale})".format(scale=num_trials_scale)

    # Read in data + make algorithm names more pretty
    czt_str = "CZT"
    chmcst_str = "CHMCTS"
    smbts_str = "SM-BTS"
    smdents_str = "SM-DENTS"

    df = read_eval_files_to_df(filenames, num_trials_scale)
    df["alg_id"] = df["alg_id"].map({
        "czt": czt_str,
        "chmcts": chmcst_str,
        "smbts": smbts_str,
        "smdents": smdents_str,
    })

    # SCALE_DIFF: only care about the final results after the search (assumes all experiments run for same search time)
    experiment_search_time = df["search_time"].max()
    df = df[df["search_time"] == experiment_search_time]

    # Get the set of alg ids working with
    alg_id_set = set(df["alg_id"])
    
    # Define line styles - (colour) palette
    # N.B. palette can be a colourmap: https://matplotlib.org/stable/api/_as_gen/matplotlib.colors.Colormap.html#matplotlib.colors.Colormap
    # Currently using dict for mapping using colours from: https://seaborn.pydata.org/generated/seaborn.color_palette.html#seaborn.color_palette
    palette = {}
    for alg_id in alg_id_set:
        if czt_str in alg_id:
            palette[alg_id] = "tab:green"
        if smdents_str in alg_id:
            palette[alg_id] = "tab:blue"
        if smbts_str in alg_id:
            palette[alg_id] = "tab:orange"
        if chmcst_str in alg_id:
            palette[alg_id] = "tab:purple"
        # other colours I used
        # palette[alg_id] = "tab:gray"
        # palette[alg_id] = "tab:red"
        # palette[alg_id] = "tab:brown"
        # palette[alg_id] = "tab:grey"

    # Define line styles - dashes (currently unused, but dont want del setup)
    # "Dashes are specified as in matplotlib: a tuple of (segment, gap) lengths, or an empty string to draw a solid line."
    dashes = {}
    for alg_id in alg_id_set:
        dashes[alg_id] = ""
        # dashes[alg_id] = (4,2)

    # Define line styles - markers (currently unused, but dont want del setup)
    # N.B. see following for valid values: https://matplotlib.org/stable/api/markers_api.html
    markers = None
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

    # Concatenate x_axis if want
    if x_axis_truncate is not None:
        df = df[df[x_axis_key] <= x_axis_truncate]

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
        alpha=alpha)

def make_num_trials_scalability_plot(
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
    num_trials_scale=1):
    """
    Essentially an overload for make_eum_scalability_plot, but plotting the number of trials instead
    """
    # Default (differing) params
    if y_axis_key is None:
        y_axis_key = "num_trials"
    if y_axis_lab is None:
        y_axis_lab = "Num Trials"

    # Forward function call
    make_eum_scalability_plot(
        filenames=filenames,
        plot_filename=plot_filename,
        hue_key=hue_key,
        title=title,
        x_axis_key=x_axis_key,
        x_axis_lab=x_axis_lab,
        y_axis_key=y_axis_key,
        y_axis_lab=y_axis_lab,
        legend_lab=legend_lab,
        x_axis_truncate=x_axis_truncate,
        y_scale_transform_forward=y_scale_transform_forward,
        y_scale_transform_inverse=y_scale_transform_inverse,
        y_axis_range=y_axis_range,
        add_markers=add_markers,
        markevery=markevery,
        use_legend=use_legend,
        alpha=alpha,
        num_trials_scale=num_trials_scale,
    )


    
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






def make_many_eval_plots(filenames, fname_base):
    make_eval_plot(
        filenames=filenames,
        plot_filename=f"mo_plots/{fname_base}_0_eum.png",
        # legend_loc="lower left",
    )
    make_eval_plot(
        filenames=filenames,
        plot_filename=f"mo_plots/{fname_base}_1_normalised_utility.png",
        y_axis_key="norm_utility",
        y_axis_lab="Normalised Expected Utility Metric", 
        # legend_loc="lower left",
    )
    make_eval_plot(
        filenames=filenames,
        plot_filename=f"mo_plots/{fname_base}_2_hypervolume.png",
        y_axis_key="hypervolume",
        y_axis_lab="Hypervolume", 
        # legend_loc="lower left",
    )
    make_eval_plot(
        filenames=filenames,
        plot_filename=f"mo_plots/{fname_base}_3_normalised_hypervolume.png",
        y_axis_key="normalised_hypervolume",
        y_axis_lab="Normalised Hypervolume", 
        # legend_loc="lower left",
    )
    make_eval_plot(
        filenames=filenames,
        plot_filename=f"mo_plots/{fname_base}_4_num_trials.png",
        y_axis_key="num_trials",
        y_axis_lab="Num Trials", 
        # legend_loc="lower left",
    )
    make_eval_plot(
        filenames=filenames,
        plot_filename=f"mo_plots/{fname_base}_5_num_backups.png",
        y_axis_key="num_backups",
        y_axis_lab="Num Backups", 
        # legend_loc="lower left",
    )
    make_eval_plot(
        filenames=filenames,
        plot_filename=f"mo_plots/{fname_base}_6_additive_eps_metric.png",
        y_axis_key="additive_eps_metric",
        y_axis_lab="Additive Epsilon Metric", 
        # legend_loc="lower left",
    )
    make_eval_plot(
        filenames=filenames,
        plot_filename=f"mo_plots/{fname_base}_7_sparsity_metric.png",
        y_axis_key="sparsity_metric",
        y_axis_lab="Sparsity Metric", 
        # legend_loc="lower left",
    )





if __name__ == "__main__":
    if not os.path.exists("mo_plots"):
        os.makedirs("mo_plots")

    # ------------------------------------------------------------------------------------------------------------------
    # DST
    # ------------------------------------------------------------------------------------------------------------------
    
    if "400" in sys.argv or "all" in sys.argv or "dst" in sys.argv:
        print("Plotting: ", "400")
        filenames = glob.glob("mo_eval_logs/400_*/**/eval_log.txt", recursive=True)
        fname_base = "400_dst"
        make_many_eval_plots(filenames, fname_base)
    
    if "401" in sys.argv or "all" in sys.argv or "dst" in sys.argv:
        print("Plotting: ", "401")
        filenames = glob.glob("mo_eval_logs/401_*/**/eval_log.txt", recursive=True)
        fname_base = "401_dst_gym"
        make_many_eval_plots(filenames, fname_base)

    if "410" in sys.argv or "all" in sys.argv or "dst" in sys.argv:
        print("Plotting: ", "410")
        filenames = glob.glob("mo_eval_logs/410_*/**/eval_log.txt", recursive=True)
        fname_base = "410_dst_stoch"
        make_many_eval_plots(filenames, fname_base)

    if "440" in sys.argv or "all" in sys.argv or "dst" in sys.argv:
        print("Plotting: ", "440")
        filenames = glob.glob("mo_eval_logs/440_*/**/eval_log.txt", recursive=True)
        fname_base = "440_dst_improved"
        make_many_eval_plots(filenames, fname_base)

    if "450" in sys.argv or "all" in sys.argv or "dst" in sys.argv:
        print("Plotting: ", "450")
        filenames = glob.glob("mo_eval_logs/450_*/**/eval_log.txt", recursive=True)
        fname_base = "450_dst_improved_stoch"
        make_many_eval_plots(filenames, fname_base)

    # ------------------------------------------------------------------------------------------------------------------
    # Gym envs
    # ------------------------------------------------------------------------------------------------------------------

    if "500" in sys.argv or "all" in sys.argv or "gym" in sys.argv:
        print("Plotting: ", "500")
        filenames = glob.glob("mo_eval_logs/500_*/**/eval_log.txt", recursive=True)
        fname_base = "500_hpopt_fruit_tree"
        make_many_eval_plots(filenames, fname_base)











def old_plots_main():
    # ------------------------------------------------------------------------------------------------------------------
    # Scalability plots
    # ------------------------------------------------------------------------------------------------------------------

    ###
    # Scalability tests - 200_dense_dimension_scalability
    ###
    if "200" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/200_dense_dimension_scalability_1736165498/**/eval.txt", recursive=True)
        make_eum_scalability_plot(
            filenames=filenames,
            plot_filename="plots/200_scalability_util_vs_reward_dimension.png",
            x_axis_key="reward_dim",
            x_axis_lab="Reward Dimension",
        )
        make_num_trials_scalability_plot(
            filenames=filenames,
            plot_filename="plots/200_scalability_num_trials_vs_reward_dimension.png",
            x_axis_key="reward_dim",
            x_axis_lab="Reward Dimension",
        )

    ###
    # Scalability tests - 210_dense_actions_scalability
    ###
    if "210" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/210_dense_actions_scalability_1736167326/**/eval.txt", recursive=True)
        make_eum_scalability_plot(
            filenames=filenames,
            plot_filename="plots/210_scalability_util_vs_ch_size.png",
            x_axis_key="num_actions",
            x_axis_lab="#Actions (Size of Optimal Convex Hull)",
        )
        make_num_trials_scalability_plot(
            filenames=filenames,
            plot_filename="plots/210_scalability_num_trials_vs_ch_size.png",
            x_axis_key="num_actions",
            x_axis_lab="#Actions (Size of Optimal Convex Hull)",
        )

    



    # ------------------------------------------------------------------------------------------------------------------
    # EUM Eval plots
    # ------------------------------------------------------------------------------------------------------------------

    # ###
    # # 600_deep_sea_treasure
    # ###
    # if "600" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
    #     filenames = glob.glob("results/TODO/**/eval.txt", recursive=True)
    #     make_eum_plot(
    #         filenames=filenames,
    #         plot_filename="plots/TODO.png",
    #     )
    #     make_num_trials_plot(
    #         filenames=filenames,
    #         plot_filename="plots/TODO.png",
    #     )



    ###
    # 611_fruit_tree (stoch, size=default)
    # - run with 1 repitition and algorithms tuned on gym's dst env
    ###  
    if "611" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/611_deep_sea_treasure_stoch_1739211378/**/eval.txt", recursive=True)
        make_eum_plot(
            filenames=filenames,
            plot_filename="plots/611_dst_stoch_eum.png",
            title="EUM vs Search Time (Deep Sea Treasure, swept_prob=0.01)",
        )
        make_num_trials_plot(
            filenames=filenames,
            plot_filename="plots/611_dst_stoch_num_trials.png",
            title="Num Trials vs Search Time (Deep Sea Treasure, swept_prob=0.01)",
        )



    ###
    # 612_fruit_tree (stoch, size=default)
    # - run with 25 repitition and algorithms tuned on gym's dst env (with making l_inf_thresh much bigger (0.05))
    # and set chmcts to use the same more reasonable bias as czt
    ###  
    if "612" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/612_deep_sea_treasure_stoch_1739213842/**/eval.txt", recursive=True)
        make_eum_plot(
            filenames=filenames,
            plot_filename="plots/612_dst_stoch_eum.png",
            title="EUM vs Search Time (Deep Sea Treasure, swept_prob=0.01)",
        )
        make_num_trials_plot(
            filenames=filenames,
            plot_filename="plots/612_dst_stoch_num_trials.png",
            title="Num Trials vs Search Time (Deep Sea Treasure, swept_prob=0.01)",
        )



    ###
    # 640_fruit_tree (det, depth=7)
    ###  
    if "640" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/640_fruit_tree_1736139636/**/eval.txt", recursive=True)
        make_eum_plot(
            filenames=filenames,
            plot_filename="plots/640_fruit_tree_7_eum.png",
            title="EUM vs Search Time (Fruit-Tree, depth=7)",
        )
        make_num_trials_plot(
            filenames=filenames,
            plot_filename="plots/640_fruit_tree_7_num_trials.png",
            title="Num Trials vs Search Time (Fruit-Tree, depth=7)",
        )
        
    ###
    # 650_fruit_tree_stoch_5 (stoch, depth=5)
    ###  
    if "650" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/650_fruit_tree_stoch_5_1736149191/**/eval.txt", recursive=True)
        make_eum_plot(
            filenames=filenames,
            plot_filename="plots/650_fruit_tree_stoch_5_eum.png",
            title="EUM vs Search Time (Stochastic-Fruit-Tree, depth=5)",
        )
        make_num_trials_plot(
            filenames=filenames,
            plot_filename="plots/650_fruit_tree_stoch_5_num_trials.png",
            title="Num Trials vs Search Time (Stochastic-Fruit-Tree, depth=5)",
        )
        
    ###
    # 660_fruit_tree_stoch_7 (stoch, depth=7)
    ###  
    if "660" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/660_fruit_tree_stoch_7_1736156862/**/eval.txt", recursive=True)
        make_eum_plot(
            filenames=filenames,
            plot_filename="plots/660_fruit_tree_stoch_7_eum.png",
            title="EUM vs Search Time (Stochastic-Fruit-Tree, depth=7)",
        )
        make_num_trials_plot(
            filenames=filenames,
            plot_filename="plots/660_fruit_tree_stoch_7_num_trials.png",
            title="Num Trials vs Search Time (Stochastic-Fruit-Tree, depth=7)",
        )

