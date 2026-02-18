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
    legend_loc=None,
    horizontal_lines=None):
    """
    General helper for plotting lineplots in our style.
    """

    plt.figure()
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
        plt.savefig(filename)
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

    df = pd.read_csv(filepath_or_buffer=filename, header=12, index_col=False, skip_blank_lines=False)
    df["alg_id"] = alg_id
    df["temp"] = temp
    df["bias"] = bias
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
    # add_markers=False,
    markevery=1,
    use_legend=True,
    alpha=1.0,
    num_trials_scale=1,
    horizontal_lines=None,
    ):
    """
    Makes an eval plot using the data in the given filenames
    Can make a non eum plot by specifying y_axis_key
    """

    # Default params
    if hue_key is None:
        hue_key = "alg_id"
    if x_axis_key is None:
        x_axis_key = "num_trials"
    if x_axis_lab is None:
        x_axis_lab = "Num Trials"
    if y_axis_key is None:
        y_axis_key = "mc_val"
    if y_axis_lab is None:
        y_axis_lab = "Monte Carlo Value Estimate"
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
        "uct": uct_str,
        "maxuct": max_uct_str,
        "hmcts": hmcts_str,
        "ments": ments_str,
        "rents": rents_str,
        "tents": tents_str,
    })

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

    # Truncate x_axis if want
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
        alpha=alpha,
        horizontal_lines=horizontal_lines)

def make_param_sens_plot(
    filenames, 
    plot_filename_frmt_str, 
    hue_key=None, 
    title=None, 
    # x_axis_key=None,
    # x_axis_lab=None, 
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
    legend_loc=None,
    seperate_plots=True,
    fl_sparse_reward_transform=False):
    """
    Makes an plot to compare  
    """

    # SCALE_DIFF: Default params
    if hue_key is None:
        hue_key = "alg_id"
    # if x_axis_key is None:
    #     raise Exception("Running make_eum_scalability_plot without providing x_axis_key argument")
    # if x_axis_lab is None:
    #     raise Exception("Running make_eum_scalability_plot without providing x_axis_lab argument")
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

    # Truncate x_axis if want
    if x_axis_truncate is not None:
        df = df[df[x_axis_key] <= x_axis_truncate]

    algs_in_data = set(df["alg_id"])

    if fl_sparse_reward_transform:
        df.loc[df["mc_val"] == 0.0, "mc_val"] = pow(0.99, 36)
        df["mc_val"] = -np.log(df["mc_val"]) / np.log(0.99)
    
    if not seperate_plots:
        x_axis_key = "temp_or_bias"
        x_axis_lab = "Temperature/Bias"
        df["temp_or_bias"] = df["temp"].fillna(df["bias"])
        local_title = title
        if local_title is None:
            local_title = y_axis_lab + " vs " + x_axis_lab
        plot_filename = plot_filename_frmt_str
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
            alpha=alpha)
        return

    for alg_id in [uct_str]:
        if alg_id not in algs_in_data:
            continue
        x_axis_key = "bias"
        x_axis_lab = "Bias"
        bias_df = df[df["alg_id"] == alg_id]
        local_title = title
        if local_title is None:
            local_title = y_axis_lab + " vs " + x_axis_lab
        plot_filename = plot_filename_frmt_str.format(alg_id=alg_id)
        make_lineplot_df(
            df=bias_df, 
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
            alpha=alpha)
        
    for alg_id in [ments_str,bts_str,dents_str]:
        if alg_id not in algs_in_data:
            continue
        x_axis_key = "temp"
        x_axis_lab = "Temperature"
        temp_df = df[df["alg_id"] == alg_id]
        local_title = title
        if local_title is None:
            local_title = y_axis_lab + " vs " + x_axis_lab
        plot_filename = plot_filename_frmt_str.format(alg_id=alg_id)
        make_lineplot_df(
            df=temp_df, 
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
            alpha=alpha)


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
    # Ments Entropy Trap Debug
    # ------------------------------------------------------------------------------------------------------------------
    
    if "999" in sys.argv or "all" in sys.argv or "debug" in sys.argv:
        filenames = glob.glob("aux_eval_logs/999_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/999_debug.png",
            # legend_loc="lower left",
        )

    # ------------------------------------------------------------------------------------------------------------------
    # DChain/EntropyTrap vs temp plots
    # ------------------------------------------------------------------------------------------------------------------
    if "100" in sys.argv or "all" in sys.argv or "dchain" in sys.argv:
        # filenames = glob.glob("results/100_supp_dchain_temp_vary_1740880779/**/eval.txt", recursive=True)
        filenames = glob.glob("aux_eval_logs_happy/100_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/100_dchain_10_vs_tempbias.png",
            seperate_plots=False,
            legend_loc="lower left",
        )
        # make_param_sens_plot(
        #     filenames=filenames,
        #     plot_filename_frmt_str="plots/100_dchain_10_vs_tempbias_{alg_id}.png",
        #     seperate_plots=True,
        #     legend_loc="lower left",
        # )

    if "101" in sys.argv or "all" in sys.argv or "dchain" in sys.argv:
        # filenames = glob.glob("results/101_supp_mod_dchain_temp_vary_1740880832/**/eval.txt", recursive=True)
        filenames = glob.glob("aux_eval_logs_happy/101_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/101_mod_dchain_10_vs_tempbias.png",
            seperate_plots=False,
            legend_loc="lower left",
        )

    if "102" in sys.argv or "all" in sys.argv or "dchain" in sys.argv:
        filenames = glob.glob("aux_eval_logs_happy/102_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/102_entropy_trap_10_vs_tempbias.png",
            seperate_plots=False,
            legend_loc="lower left",
        )

    if "102a" in sys.argv or "all" in sys.argv or "dchain" in sys.argv:
        filenames = glob.glob("aux_eval_logs_happy/102a_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/102a_entropy_trap_10_vs_tempbias.png",
            seperate_plots=False,
            legend_loc="lower left",
        )

    if "102b" in sys.argv or "all" in sys.argv or "dchain" in sys.argv:
        filenames = glob.glob("aux_eval_logs/102b_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/102b_entropy_trap_10_vs_tempbias.png",
            seperate_plots=False,
            legend_loc="lower left",
        )

    if "103" in sys.argv or "all" in sys.argv or "dchain" in sys.argv:
        # filenames = glob.glob("aux_eval_logs_happy/103_*/**/eval_log.txt", recursive=True)
        filenames = glob.glob("aux_eval_logs/103_*/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/103_entropy_trap_15_vs_tempbias.png",
            seperate_plots=False,
            legend_loc="lower left",
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Frozen Lake Scenic Route
    # ------------------------------------------------------------------------------------------------------------------
    if "110" in sys.argv or "all" in sys.argv or "scenic_route" in sys.argv:
        filenames = glob.glob("aux_eval_logs/110_supp_frozen_lake_dense_scenic_route_8x8_1767177529/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/110_FL_scenic_route_8x8.png",
            seperate_plots=False,
            legend_loc="lower left",
        )

    if "111" in sys.argv or "all" in sys.argv or "scenic_route" in sys.argv:
        filenames = glob.glob("aux_eval_logs/111_supp_frozen_lake_dense_scenic_route_8x16_1767196367/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/111_FL_scenic_route_8x16.png",
            seperate_plots=False,
            legend_loc="lower left",
        )

    if "112" in sys.argv or "all" in sys.argv or "scenic_route" in sys.argv:
        filenames = glob.glob("aux_eval_logs/112_supp_frozen_lake_dense_scenic_route_16x16_1767218799/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/112_FL_scenic_route_16x16.png",
            seperate_plots=False,
            legend_loc="lower left",
        )

    if "120" in sys.argv or "all" in sys.argv or "scenic_route" in sys.argv:
        filenames = glob.glob("aux_eval_logs/120_supp_frozen_lake_sparse_scenic_route_8x8_1767085584/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/120_FL_SPARSE_scenic_route_8x8.png",
            seperate_plots=False,
            legend_loc="lower left",
        )

    if "121" in sys.argv or "all" in sys.argv or "scenic_route" in sys.argv:
        filenames = glob.glob("aux_eval_logs/121_supp_frozen_lake_sparse_scenic_route_8x16_1767116486/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/121_FL_SPARSE_scenic_route_8x16.png",
            seperate_plots=False,
            legend_loc="lower left",
        )

    if "122" in sys.argv or "all" in sys.argv or "scenic_route" in sys.argv:
        filenames = glob.glob("aux_eval_logs/122_supp_frozen_lake_sparse_scenic_route_16x16_1767145327/**/eval_log.txt", recursive=True)
        make_param_sens_plot(
            filenames=filenames,
            plot_filename_frmt_str="plots/121_FL_SPARSE_scenic_route_16x16.png",
            seperate_plots=False,
            legend_loc="lower left",
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Eval - Frozen Lake Dense
    # ------------------------------------------------------------------------------------------------------------------
    if "400" in sys.argv or "all" in sys.argv or "fl_dense" in sys.argv:
        filenames = glob.glob("aux_eval_logs/400_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/400_FL_dense_8x8.png",
            # legend_loc="lower left",
        )

    if "401" in sys.argv or "all" in sys.argv or "fl_dense" in sys.argv:
        filenames = glob.glob("aux_eval_logs/401_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/401_FL_dense_8x16.png",
            # legend_loc="lower left",
        )

    if "402" in sys.argv or "all" in sys.argv or "fl_dense" in sys.argv:
        filenames = glob.glob("aux_eval_logs/402_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/402_FL_dense_16x16.png",
            # legend_loc="lower left",
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Eval - Frozen Lake Sparse
    # ------------------------------------------------------------------------------------------------------------------
    if "410" in sys.argv or "all" in sys.argv or "fl_sparse" in sys.argv:
        filenames = glob.glob("aux_eval_logs/410_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/410_FL_SPARSE_8x8.png",
            # legend_loc="lower left",
        )

    if "411" in sys.argv or "all" in sys.argv or "fl_sparse" in sys.argv:
        filenames = glob.glob("aux_eval_logs/411_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/411_FL_SPARSE_8x16.png",
            # legend_loc="lower left",
        )

    if "412" in sys.argv or "all" in sys.argv or "fl_sparse" in sys.argv:
        filenames = glob.glob("aux_eval_logs/412_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/412_FL_SPARSE_16x16.png",
            # legend_loc="lower left",
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Eval - Slippy Frozen Lake Sparse
    # ------------------------------------------------------------------------------------------------------------------
    if "430" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
        filenames = glob.glob("aux_eval_logs/430_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/430_FL_SLIPPY_4x4.png",
            # legend_loc="lower left",
            horizontal_lines=[0.5223],
        )

    if "431" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
        filenames = glob.glob("aux_eval_logs/431_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/431_FL_SLIPPY_5x5.png",
            # legend_loc="lower left",
            horizontal_lines=[0.6116],
        )

    if "432" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
        filenames = glob.glob("aux_eval_logs/432_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/432_FL_SLIPPY_6x6.png",
            # legend_loc="lower left",
            horizontal_lines=[0.4928],
        )

    if "433" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
        filenames = glob.glob("aux_eval_logs/433_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/433_FL_SLIPPY_4x8.png",
            # legend_loc="lower left",
            horizontal_lines=[0.3522],
        )

    if "434" in sys.argv or "all" in sys.argv or "fl_slippy" in sys.argv:
        filenames = glob.glob("aux_eval_logs/434_*/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/434_FL_SLIPPY_4x12.png",
            # legend_loc="lower left",
            horizontal_lines=[0.2969],
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Eval - Sailing (North)
    # ------------------------------------------------------------------------------------------------------------------
    if "440" in sys.argv or "all" in sys.argv or "sailing_n" in sys.argv:
        filenames = glob.glob("aux_eval_logs/440_eval_sailing_north_8x8_1767072325/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/440_SAILING_NORTH_8x8.png",
            # legend_loc="lower left",
        )

    if "441" in sys.argv or "all" in sys.argv or "sailing_n" in sys.argv:
        filenames = glob.glob("aux_eval_logs/441_eval_sailing_north_8x16_1767282517/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/441_SAILING_NORTH_8x16.png",
            # legend_loc="lower left",
        )

    if "442" in sys.argv or "all" in sys.argv or "sailing_n" in sys.argv:
        filenames = glob.glob("aux_eval_logs/442_eval_sailing_north_16x16_1767285588/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/442_SAILING_NORTH_16x16.png",
            # legend_loc="lower left",
        )


    # ------------------------------------------------------------------------------------------------------------------
    # Eval - Sailing (South East)
    # ------------------------------------------------------------------------------------------------------------------
    if "450" in sys.argv or "all" in sys.argv or "sailing_se" in sys.argv:
        filenames = glob.glob("aux_eval_logs/450_eval_sailing_south_east_8x8_1767078921/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/450_SAILING_SOUTH_EAST_8x8.png",
            legend_loc="lower left",
        )

    if "451" in sys.argv or "all" in sys.argv or "sailing_se" in sys.argv:
        filenames = glob.glob("aux_eval_logs/451_eval_sailing_south_east_8x16_1767288640/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/451_SAILING_SOUTH_EAST_8x16.png",
            # legend_loc="lower left",
        )

    if "452" in sys.argv or "all" in sys.argv or "sailing_se" in sys.argv:
        filenames = glob.glob("aux_eval_logs/452_eval_sailing_south_east_16x16_1767291659/**/eval_log.txt", recursive=True)
        make_eval_plot(
            filenames=filenames,
            plot_filename="plots/452_SAILING_SOUTH_EAST_16x16.png",
            # legend_loc="lower left",
        )


                           
                 






    
    ## Example of plotting per algorithm and not combining into one plot
    # if "111" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
    #     filenames = glob.glob("aux_eval_logs/111_uct_on_fl_sparse_len_1742131975/**/eval_log.txt", recursive=True)
    #     make_param_sens_plot(
    #         filenames=filenames,
    #         plot_filename_frmt_str="plots/111={alg_id}.png",
    #     )