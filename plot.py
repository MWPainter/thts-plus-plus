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

def make_plot_df(
    df, 
    xaxis_key, 
    yaxis_key, 
    hue_key=None, 
    style_key=None, 
    title=None, 
    xaxis_lab=None, 
    yaxis_lab=None, 
    legend_lab=None,
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
    font_scale=1.2):
    """General helper for plotting in our style."""

    plt.figure()
    sns.set(style="darkgrid")

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
    sns.set(font_scale=font_scale)

    if y_scale_transform_forward is not None and y_scale_transform_inverse is not None:
        plt.yscale("function", functions=(y_scale_transform_forward, y_scale_transform_inverse))

    if palette is None:
        palette = "deep"
    if dashes is None:
        dashes = False
    if markers is None:
        markers = False

    sns.lineplot(
        data=df, 
        x=xaxis_key, 
        y=yaxis_key, 
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
    if xaxis_lab is not None:
        plt.xlabel(xaxis_lab)
    if yaxis_lab is not None:
        plt.ylabel(yaxis_lab)
    if vertical_lines is not None:
        for x in vertical_lines:
            plt.axvline(x=x, color='k', linestyle='--')
    if legend_lab is not None:
        plt.legend(loc="lower right", title=legend_lab)
    if y_axis_range is not None:
        plt.gca().set_ylim(y_axis_range)
    if not use_legend:
        plt.gca().get_legend().remove()

    if filename is not None:
        plt.savefig(filename)
        plt.close()


def read_mc_eval_into_arrays(filename, alg_ids, search_times, bias_or_temps, replicates, values, num_trialss, epsilons, hmcts_uct_threshs, dents_temps, alg_id=None, num_trials_scale=1):
    with open(filename) as f:
        param_ids = f.readline().strip().split(",")
        param_vals = f.readline().strip().split(",")
        _ = f.readline()

        epsilon = 0.0
        uct_thresh = 0
        dents_temp = 0.0
        for param_id, val in zip(param_ids,param_vals):
            if param_id == "alg":
                alg_id = val
            # elif param_id in ["bias", "temp"]:
            #     bias_or_temp = float(val)
            # elif param_id in ["dents_temp"]:
            #     dents_temp = float(val)
            # elif param_id in ["epsilon"]:
            #     epsilon = float(val)
            # elif param_id in ["uct_budget_threshold"]:
            #     uct_thresh = int(val)
        
        eval_ids = f.readline().strip().split(",")
        i = 0
        for eval_id in eval_ids:
            if eval_id == "replicate":
                replicate_idx = i
            elif eval_id == "search_time":
                search_time_idx = i
            elif eval_id == "num_trials":
                num_trials_idx = i
            elif eval_id == "mc_eval_utility_mean":
                value_idx = i
            i += 1

        for line in f.readlines():
            csv_vals = line.strip().split(",")
            alg_ids.append(alg_id)
            # bias_or_temps.append(float(bias_or_temp))
            replicates.append(int(csv_vals[replicate_idx]))
            values.append(float(csv_vals[value_idx]))
            search_times.append(float(csv_vals[search_time_idx]))
            num_trialss.append(int(csv_vals[num_trials_idx])/num_trials_scale)
            # epsilons.append(float(epsilon))
            # hmcts_uct_threshs.append(uct_thresh)
            # dents_temps.append(float(dents_temp))

def read_eval_files(filenames,num_trials_scale):
    alg_ids, search_times, bias_or_temps, replicates, values, num_trialss, epsilons, hmcts_uct_threshs, dents_temps = [], [], [], [], [], [], [], [], []

    for filename in filenames:
        alg_id = None
        poss_alg_ids = ["czt","chmcts","smbts","smdents"]
        for poss_alg_id in poss_alg_ids:
            if poss_alg_id in filename:
                alg_id = poss_alg_id
        
        read_mc_eval_into_arrays(
            filename=filename, 
            alg_ids=alg_ids, 
            search_times=search_times,
            bias_or_temps=bias_or_temps, 
            replicates=replicates,
            values=values, 
            num_trialss=num_trialss, 
            epsilons=epsilons,
            hmcts_uct_threshs=hmcts_uct_threshs,
            dents_temps=dents_temps,
            alg_id=alg_id,
            num_trials_scale=num_trials_scale)

    return alg_ids, search_times, bias_or_temps, replicates, values, num_trialss, epsilons, hmcts_uct_threshs, dents_temps

def make_plot(
    filenames, 
    plot_filename, 
    hue_key=None, 
    title=None, 
    xaxis_lab=None, 
    yaxis_lab=None, 
    legend_lab=None, 
    num_trials_truncate=None,
    alg_ids_to_add_param_to=None,
    y_scale_transform_forward=None,
    y_scale_transform_inverse=None,
    sep_eps_plots=False,
    y_axis_range=None,
    add_markers=False,
    markevery=1,
    use_legend=True,
    hue_per_algo=True,
    alpha=1.0,
    num_trials_scale=1,
    plot_num_trials=False):
    """Read in data, preprocess, and then call make plot"""


    if hue_key is None:
        hue_key = "pretty_alg_id"
    if title is None:
        title = "Utility vs Search Time"
    if xaxis_lab is None:
        xaxis_lab = "Search Time"
    if yaxis_lab is None:
        yaxis_lab = "Utility"
    if legend_lab is None and use_legend:
        legend_lab = "Algorithm"
    if alg_ids_to_add_param_to is None:
        alg_ids_to_add_param_to = []

    if num_trials_scale > 1:
        xaxis_lab += " (x{scale})".format(scale=num_trials_scale)

    alg_ids, search_times, bias_or_temps, replicates, values, num_trialss, epsilons, hmcts_uct_threshs, dents_temps = read_eval_files(filenames,num_trials_scale)   

    pretty_alg_ids = []
    for i, alg_id in enumerate(alg_ids):
        pretty_alg_id = alg_id
        # if pretty_alg_id == "est":
        #     pretty_alg_id = "bts"
        # if pretty_alg_id == "db-ments":
        #     pretty_alg_id = "dents"
        # if alg_id in alg_ids_to_add_param_to:
        #     pretty_alg_id = "{alg_id}({param})".format(alg_id=pretty_alg_id,param=bias_or_temps[i])
        pretty_alg_ids.append(pretty_alg_id.upper())
    
    mc_eval_df_dict = {
        "num_trials": num_trialss,
        "utility": values,
        # "log_abs_mc_value_estimate": log_ys,
        "algorithm_id": alg_ids,
        # "bias_or_temp": bias_or_temps,
        "pretty_alg_id": pretty_alg_ids,
        "replicates": replicates,
        "search_time": search_times,
        # "eps": epsilons,
        # "uct_budget_threshold": hmcts_uct_threshs,
        # "dents_temp": dents_temps,
    }
    
    palette = None
    dashes = None
    if hue_per_algo:
        palette = {}
        dashes = {}
        alg_set = set(pretty_alg_ids)
        for alg_id in alg_set:
            dashes[alg_id] = ""
            # if "1.0" in alg_id:
            #     dashes[alg_id] = (4,2)
            if "CZT" in alg_id:
                palette[alg_id] = "tab:green"
            # if "PUCT" in alg_id:
            #     palette[alg_id] = "tab:gray"
            # if "MENTS" in alg_id:
            #     palette[alg_id] = "tab:red"
            if "SMDENTS" in alg_id:
                palette[alg_id] = "tab:blue"
            if "SMBTS" in alg_id:
                palette[alg_id] = "tab:orange"
            if "CHMCTS" in alg_id:
                palette[alg_id] = "tab:purple"
            # if "RENTS" in alg_id:
            #     palette[alg_id] = "tab:brown"
            # if "HMCTS" in alg_id:
            #     palette[alg_id] = "tab:grey"

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


    df = pd.DataFrame(mc_eval_df_dict)

    if num_trials_truncate is not None:
        df = df[df["num_trials"] <= num_trials_truncate]

    if not plot_num_trials:
        make_plot_df(
            df=df, 
            xaxis_key="search_time", 
            yaxis_key="utility", 
            title=title,
            hue_key=hue_key,
            palette=palette,
            style_key=hue_key,
            dashes=dashes,
            xaxis_lab=xaxis_lab,
            yaxis_lab=yaxis_lab,
            y_scale_transform_forward=y_scale_transform_forward,
            y_scale_transform_inverse=y_scale_transform_inverse,
            legend_lab=legend_lab,
            filename=plot_filename,
            y_axis_range=y_axis_range,
            markers=markers,
            markevery=markevery,
            use_legend=use_legend,
            alpha=alpha)
        return

    title = "Num Trials vs Search Time"
    xaxis_lab = "Search Time"
    yaxis_lab = "Trials"

    make_plot_df(
        df=df, 
        xaxis_key="search_time", 
        yaxis_key="num_trials", 
        title=title,
        hue_key=hue_key,
        palette=palette,
        style_key=hue_key,
        dashes=dashes,
        xaxis_lab=xaxis_lab,
        yaxis_lab=yaxis_lab,
        y_scale_transform_forward=y_scale_transform_forward,
        y_scale_transform_inverse=y_scale_transform_inverse,
        legend_lab=legend_lab,
        filename=plot_filename,
        y_axis_range=y_axis_range,
        markers=markers,
        markevery=markevery,
        use_legend=use_legend,
        alpha=alpha)
    
    # if not sep_eps_plots:
    #     return

    # eps_set = set(epsilons)
    # for eps in eps_set:
    #     eps_df = df[df['eps'] == eps]
    #     eps_filename = plot_filename.format(eps=eps)
    #     make_plot_df(
    #         df=eps_df, 
    #         xaxis_key="num_trials", 
    #         yaxis_key="mc_value_estimate", 
    #         hue_key=hue_key,
    #         palette=palette,
    #         style_key=hue_key,
    #         dashes=dashes,
    #         xaxis_lab=xaxis_lab,
    #         yaxis_lab=yaxis_lab,
    #         y_scale_transform_forward=y_scale_transform_forward,
    #         y_scale_transform_inverse=y_scale_transform_inverse,
    #         legend_lab=legend_lab,
    #         filename=eps_filename,
    #         y_axis_range=y_axis_range,
    #         markers=markers,
    #         markevery=markevery,
    #         use_legend=use_legend,
    #         alpha=alpha)

    
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



    ###
    # Old plots for RG pressie
    ###    
    
    # if "001" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
    #     # filenames = glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/*/eval_*.csv")
    #     # filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/hmcts/eval_*.csv")
    #     filenames = [
    #         "results/deep-sea-treasure-v0/001_poc_dst_1710339245/smdents/eval.csv",
    #         "results/deep-sea-treasure-v0/001_poc_dst_1710339245/czt/eval.csv",
    #         "results/deep-sea-treasure-v0/001_poc_dst_1710339245/chmcts/eval.csv",
    #     ]
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/001_num_trials.png",
    #         hue_key="pretty_alg_id",
    #         alpha=0.8,
    #         use_legend=True,
    #         plot_num_trials=True)
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/001_utility.png",
    #         hue_key="pretty_alg_id",
    #         alpha=0.8,
    #         use_legend=True,
    #         plot_num_trials=False)
        
    # if "001_1" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
    #     # filenames = glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/*/eval_*.csv")
    #     # filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/hmcts/eval_*.csv")
    #     filenames = [
    #         "results/deep-sea-treasure-v0/001_poc_dst_1710339518/smdents/eval.csv",
    #         "results/deep-sea-treasure-v0/001_poc_dst_1710339518/czt/eval.csv",
    #         "results/deep-sea-treasure-v0/001_poc_dst_1710339518/chmcts/eval.csv",
    #     ]
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/001_1_num_trials.png",
    #         hue_key="pretty_alg_id",
    #         alpha=0.8,
    #         use_legend=True,
    #         plot_num_trials=True)
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/001_1_utility.png",
    #         hue_key="pretty_alg_id",
    #         alpha=0.8,
    #         use_legend=True,
    #         plot_num_trials=False)





    # ------------------------------------------------------------------------------------------------------------------
    




    ###
    # Debugging Py vs C++ plots
    ###    
    
    if "001" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = [
            "results/debug_env_1/001_debug_env_1_1714680035/chmcts/eval.csv",
            "results/debug_env_1/001_debug_env_1_1714680035/czt/eval.csv",
            "results/debug_env_1/001_debug_env_1_1714680035/smbts/eval.csv",
            "results/debug_env_1/001_debug_env_1_1714680035/smdents/eval.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_utility.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=False)
    
    if "002" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = [
            "results/py_debug_env_1/002_debug_py_env_1_1714683371/chmcts/eval.csv",
            "results/py_debug_env_1/002_debug_py_env_1_1714683371/czt/eval.csv",
            "results/py_debug_env_1/002_debug_py_env_1_1714683371/smbts/eval.csv",
            "results/py_debug_env_1/002_debug_py_env_1_1714683371/smdents/eval.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_utility.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=False)
    
    if "003" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = [
            "results/debug_env_2/003_debug_env_2_1714690562/chmcts/eval.csv",
            "results/debug_env_2/003_debug_env_2_1714690562/czt/eval.csv",
            "results/debug_env_2/003_debug_env_2_1714690562/smbts/eval.csv",
            "results/debug_env_2/003_debug_env_2_1714690562/smdents/eval.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_utility.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=False)
    
    if "004" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = [
            "results/py_debug_env_2/004_debug_py_env_2_1714689024/chmcts/eval.csv",
            "results/py_debug_env_2/004_debug_py_env_2_1714689024/czt/eval.csv",
            "results/py_debug_env_2/004_debug_py_env_2_1714689024/smbts/eval.csv",
            "results/py_debug_env_2/004_debug_py_env_2_1714689024/smdents/eval.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/004_utility.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=False)
    
    if "005" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = [
            "results/debug_env_3/005_debug_env_3_1714683965/chmcts/eval.csv",
            "results/debug_env_3/005_debug_env_3_1714683965/czt/eval.csv",
            "results/debug_env_3/005_debug_env_3_1714683965/smbts/eval.csv",
            "results/debug_env_3/005_debug_env_3_1714683965/smdents/eval.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/005_utility.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=False)
    
    if "006" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/py_debug_env_3/006_debug_py_env_3_1714689229/*/eval.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/006_utility.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=False)
    
    if "007" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/debug_env_4/007_debug_env_4_1714684155/*/eval.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/007_utility.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=False)
    
    if "008" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/py_debug_env_4/008_debug_py_env_4_1714689439/*/eval.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/008_utility.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=False)





    # ------------------------------------------------------------------------------------------------------------------
    




    ###
    # Proof of concept plots on gym envs
    ###  
    
    if "009" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/deep-sea-treasure-v0/009_poc_dst_1714743803/*/eval.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/009_dst_num_trials.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=True)
        make_plot(
            filenames=filenames,
            plot_filename="plots/009_dst_utility.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=False)
    
    if "010" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/fruit-tree-v0/010_poc_ft_1714744312/*/eval.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/010_ft_num_trials.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=True)
        make_plot(
            filenames=filenames,
            plot_filename="plots/010_ft_utility.png",
            hue_key="pretty_alg_id",
            alpha=0.8,
            use_legend=True,
            plot_num_trials=False)

