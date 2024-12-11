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


def read_mc_eval_into_arrays(filename, alg_ids, bias_or_temps, replicates, values, num_trialss, epsilons, hmcts_uct_threshs, dents_temps, alg_id=None, num_trials_scale=1):
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
            elif param_id in ["bias", "temp"]:
                bias_or_temp = float(val)
            elif param_id in ["dents_temp"]:
                dents_temp = float(val)
            elif param_id in ["epsilon"]:
                epsilon = float(val)
            elif param_id in ["uct_budget_threshold"]:
                uct_thresh = int(val)
        
        eval_ids = f.readline().strip().split(",")
        i = 0
        for eval_id in eval_ids:
            if eval_id == "replicate":
                replicate_idx = i
            elif eval_id == "num_trials":
                num_trials_idx = i
            elif eval_id == "mc_eval_mean":
                value_idx = i
            i += 1

        for line in f.readlines():
            csv_vals = line.strip().split(",")
            alg_ids.append(alg_id)
            bias_or_temps.append(float(bias_or_temp))
            replicates.append(int(csv_vals[replicate_idx]))
            values.append(float(csv_vals[value_idx]))
            num_trialss.append(int(csv_vals[num_trials_idx])/num_trials_scale)
            epsilons.append(float(epsilon))
            hmcts_uct_threshs.append(uct_thresh)
            dents_temps.append(float(dents_temp))

def read_eval_files(filenames,num_trials_scale):
    alg_ids, bias_or_temps, replicates, values, num_trialss, epsilons, hmcts_uct_threshs, dents_temps = [], [], [], [], [], [], [], []

    for filename in filenames:
        alg_id = None
        poss_alg_ids = ["db-dents","db-ments","dents","ments","puct","rents","tents","uct","hmcts"]
        for poss_alg_id in poss_alg_ids:
            if poss_alg_id in filename:
                alg_id = poss_alg_id
        
        read_mc_eval_into_arrays(
            filename=filename, 
            alg_ids=alg_ids, 
            bias_or_temps=bias_or_temps, 
            replicates=replicates,
            values=values, 
            num_trialss=num_trialss, 
            epsilons=epsilons,
            hmcts_uct_threshs=hmcts_uct_threshs,
            dents_temps=dents_temps,
            alg_id=alg_id,
            num_trials_scale=num_trials_scale)

    return alg_ids, bias_or_temps, replicates, values, num_trialss, epsilons, hmcts_uct_threshs, dents_temps

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
    num_trials_scale=1):
    """Read in data, preprocess, and then call make plot"""


    if hue_key is None:
        hue_key = "pretty_alg_id"
    if title is None:
        title = "MC Eval vs Num Trials"
    if xaxis_lab is None:
        xaxis_lab = "Number of Trials"
    if yaxis_lab is None:
        yaxis_lab = "Monte-Carlo Value Estimate"
    if legend_lab is None and use_legend:
        legend_lab = "Algorithm"
    if alg_ids_to_add_param_to is None:
        alg_ids_to_add_param_to = []

    if num_trials_scale > 1:
        xaxis_lab += " (x{scale})".format(scale=num_trials_scale)

    alg_ids, bias_or_temps, replicates, values, num_trialss, epsilons, hmcts_uct_threshs, dents_temps = read_eval_files(filenames,num_trials_scale)   

    pretty_alg_ids = []
    for i, alg_id in enumerate(alg_ids):
        pretty_alg_id = alg_id
        if pretty_alg_id == "est":
            pretty_alg_id = "bts"
        if pretty_alg_id == "db-ments":
            pretty_alg_id = "dents"
        if alg_id in alg_ids_to_add_param_to:
            pretty_alg_id = "{alg_id}({param})".format(alg_id=pretty_alg_id,param=bias_or_temps[i])
        pretty_alg_ids.append(pretty_alg_id.upper())
    
    mc_eval_df_dict = {
        "num_trials": num_trialss,
        "mc_value_estimate": values,
        # "log_abs_mc_value_estimate": log_ys,
        "algorithm_id": alg_ids,
        "bias_or_temp": bias_or_temps,
        "pretty_alg_id": pretty_alg_ids,
        "replicates": replicates,
        "eps": epsilons,
        "uct_budget_threshold": hmcts_uct_threshs,
        "dents_temp": dents_temps,
    }
    
    palette = None
    dashes = None
    if hue_per_algo:
        palette = {}
        dashes = {}
        alg_set = set(pretty_alg_ids)
        for alg_id in alg_set:
            dashes[alg_id] = ""
            if "1.0" in alg_id:
                dashes[alg_id] = (4,2)
            if "UCT" in alg_id:
                palette[alg_id] = "tab:green"
            if "PUCT" in alg_id:
                palette[alg_id] = "tab:gray"
            if "MENTS" in alg_id:
                palette[alg_id] = "tab:red"
            if "BTS" in alg_id:
                palette[alg_id] = "tab:blue"
            if "DENTS" in alg_id:
                palette[alg_id] = "tab:orange"
            if "TENTS" in alg_id:
                palette[alg_id] = "tab:purple"
            if "RENTS" in alg_id:
                palette[alg_id] = "tab:brown"
            if "HMCTS" in alg_id:
                palette[alg_id] = "tab:grey"

    markers = None
    if add_markers:
        markers = {}
        for alg_id in alg_set:
            markers[alg_id] = ""
            if "UCT" in alg_id:
                markers[alg_id] = 5
            if "MENTS" in alg_id:
                markers[alg_id] = 7
            if "DENTS" in alg_id:
                markers[alg_id] = 6


    df = pd.DataFrame(mc_eval_df_dict)

    if num_trials_truncate is not None:
        df = df[df["num_trials"] <= num_trials_truncate]

    make_plot_df(
        df=df, 
        xaxis_key="num_trials", 
        yaxis_key="mc_value_estimate", 
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
    
    if not sep_eps_plots:
        return

    eps_set = set(epsilons)
    for eps in eps_set:
        eps_df = df[df['eps'] == eps]
        eps_filename = plot_filename.format(eps=eps)
        make_plot_df(
            df=eps_df, 
            xaxis_key="num_trials", 
            yaxis_key="mc_value_estimate", 
            hue_key=hue_key,
            palette=palette,
            style_key=hue_key,
            dashes=dashes,
            xaxis_lab=xaxis_lab,
            yaxis_lab=yaxis_lab,
            y_scale_transform_forward=y_scale_transform_forward,
            y_scale_transform_inverse=y_scale_transform_inverse,
            legend_lab=legend_lab,
            filename=eps_filename,
            y_axis_range=y_axis_range,
            markers=markers,
            markevery=markevery,
            use_legend=use_legend,
            alpha=alpha)

    
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

    #
    # 000 = plots for the main paper
    #
    if "000_fig1a" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = [
            "results/dchain_env/10-1.0/021_len_10_main_paper/uct/eval_bias=-1.csv",
            # "results/dchain_env/10-1.0/021_len_10_main_paper/dents/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-1.0/021_len_10_main_paper/db-ments/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-1.0/021_len_10_main_paper/ments/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-1.0/021_len_10_main_paper/ments/eval_epsilon=0.1,temp=0.01.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_fig1a_dchain.png",
            hue_key="pretty_alg_id",
            title="",
            num_trials_truncate=3000,
            alg_ids_to_add_param_to=["ments"],
            add_markers=True,
            markevery=25,
            alpha=0.8)
        
    if "000_fig1b" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = [
            "results/dchain_env/10-0.5/021_len_10_main_paper/uct/eval_bias=-1.csv",
            "results/dchain_env/10-0.5/021_len_10_main_paper/dents/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-0.5/021_len_10_main_paper/ments/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-0.5/021_len_10_main_paper/ments/eval_epsilon=0.1,temp=0.01.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_fig1b_dchain_half.png",
            title="",
            hue_key="pretty_alg_id",
            num_trials_truncate=3000,
            alg_ids_to_add_param_to=["ments"],
            add_markers=True,
            markevery=25,
            alpha=0.8)
        
    if "000_fig_fl" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/uct/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/ments/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_fig_fl.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=100000)
        
    if "000_fig_fl_full" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        # filenames = glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/*/eval_*.csv")
        filenames = glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/uct/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/ments/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/est/eval_*.csv")
        # filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/db-ments/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/rents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/tents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/hmcts/eval_*.csv")
        puct_filename = None
        for filename in filenames:
            if "puct" in filename:
                puct_filename = filename
        if puct_filename is not None:
            filenames.remove(puct_filename)
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_fig_fl_full.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000,
            y_scale_transform_forward=y_scale_piecwise_linear_forward,
            y_scale_transform_inverse=y_scale_piecwise_linear_inverse,
            alpha=0.8,
            use_legend=True,
            num_trials_scale=1000)
        
    if "000_fig_sail" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/sailing_env/6_test/092_s6_test/uct/eval_*.csv")
        filenames += glob.glob("results/sailing_env/6_test/092_s6_test/ments/eval_*.csv")
        filenames += glob.glob("results/sailing_env/6_test/092_s6_test/est/eval_*.csv")
        filenames += glob.glob("results/sailing_env/6_test/092_s6_test/dents/eval_*.csv")
        filenames += glob.glob("results/sailing_env/6_test/092_s6_test/rents/eval_*.csv")
        filenames += glob.glob("results/sailing_env/6_test/092_s6_test/tents/eval_*.csv")
        filenames += glob.glob("results/sailing_env/6_test/092_s6_test/hmcts/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_fig_sail.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000,
            y_scale_transform_forward=get_piecwise_linear_forward_transform(-120.0,-40.0,-90.0,0),
            y_scale_transform_inverse=get_piecwise_linear_inverse_transform(-120.0,-40.0,-90.0,0),
            alpha=0.8,
            num_trials_scale=1000)
        






    #
    # Rebuttal plots
    #
    if "000_rebuttal_one" in sys.argv or "all" in sys.argv:
        filenames = [
            "results/dchain_env/10-1.0/021_len_10_main_paper/ments/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-1.0/021_len_10_main_paper/db-ments/eval_epsilon=0.1,temp=1.csv",
            # "results/dchain_env/10-1.0/021_len_10_main_paper/hmcts/eval_uct_budget_threshold=30,hmcts_total_budget=10000,bias=100.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_rebuttal_dchain.png",
            hue_key="pretty_alg_id",
            title="",
            num_trials_truncate=3000,
            add_markers=False,
            markevery=25)
        
    if "000_rebuttal_two" in sys.argv or "all" in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/ments/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/db-ments/eval_*.csv")
        # filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/hmcts/eval_*.csv")
        # filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_rebuttal_fl.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000,
            add_markers=False,
            markevery=100)

        





    #
    # 10-chain, R_f = 1.0
    #
    expr_id = "001_len_10"
    if "001_10chain_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-1.0/001_len_10/uct/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain_01_uct_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-1.0/001_len_10/puct/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain_01_puct_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-1.0/001_len_10/ments/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain_01_ments_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-1.0/001_len_10/rents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain_01_rents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-1.0/001_len_10/tents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain_01_tents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-1.0/001_len_10/dents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain_01_dents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-1.0/001_len_10/est/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain_01_est_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain_hmcts_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-1.0/001_len_10/hmcts/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain_01_hmcts_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
        





    #
    # 10-chain, R_f = 0.5
    #
    expr_id = "001_len_10"
    if "001_10chain5_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-0.5/001_len_10/uct/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain5_01_uct_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain5_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-0.5/001_len_10/puct/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain5_01_puct_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain5_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-0.5/001_len_10/ments/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain5_01_ments_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain5_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-0.5/001_len_10/rents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain5_01_rents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain5_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-0.5/001_len_10/tents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain5_01_tents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain5_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-0.5/001_len_10/dents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain5_01_dents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain5_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-0.5/001_len_10/est/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain5_01_est_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "001_10chain5_hmcts_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/10-0.5/001_len_10/hmcts/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/001_10chain5_01_hmcts_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
        





    #
    # 20-chain, R_f = 1.0
    #
    expr_id = "003_len_20"
    if "003_20chain_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/003_len_20/uct/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain_01_uct_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/003_len_20/puct/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain_01_puct_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/003_len_20/ments/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain_01_ments_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/003_len_20/rents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain_01_rents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/003_len_20/tents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain_01_tents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/003_len_20/dents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain_01_dents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/003_len_20/est/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain_01_est_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
        





    #
    # 20-chain, R_f = 0.5
    #
    expr_id = "003_len_20"
    if "003_20chain5_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-0.5/003_len_20/uct/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain5_01_uct_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain5_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-0.5/003_len_20/puct/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain5_01_puct_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain5_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-0.5/003_len_20/ments/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain5_01_ments_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain5_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-0.5/003_len_20/rents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain5_01_rents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain5_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-0.5/003_len_20/tents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain5_01_tents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain5_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-0.5/003_len_20/dents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain5_01_dents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "003_20chain5_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-0.5/003_len_20/est/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/003_20chain5_01_est_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=10000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
        





    #
    # Selected 20-chain, final experiments for appendix
    #
    expr_id = "005_len_20"
    if "005_20chain_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/005_len_20/ments/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/005_20chain_01_ments_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=25000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "005_20chain5_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-0.5/005_len_20/ments/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/005_20chain5_01_ments_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=25000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "005_20chain_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/005_len_20/est/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/005_20chain_01_est_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=25000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    if "005_20chain_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/005_len_20/dents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/005_20chain_01_dents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=25000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    expr_id = "006_len_20"
    if "006_20chain_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/006_len_20/dents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/006_20chain_01_dents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=25000,
                y_axis_range=[0.0,1.1],
                use_legend=False)
            
    expr_id = "007_len_20"
    if "007_20chain_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        prefix = "results/dchain_env/20-1.0/007_len_20/dents/eval_"
        filenames = glob.glob(prefix + "*.csv")
        for filename in filenames:
            params = filename[len(prefix):-4]
            make_plot(
                filenames=[filename],
                plot_filename="plots/007_20chain_01_dents_{params}.png".format(params=params),
                hue_key="pretty_alg_id",
                num_trials_truncate=25000,
                y_axis_range=[0.0,1.1],
                use_legend=False)











    #
    # FL, 8x12 hps
    #
    expr_id = "051_fl12_hps"
    if "051_01_fl12_hps_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12/051_fl12_hps/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/051_01_fl12_hps_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)
        
    if "051_02_fl12_hps_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12/051_fl12_hps/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/051_02_fl12_hps_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)
        
    if "051_03_fl12_hps_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12/051_fl12_hps/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/051_03_fl12_hps_ments_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "051_04_fl12_hps_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12/051_fl12_hps/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/051_04_fl12_hps_rents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "051_05_fl12_hps_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12/051_fl12_hps/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/051_05_fl12_hps_tents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "051_06_fl12_hps_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12/051_fl12_hps/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/051_06_fl12_hps_dents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "051_06_fl12_hps_dents_02" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12/051a_fl12_hps/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/051_06_fl12_hps_dents_02.png",
            hue_key="dents_temp",
            num_trials_truncate=1000000,
            hue_per_algo=False)
        
    if "051_07_fl12_hps_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12/051_fl12_hps/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/051_07_fl12_hps_est_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "051_08_fl12_hps_hmcts_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12/051_fl12_hps/hmcts/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/051_08_fl12_hps_hmcts_01.png",
            hue_key="uct_budget_threshold",
            num_trials_truncate=1000000,
            hue_per_algo=False)
        




    #
    # FL, 8x12 test
    #
    expr_id = "052_fl12_test"
    if "052_01_fl12_test_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/*/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/052_01_fl12_test_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "052_01_fl12_test_02" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/uct/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/ments/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x12_test/052_fl12_test/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/052_01_fl12_test_02.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=100000)
        





    #
    # FL, 8x16 test
    #
    expr_id = "050_fl16_test"
    if "050_01_fl16_test_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/050_fl16_test/*/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/050_01_fl16_test_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "050_01_fl16_test_02" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/050_fl16_test/uct/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x16_test/050_fl16_test/ments/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x16_test/050_fl16_test/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x16_test/050_fl16_test/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/050_01_fl16_test_02.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=100000)
        




    #
    # S6 hps
    #
    expr_id = "091_s6_hps"
    if "091_01_s6_hps_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/6/091_s6_hps/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/091_01_s6_hps_uct_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)
        
    if "091_02_s6_hps_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/6/091_s6_hps/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/091_02_s6_hps_puct_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)
        
    if "091_03_s6_hps_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/6/091_s6_hps/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/091_03_s6_hps_ments_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "091_04_s6_hps_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/6/091_s6_hps/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/091_04_s6_hps_rents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "091_05_s6_hps_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/6/091_s6_hps/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/091_05_s6_hps_tents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "091_06_s6_hps_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/6/091_s6_hps/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/091_06_s6_hps_dents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "091_07_s6_hps_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/6/091_s6_hps/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/091_07_s6_hps_est_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "091_08_s6_hps_hmcts_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/6/091_s6_hps/hmcts/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/091_08_s6_hps_hmcts_01.png",
            hue_key="uct_budget_threshold",
            num_trials_truncate=1000000,
            hue_per_algo=False)
        




    #
    # S6 test
    #
    expr_id = "092_s6_test"
    if "092_01_s6_test_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/6_test/092_s6_test/*/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/092_01_s6_test_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "092_01_s6_test_02" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/6_test/092_s6_test/uct/eval_*.csv")
        filenames += glob.glob("results/sailing_env/6_test/092_s6_test/ments/eval_*.csv")
        filenames += glob.glob("results/sailing_env/6_test/092_s6_test/dents/eval_*.csv")
        filenames += glob.glob("results/sailing_env/6_test/092_s6_test/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/092_01_s6_test_02.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        




    #
    # FL8 demos - seperate plots for MENTS (01), RENTS (02) and TENTS (03)
    #
    expr_id = "053_fl8_1_0"
    if "053_fl8_1_0_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/053_fl8_1_0/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/053_fl8_1_0/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/053_fl8_1_0/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/053_fl8_1_0_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "053_fl8_1_0_02" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/053_fl8_1_0/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/053_fl8_1_0/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/053_fl8_1_0/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/053_fl8_1_0_02.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "053_fl8_1_0_03" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/053_fl8_1_0/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/053_fl8_1_0/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/053_fl8_1_0/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/053_fl8_1_0_03.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        


    expr_id = "054_fl8_0_5"
    if "054_fl8_0_5_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/054_fl8_0_5/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/054_fl8_0_5/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/054_fl8_0_5/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/054_fl8_0_5_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "054_fl8_0_5_02" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/054_fl8_0_5/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/054_fl8_0_5/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/054_fl8_0_5/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/054_fl8_0_5_02.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "054_fl8_0_5_03" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/054_fl8_0_5/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/054_fl8_0_5/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/054_fl8_0_5/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/054_fl8_0_5_03.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        


    expr_id = "055_fl8_0_1"
    if "055_fl8_0_1_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/055_fl8_0_1/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/055_fl8_0_1/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/055_fl8_0_1/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/055_fl8_0_1_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "055_fl8_0_1_02" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/055_fl8_0_1/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/055_fl8_0_1/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/055_fl8_0_1/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/055_fl8_0_1_02.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "055_fl8_0_1_03" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/055_fl8_0_1/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/055_fl8_0_1/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/055_fl8_0_1/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/055_fl8_0_1_03.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        


    expr_id = "056_fl8_0_05"
    if "056_fl8_0_05_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/056_fl8_0_05/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/056_fl8_0_05/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/056_fl8_0_05/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/056_fl8_0_05_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "056_fl8_0_05_02" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/056_fl8_0_05/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/056_fl8_0_05/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/056_fl8_0_05/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/056_fl8_0_05_02.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "056_fl8_0_05_03" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/056_fl8_0_05/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/056_fl8_0_05/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/056_fl8_0_05/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/056_fl8_0_05_03.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        

        
    expr_id = "057_fl8_0_01"
    if "057_fl8_0_01_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/057_fl8_0_01/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/057_fl8_0_01/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/057_fl8_0_01/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/057_fl8_0_01_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "057_fl8_0_01_02" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/057_fl8_0_01/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/057_fl8_0_01/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/057_fl8_0_01/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/057_fl8_0_01_02.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "057_fl8_0_01_03" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/057_fl8_0_01/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/057_fl8_0_01/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/057_fl8_0_01/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/057_fl8_0_01_03.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        

        
    expr_id = "058_fl8_0_005"
    if "058_fl8_0_005_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/058_fl8_0_005/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/058_fl8_0_005/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/058_fl8_0_005/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/058_fl8_0_005_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "058_fl8_0_005_02" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/058_fl8_0_005/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/058_fl8_0_005/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/058_fl8_0_005/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/058_fl8_0_005_02.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "058_fl8_0_005_03" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/058_fl8_0_005/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/058_fl8_0_005/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/058_fl8_0_005/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/058_fl8_0_005_03.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        

        
    expr_id = "059_fl8_0_001"
    if "059_fl8_0_001_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/059_fl8_0_001/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/059_fl8_0_001/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/059_fl8_0_001/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/059_fl8_0_001_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "059_fl8_0_001_02" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/059_fl8_0_001/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/059_fl8_0_001/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/059_fl8_0_001/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/059_fl8_0_001_02.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        
    if "059_fl8_0_001_03" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/059_fl8_0_001/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/059_fl8_0_001/dents/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8/059_fl8_0_001/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/059_fl8_0_001_03.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)



