import os

import matplotlib
if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
    matplotlib.use('Agg')
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
    y_scale_transform=None,
    y_scale_inv_transform=None,
    filename=None, 
    vertical_lines=None):
    """General helper for plotting in our style."""

    plt.figure()
    sns.set(style="darkgrid")
    if y_scale_transform is not None and y_scale_inv_transform is not None:
        # plt.gca().set_yscale("function", function=[y_scale_transform, y_scale_inv_transform])
        plt.gca().set_yscale("function", functions=[y_scale_transform, y_scale_inv_transform])
    sns.lineplot(data=df, x=xaxis_key, y=yaxis_key, hue=hue_key, style=style_key, palette="deep")

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
        plt.legend(title=legend_lab)
    if filename is not None:
        plt.savefig(filename)
        plt.close()


def read_mc_eval_into_arrays(filename, alg_ids, bias_or_temps, replicates, values, num_trialss, epsilons, alg_id=None):
    with open(filename) as f:
        param_ids = f.readline().strip().split(",")
        param_vals = f.readline().strip().split(",")
        _ = f.readline()

        epsilon = 0.0
        for param_id, val in zip(param_ids,param_vals):
            if param_id == "alg":
                alg_id = val
            elif param_id in ["bias", "temp", "init_temp"]:
                bias_or_temp = float(val)
            elif param_id in ["epsilon"]:
                epsilon = float(val)
        
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
            num_trialss.append(int(csv_vals[num_trials_idx]))
            epsilons.append(float(epsilon))

def read_eval_files(filenames):
    alg_ids, bias_or_temps, replicates, values, num_trialss, epsilons = [], [], [], [], [], []

    for filename in filenames:
        alg_id = None
        poss_alg_ids = ["db-dents","db-ments","dents","ments","puct","rents","tents","uct"]
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
            alg_id=alg_id)

    return alg_ids, bias_or_temps, replicates, values, num_trialss, epsilons

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
    y_transform=None,
    sep_eps_plots=False):
    """Read in data, preprocess, and then call make plot"""


    if hue_key is None:
        hue_key = "algorithm_id"
    if title is None:
        title = "MC Eval vs Num Trials"
    if xaxis_lab is None:
        xaxis_lab = "Number of Trials"
    if yaxis_lab is None:
        yaxis_lab = "Monte-Carlo Value Estimate"
    if legend_lab is None:
        legend_lab = "Algorithm"
    if alg_ids_to_add_param_to is None:
        alg_ids_to_add_param_to = []

    alg_ids, bias_or_temps, replicates, values, num_trialss, epsilons = read_eval_files(filenames)   

    pretty_alg_ids = []
    for i, alg_id in enumerate(alg_ids):
        pretty_alg_id = alg_id
        if pretty_alg_id == "est":
            pretty_alg_id = "ets"
        if alg_id in alg_ids_to_add_param_to:
            pretty_alg_id = "{alg_id}({param})".format(alg_id=pretty_alg_id,param=bias_or_temps[i])
        pretty_alg_ids.append(pretty_alg_id)

    if y_transform is not None:
        for i, val in enumerate(values):
            values[i] = y_transform(val)
    
    mc_eval_df_dict = {
        "num_trials": num_trialss,
        "mc_value_estimate": values,
        # "log_abs_mc_value_estimate": log_ys,
        "algorithm_id": alg_ids,
        "bias_or_temp": bias_or_temps,
        "pretty_alg_id": pretty_alg_ids,
        "replicates": replicates,
        "eps": epsilons,
    }

    df = pd.DataFrame(mc_eval_df_dict)

    if num_trials_truncate is not None:
        df = df[df["num_trials"] <= num_trials_truncate]

    if not sep_eps_plots:
        make_plot_df(
            df=df, 
            xaxis_key="num_trials", 
            yaxis_key="mc_value_estimate", 
            hue_key=hue_key,
            xaxis_lab=xaxis_lab,
            yaxis_lab=yaxis_lab,
            # y_scale_transform=y_scale_transform,
            # y_scale_inv_transform=y_scale_inv_transform,
            legend_lab=legend_lab,
            filename=plot_filename)
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
            xaxis_lab=xaxis_lab,
            yaxis_lab=yaxis_lab,
            # y_scale_transform=y_scale_transform,
            # y_scale_inv_transform=y_scale_inv_transform,
            legend_lab=legend_lab,
            filename=eps_filename)

    
def negative_log_transform(x):
    # EPS = 1e-10
    # if x < EPS: x = EPS
    return np.log(x)







if __name__ == "__main__":
    if not os.path.exists("plots"):
        os.makedirs("plots")

    #
    # DChain figures
    #
    if "000_fig1a" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = [
            "results/dchain_env/10-1.0/100_len_10_main_paper/uct/eval_bias=-1.csv",
            "results/dchain_env/10-1.0/100_len_10_main_paper/dents/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-1.0/100_len_10_main_paper/ments/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-1.0/100_len_10_main_paper/ments/eval_epsilon=0.1,temp=0.01.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_fig1a_dchain.png",
            hue_key="pretty_alg_id",
            title="",
            num_trials_truncate=3000,
            alg_ids_to_add_param_to=["ments"])
        
    if "000_fig1b" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = [
            "results/dchain_env/10-0.5/100_len_10_main_paper/uct/eval_bias=-1.csv",
            "results/dchain_env/10-0.5/100_len_10_main_paper/dents/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-0.5/100_len_10_main_paper/ments/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-0.5/100_len_10_main_paper/ments/eval_epsilon=0.1,temp=0.01.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_fig1b_dchain_half.png",
            title="",
            hue_key="pretty_alg_id",
            num_trials_truncate=3000,
            alg_ids_to_add_param_to=["ments"])
        
    if "000_fig_pres_a" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = [
            "results/dchain_env/10-1.0/100_len_10_main_paper/ments/eval_epsilon=0.1,temp=1.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_fig_pres_a.png",
            hue_key="pretty_alg_id",
            title="",
            num_trials_truncate=3000)
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_fig_pres_b.png",
            title="",
            hue_key="pretty_alg_id",
            num_trials_truncate=3000)
        

    if "000_fig_pres_c" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/*/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_fig_pres_c.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=500000)
        

    if "000_fig_pres_d" in sys.argv or "all" in sys.argv or "all_figs" in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/uct/eval_*.csv")
        # filenames += glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/puct/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/ments/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/est/eval_*.csv")
        filenames += glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/000_fig_pres_d.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=500000)
        





    #
    # 10-chain, R_f = 1.0
    #
    expr_id = "001_len_10"
    if "001_10chain_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-1.0/001_len_10/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1500)

    if "001_10chain_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-1.0/001_len_10/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1500)

    if "001_10chain_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-1.0/001_len_10/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain_ments_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)

    if "001_10chain_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:      
        filenames = glob.glob("results/dchain_env/10-1.0/001_len_10/dents/eval_*.csv")  
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain_dents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)

    if "001_10chain_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:        
        filenames = glob.glob("results/dchain_env/10-1.0/001_len_10/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain_est_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)
        
    if "001_10chain_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-1.0/001_len_10/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain_rents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)

    if "001_10chain_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-1.0/001_len_10/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain_tents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)





    #
    # 10-chain, R_f = 0.8
    #
    expr_id = "001_len_10"
    if "001_10chain8_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.8/001_len_10/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain8_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1500)

    if "001_10chain8_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.8/001_len_10/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain8_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1500)

    if "001_10chain8_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.8/001_len_10/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain8_ments_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)

    if "001_10chain8_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.8/001_len_10/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain8_dents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)

    if "001_10chain8_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
        filenames = glob.glob("results/dchain_env/10-0.8/001_len_10/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain8_est_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)
        
    if "001_10chain8_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.8/001_len_10/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain8_rents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)

    if "001_10chain8_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.8/001_len_10/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain8_tents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)





    #
    # 10-chain, R_f = 0.5
    #
    expr_id = "001_len_10"
    if "001_10chain5_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.5/001_len_10/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain5_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1500)

    if "001_10chain5_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.5/001_len_10/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain5_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1500)

    if "001_10chain5_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.5/001_len_10/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain5_ments_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)

    if "001_10chain5_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.5/001_len_10/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain5_dents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)

    if "001_10chain5_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
        filenames = glob.glob("results/dchain_env/10-0.5/001_len_10/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain5_est_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)
        
    if "001_10chain5_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.5/001_len_10/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain5_rents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)

    if "001_10chain5_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/10-0.5/001_len_10/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/001_10chain5_tents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=10000)





    #
    # 20-chain, R_f = 1.0
    #
    expr_id = "003_len_20"
    if "002_20chain_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-1.0/003_len_20/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-1.0/003_len_20/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-1.0/003_len_20/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain_ments_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:     
        filenames = glob.glob("results/dchain_env/20-1.0/003_len_20/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain_dents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv: 
        filenames = glob.glob("results/dchain_env/20-1.0/003_len_20/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain_est_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)
        
    if "002_20chain_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-1.0/003_len_20/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain_rents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-1.0/003_len_20/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain_tents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)





    #
    # 20-chain, R_f = 0.8
    #
    expr_id = "003_len_20"
    if "002_20chain8_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.8/003_len_20/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain8_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain8_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.8/003_len_20/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain8_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain8_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.8/003_len_20/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain8_ments_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain8_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.8/003_len_20/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain8_dents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain8_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
        filenames = glob.glob("results/dchain_env/20-0.8/003_len_20/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain8_est_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)
        
    if "002_20chain8_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.8/003_len_20/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain8_rents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain8_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.8/003_len_20/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain8_tents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)
    # 20-chain, R_f = 0.5
    #
    expr_id = "003_len_20"
    if "002_20chain5_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.5/003_len_20/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain5_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain5_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.5/003_len_20/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain5_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain5_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.5/003_len_20/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain5_ments_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain5_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.5/003_len_20/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain5_dents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain5_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:        
        filenames = glob.glob("results/dchain_env/20-0.5/003_len_20/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain5_est_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)
        
    if "002_20chain5_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.5/003_len_20/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain5_rents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)

    if "002_20chain5_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/dchain_env/20-0.5/003_len_20/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/002_20chain5_tents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=25000)





    #
    # frozen lake, 8x8 hps
    #
    expr_id = "001_fl8_hps"
    if "003_fl8_hps_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/001_fl8_hps/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_fl8_hps_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000)

    if "003_fl8_hps_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/001_fl8_hps/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_fl8_hps_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000)

    if "003_fl8_hps_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/001_fl8_hps/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_fl8_hps_ments_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000,
            sep_eps_plots=True)

    if "003_fl8_hps_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:  
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/001_fl8_hps/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_fl8_hps_dents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000,
            sep_eps_plots=True)

    if "003_fl8_hps_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/001_fl8_hps/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_fl8_hps_est_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000,
            sep_eps_plots=True)
        
    if "003_fl8_hps_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/001_fl8_hps/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_fl8_hps_rents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000,
            sep_eps_plots=True)

    if "003_fl8_hps_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/001_fl8_hps/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_fl8_hps_tents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000,
            sep_eps_plots=True)





    #
    # frozen lake, 8x8 test
    #
    expr_id = "002_fl8_test"
    if "004_fl8_test_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/004_fl8_test_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=500000)

    if "004_fl8_test_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/004_fl8_test_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=500000)

    if "004_fl8_test_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/004_fl8_test_ments_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=500000)

    if "004_fl8_test_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:  
        filenames = glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/004_fl8_test_dents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=500000)

    if "004_fl8_test_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
        filenames = glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/004_fl8_test_est_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=500000)
        
    if "004_fl8_test_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/004_fl8_test_rents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=500000)

    if "004_fl8_test_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/004_fl8_test_tents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=500000)

    if "004_fl8_test_all_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8_test/002_fl8_test/*/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/004_fl8_test_all_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=500000)





    # #
    # # frozen lake, 8x16 test (OLD)
    # #
    # expr_id = "003_fl16_test"
    # if "005_fl16_test_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
    #     filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/003_fl16_test/uct/eval_*.csv")
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/005_fl16_test_uct_01.png",
    #         hue_key="bias_or_temp",
    #         num_trials_truncate=1000000)

    # if "005_fl16_test_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
    #     filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/003_fl16_test/puct/eval_*.csv")
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/005_fl16_test_puct_01.png",
    #         hue_key="bias_or_temp",
    #         num_trials_truncate=1000000)

    # if "005_fl16_test_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
    #     filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/003_fl16_test/ments/eval_*.csv")
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/005_fl16_test_ments_01.png",
    #         hue_key="bias_or_temp",
    #         num_trials_truncate=1000000)

    # if "005_fl16_test_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:  
    #     filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/003_fl16_test/dents/eval_*.csv")
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/005_fl16_test_dents_01.png",
    #         hue_key="bias_or_temp",
    #         num_trials_truncate=1000000)

    # if "005_fl16_test_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
    #     filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/003_fl16_test/est/eval_*.csv")
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/005_fl16_test_est_01.png",
    #         hue_key="bias_or_temp",
    #         num_trials_truncate=1000000)
        
    # if "005_fl16_test_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
    #     filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/003_fl16_test/rents/eval_*.csv")
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/005_fl16_test_rents_01.png",
    #         hue_key="bias_or_temp",
    #         num_trials_truncate=1000000)

    # if "005_fl16_test_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
    #     filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/003_fl16_test/tents/eval_*.csv")
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/005_fl16_test_tents_01.png",
    #         hue_key="bias_or_temp",
    #         num_trials_truncate=1000000)

    # if "005_fl16_test_all_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
    #     filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/003_fl16_test/*/eval_*.csv")
    #     make_plot(
    #         filenames=filenames,
    #         plot_filename="plots/005_fl16_test_all_01.png",
    #         hue_key="pretty_alg_id",
    #         num_trials_truncate=1000000)





    #
    # frozen lake, 8x16 hps
    #
    expr_id = "003_fl16_hps"
    if "006_fl16_hps_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16/003_fl16_hps/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/006_fl16_hps_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "006_fl16_hps_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16/003_fl16_hps/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/006_fl16_hps_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "006_fl16_hps_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16/003_fl16_hps/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/006_fl16_hps_ments_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "006_fl16_hps_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:  
        filenames = glob.glob("results/frozen_lake_env/FL_8x16/003_fl16_hps/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/006_fl16_hps_dents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "006_fl16_hps_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
        filenames = glob.glob("results/frozen_lake_env/FL_8x16/003_fl16_hps/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/006_fl16_hps_est_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "006_fl16_hps_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16/003_fl16_hps/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/006_fl16_hps_rents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "006_fl16_hps_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16/003_fl16_hps/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/006_fl16_hps_tents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)





    #
    # frozen lake, 8x16 test
    #
    expr_id = "004_fl16_test"
    if "007_fl16_test_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/004_fl16_test/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/007_fl16_test_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "007_fl16_test_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/004_fl16_test/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/007_fl16_test_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "007_fl16_test_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/004_fl16_test/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/007_fl16_test_ments_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "007_fl16_test_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:  
        filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/004_fl16_test/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/007_fl16_test_dents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "007_fl16_test_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
        filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/004_fl16_test/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/007_fl16_test_est_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)
        
    if "007_fl16_test_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/004_fl16_test/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/007_fl16_test_rents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "007_fl16_test_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/004_fl16_test/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/007_fl16_test_tents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "007_fl16_test_all_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x16_test/004_fl16_test/*/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/007_fl16_test_all_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)





    #
    # sailing, 5x5 hps
    #
    expr_id = "002_s5_hps"
    if "008_s5_hps_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/5/002_s5_hps/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/008_s5_hps_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "008_s5_hps_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/5/002_s5_hps/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/008_s5_hps_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "008_s5_hps_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/5/002_s5_hps/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/008_s5_hps_ments_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "008_s5_hps_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:  
        filenames = glob.glob("results/sailing_env/5/002_s5_hps/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/008_s5_hps_dents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "008_s5_hps_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
        filenames = glob.glob("results/sailing_env/5/002_s5_hps/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/008_s5_hps_est_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "008_s5_hps_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/5/002_s5_hps/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/008_s5_hps_rents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "008_s5_hps_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/5/002_s5_hps/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/008_s5_hps_tents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)





    #
    # sailing, 7x7 hps
    #
    expr_id = "002_s7_hps"
    if "009_s7_hps_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/7/002_s7_hps/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/009_s7_hps_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "009_s7_hps_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/7/002_s7_hps/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/009_s7_hps_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "009_s7_hps_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/7/002_s7_hps/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/009_s7_hps_ments_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "009_s7_hps_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:  
        filenames = glob.glob("results/sailing_env/7/002_s7_hps/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/009_s7_hps_dents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "009_s7_hps_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
        filenames = glob.glob("results/sailing_env/7/002_s7_hps/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/009_s7_hps_est_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "009_s7_hps_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/7/002_s7_hps/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/009_s7_hps_rents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "009_s7_hps_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/7/002_s7_hps/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/009_s7_hps_tents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)





    #
    # sailing, 10x10 hps
    #
    expr_id = "002_s10_hps"
    if "010_s10_hps_uct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/10/002_s10_hps/uct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/010_s10_hps_uct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "010_s10_hps_puct_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/10/002_s10_hps/puct/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/010_s10_hps_puct_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000)

    if "010_s10_hps_ments_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/10/002_s10_hps/ments/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/010_s10_hps_ments_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "010_s10_hps_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:  
        filenames = glob.glob("results/sailing_env/10/002_s10_hps/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/010_s10_hps_dents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "010_s10_hps_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
        filenames = glob.glob("results/sailing_env/10/002_s10_hps/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/010_s10_hps_est_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        
    if "010_s10_hps_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/10/002_s10_hps/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/010_s10_hps_rents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)

    if "010_s10_hps_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/10/002_s10_hps/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/010_s10_hps_tents_eps={eps}_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=1000000,
            sep_eps_plots=True)
        




    #
    # sailing, 5x5 test
    #
    expr_id = "001_s5_test"
    if "011_s5_test_all_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/5/001_s5_test/*/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/011_s5_test_all_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        




    #
    # sailing, 7x7 test
    #
    expr_id = "001_s7_test"
    if "012_s7_test_all_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/7/001_s7_test/*/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/012_s7_test_all_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)
        




    #
    # sailing, 10x10 test
    #
    expr_id = "001_s10_test"
    if "013_s10_test_all_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/sailing_env/10/001_s10_test/*/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/013_s10_test_all_01.png",
            hue_key="pretty_alg_id",
            num_trials_truncate=1000000)



