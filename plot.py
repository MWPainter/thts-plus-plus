import os

import matplotlib
if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

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


def read_mc_eval_into_arrays(filename, alg_ids, bias_or_temps, replicates, values, num_trialss, alg_id=None):
    with open(filename) as f:
        param_ids = f.readline().strip().split(",")
        param_vals = f.readline().strip().split(",")
        _ = f.readline()

        for param_id, val in zip(param_ids,param_vals):
            if param_id == "alg":
                alg_id = val
            elif param_id in ["bias", "temp", "init_temp"]:
                bias_or_temp = float(val)
        
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

def read_eval_files(filenames):
    alg_ids, bias_or_temps, replicates, values, num_trialss = [], [], [], [], []

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
            alg_id=alg_id)

    return alg_ids, bias_or_temps, replicates, values, num_trialss

def make_plot(
    filenames, 
    plot_filename, 
    hue_key=None, 
    title=None, 
    xaxis_lab=None, 
    yaxis_lab=None, 
    legend_lab=None, 
    num_trials_truncate=None,
    alg_ids_to_add_param_to=None):
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

    alg_ids, bias_or_temps, replicates, values, num_trialss = read_eval_files(filenames)   

    pretty_alg_ids = []
    for i, alg_id in enumerate(alg_ids):
        pretty_alg_id = alg_id
        if alg_id in alg_ids_to_add_param_to:
            pretty_alg_id = "{alg_id}({param})".format(alg_id=alg_id,param=bias_or_temps[i])
        pretty_alg_ids.append(pretty_alg_id)
    
    mc_eval_df_dict = {
        "num_trials": num_trialss,
        "mc_value_estimate": values,
        # "log_abs_mc_value_estimate": log_ys,
        "algorithm_id": alg_ids,
        "bias_or_temp": bias_or_temps,
        "pretty_alg_id": pretty_alg_ids,
        "replicates": replicates,
    }
    df = pd.DataFrame(mc_eval_df_dict)

    if num_trials_truncate is not None:
        df = df[df["num_trials"] <= num_trials_truncate]

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





    #
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
            plot_filename="plots/003_fl8_hps_ments_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000)

    if "003_fl8_hps_dents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:  
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/001_fl8_hps/dents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_fl8_hps_dents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000)

    if "003_fl8_hps_est_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:    
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/001_fl8_hps/est/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_fl8_hps_est_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000)
        
    if "003_fl8_hps_rents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/001_fl8_hps/rents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_fl8_hps_rents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000)

    if "003_fl8_hps_tents_01" in sys.argv or "all" in sys.argv or expr_id in sys.argv:
        filenames = glob.glob("results/frozen_lake_env/FL_8x8/001_fl8_hps/tents/eval_*.csv")
        make_plot(
            filenames=filenames,
            plot_filename="plots/003_fl8_hps_tents_01.png",
            hue_key="bias_or_temp",
            num_trials_truncate=150000)



