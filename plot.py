import os

import matplotlib
if 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ:
    matplotlib.use('Agg')
from matplotlib import pyplot as plt
import seaborn as sns

import pandas as pd
import sys

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
            alg_id=alg_id).replace("H","_")

    return alg_ids, bias_or_temps, replicates, values, num_trialss

def make_plot(filenames, plot_filename, hue_key=None, title=None, xaxis_lab=None, yaxis_lab=None, legend_lab=None, num_trials_truncate=None):
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

    alg_ids, bias_or_temps, replicates, values, num_trialss = read_eval_files(filenames)    
    
    mc_eval_df_dict = {
        "num_trials": num_trialss,
        "mc_value_estimate": values,
        # "log_abs_mc_value_estimate": log_ys,
        "algorithm_id": alg_ids,
        "bias_or_temp": bias_or_temps,
        "pretty_alg_id": alg_ids,
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

    if sys.argv[1] == "001_01_10chain_uct":
        filenames = [
            "results/dchain_env/10-1.0/001_len_10/uct/eval_bias=-1.csv",
            "results/dchain_env/10-1.0/001_len_10/uct/eval_bias=10.csv",
            "results/dchain_env/10-1.0/001_len_10/uct/eval_bias=3.csv",
            "results/dchain_env/10-1.0/001_len_10/uct/eval_bias=2.csv",
            "results/dchain_env/10-1.0/001_len_10/uct/eval_bias=1.csv",
            "results/dchain_env/10-1.0/001_len_10/uct/eval_bias=0.3.csv",
            "results/dchain_env/10-1.0/001_len_10/uct/eval_bias=0.1.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/{expr_id}.png".format(expr_id=sys.argv[1]),
            hue_key="bias_or_temp",
            num_trials_truncate=1500)



    elif sys.argv[1] == "001_01_10chain_puct":
        filenames = [
            "results/dchain_env/10-1.0/001_len_10/puct/eval_bias=-1.csv",
            "results/dchain_env/10-1.0/001_len_10/puct/eval_bias=3.csv",
            "results/dchain_env/10-1.0/001_len_10/puct/eval_bias=2.csv",
            "results/dchain_env/10-1.0/001_len_10/puct/eval_bias=1.csv",
            "results/dchain_env/10-1.0/001_len_10/puct/eval_bias=0.3.csv",
            "results/dchain_env/10-1.0/001_len_10/puct/eval_bias=0.1.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/{expr_id}.png".format(expr_id=sys.argv[1]),
            hue_key="bias_or_temp",
            num_trials_truncate=1500)



    elif sys.argv[1] == "001_01_10chain_ments":
        filenames = [
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=0.01.csv",
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=0.03.csv",
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=0.1.csv",
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=0.13.csv",
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=0.137.csv",
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=0.138.csv",
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=0.139.csv",
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=0.14.csv",
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=0.2.csv",
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=0.3.csv",
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=1.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/{expr_id}.png".format(expr_id=sys.argv[1]),
            hue_key="bias_or_temp",
            num_trials_truncate=1500)



    elif sys.argv[1] == "001_01_10chain_dbments":
        filenames = [.replace("H","_")
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=0.01.csv",
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=0.03.csv",
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=0.1.csv",
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=0.13.csv",
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=0.137.csv",
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=0.138.csv",
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=0.139.csv",
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=0.14.csv",
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=0.2.csv",
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=0.3.csv",
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=1.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/{expr_id}.png".format(expr_id=sys.argv[1]),
            hue_key="bias_or_temp",
            num_trials_truncate=1500)



    elif sys.argv[1] == "001_01_10chain_dents":
        filenames = [
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=0.01.csv",
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=0.03.csv",
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=0.1.csv",
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=0.13.csv",
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=0.137.csv",
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=0.138.csv",
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=0.139.csv",
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=0.14.csv",
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=0.2.csv",
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=0.3.csv",
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=1.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/{expr_id}.png".format(expr_id=sys.argv[1]),
            hue_key="bias_or_temp",
            num_trials_truncate=1500)



    elif sys.argv[1] == "001_01_10chain_dbdents":
        filenames = [
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=0.01.csv",
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=0.03.csv",
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=0.1.csv",
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=0.13.csv",
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=0.137.csv",
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=0.138.csv",
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=0.139.csv",
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=0.14.csv",
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=0.2.csv",
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=0.3.csv",
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=1.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/{expr_id}.png".format(expr_id=sys.argv[1]),
            hue_key="bias_or_temp",
            num_trials_truncate=1500)



    elif sys.argv[1] == "001_01_10chain_rents":
        filenames = [
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=0.01.csv",
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=0.03.csv",
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=0.1.csv",
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=0.13.csv",
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=0.137.csv",
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=0.138.csv",
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=0.139.csv",
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=0.14.csv",
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=0.2.csv",
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=0.3.csv",
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=1.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/{expr_id}.png".format(expr_id=sys.argv[1]),
            hue_key="bias_or_temp",
            num_trials_truncate=1500)



    elif sys.argv[1] == "001_01_10chain_tents":
        filenames = [
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=0.01.csv",
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=0.03.csv",
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=0.1.csv",
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=0.13.csv",
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=0.137.csv",
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=0.138.csv",
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=0.139.csv",
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=0.14.csv",
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=0.2.csv",
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=0.3.csv",
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=1.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/{expr_id}.png".format(expr_id=sys.argv[1]),
            hue_key="bias_or_temp",
            num_trials_truncate=1500)

    
    
    elif sys.argv[1] == "001_01_10chain_mixed":
        filenames = [
            "results/dchain_env/10-1.0/001_len_10/uct/eval_bias=-1.csv",
            "results/dchain_env/10-1.0/001_len_10/puct/eval_bias=-1.csv",
            "results/dchain_env/10-1.0/001_len_10/ments/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-1.0/001_len_10/db-ments/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-1.0/001_len_10/dents/eval_epsilon=0.1,init_temp=1.csv",
            "results/dchain_env/10-1.0/001_len_10/db-dents/eval_epsilon=0.1,init_temp=1.csv",
            "results/dchain_env/10-1.0/001_len_10/rents/eval_epsilon=0.1,temp=1.csv",
            "results/dchain_env/10-1.0/001_len_10/tents/eval_epsilon=0.1,temp=1.csv",
        ]
        make_plot(
            filenames=filenames,
            plot_filename="plots/{expr_id}.png".format(expr_id=sys.argv[1]),
            num_trials_truncate=1500)




#
#