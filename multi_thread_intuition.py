"""

"""


import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd




def non_stationary_reward(visit_count):
    if visit_count < 100:
        return 0
    if visit_count > 500:
        return 2
    return (visit_count - 100) / 200.0



def boltz_sim():
    counts = [0,0]
    rewards = [0,1]

    probs = [np.exp(rewards[j]) / (np.exp(0) + np.exp(1)) for j in range(2)]
    expects = [probs[0]*1000.0, probs[1]*1000.0]
    print("(BOLTZ) Expected pull counts per thousand pulls were: {cnts}".format(cnts=expects))



def uct_sim():
    counts = [0.0,0.0]
    rewards = [0.0,1.0]


    for k in range(5):
        local_counts = [0,0]
        for i in range(1000):
            total_pulls = counts[0] + counts[1]
            ucb = [rewards[j] + 2.0 * np.sqrt(np.log(total_pulls+1) / (counts[j]+1)) for j in range(2)]
            if ucb[0] >= ucb[1]:
                counts[0] += 1
                local_counts[0] += 1
            else:
                counts[1] += 1
                local_counts[1] += 1
        
        print("(UCT) Pull counts for first {k} thousand pulls were: {cnts}".format(k=k+1, cnts=local_counts))




def uct_sim_non_stationary():
    """N.B. this is even wrong, it uses UCB around the true value of the arms, rather than the average return..."""
    counts = [0.0,0.0]
    rewards = [0.0,1.0]

    for i in range(15000):
        total_pulls = counts[0] + counts[1]
        ucb = [rewards[j] + 2.0 * np.sqrt(np.log(total_pulls+1) / (counts[j]+1)) for j in range(2)]
        if ucb[0] >= ucb[1]:
            counts[0] += 1
        else:
            counts[1] += 1
        
    print("(UCT) Pull counts for non stationary (15000 trials) were: {cnts}".format(cnts=counts))



def compute_expected_num_pulls_bts(rewards=None, num_trials=10000):
    rewards = np.array([0.0, 1.0])
    probs = np.array([np.exp(rewards[j]) / (np.exp(0) + np.exp(1)) for j in range(2)])
    counts = np.zeros((num_trials+1, len(rewards)))
    for i in range(1,num_trials+1):
        counts[i] = counts[i-1] + probs
    return counts

def compute_expected_num_pulls_uct(rewards=None, bias=10.0, num_trials=10000):
    rewards = np.array([0.0, 1.0])
    counts = np.zeros((num_trials+1, len(rewards)))
    counts[1] = np.array([1.0, 0.0])
    counts[2] = np.array([1.0, 1.0])
    for i in range(3,num_trials+1):
        counts[i] = counts[i-1]
        total_pulls = counts[i,0] + counts[i,1]
        ucb = [rewards[j] + bias * np.sqrt(np.log(total_pulls+1) / (counts[i,j]+1)) for j in range(2)]
        if ucb[0] >= ucb[1]:
            counts[i,0] += 1
        else:
            counts[i,1] += 1
    return counts

def plot_expected_num_pulls(bts_counts, uct_counts, num_trials=10000):
    counts = []
    trials = []
    algo = []
    for i in range(num_trials+1):
        counts.append(bts_counts[i])
        trials.append(i)
        algo.append("BTS")
    for i in range(num_trials+1):
        counts.append(uct_counts[i])
        trials.append(i)
        algo.append("UCT")
    df = pd.DataFrame({
        "arm_pulls": counts,
        "trials": trials,
        "algo": algo,
    })
    sns.lineplot(
        data=df,
        x="trials",
        y="arm_pulls",
        hue="algo",
    )
    # plt.rcParams['text.usetex'] = True
    plt.xlabel("Trials")
    # plt.ylabel(r"\mathbb{E}\[N(s,a_0)\]")
    # plt.ylabel("E[N(s,a_0)]")
    plt.legend(title="Algorithm")
    plt.show()

def main_plot():
    bts_countss = compute_expected_num_pulls_bts()
    uct_countss = compute_expected_num_pulls_uct()
    plot_expected_num_pulls(bts_counts=bts_countss[:,0], uct_counts=uct_countss[:,0])



if __name__ == "__main__":
    # uct_sim()
    # boltz_sim()
    # uct_sim_non_stationary()

    main_plot()
