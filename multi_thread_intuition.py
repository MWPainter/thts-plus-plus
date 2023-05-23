"""

"""


import numpy as np




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



if __name__ == "__main__":
    uct_sim()
    boltz_sim()
    uct_sim_non_stationary()
