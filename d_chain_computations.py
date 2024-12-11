"""
Code to read in the moves made in a game of go, and print images of the board after every move, including the areas
used for the scoring.

N.B. need to make sure that "external/katago/python" is in the $PYTHONPATH bash variable for this code to work.
"""

import numpy as np

def q_one_one(D, alpha, R_f):
    """Computes value of taking action 1 at initial state of D-chain"""
    value = R_f
    for i in range(D-1):
        value = alpha * np.log( np.exp(value/alpha) + np.exp(float(i)/D/alpha) )
    return value


if __name__ == "__main__":
    print("Value of taking action 2 at init state in 10-chain: 0.9")

    for i in range(100):
        alpha = float(i+1)/100
        R_f = 0.8
        q = q_one_one(10, alpha, R_f)
        print("Soft value of Q(1,1) 10-chain with temp of {alpha} and R_f={R_f} is: {q}".format(alpha=alpha, R_f=R_f, q=q))
              
    for i in range(100):
        alpha = float(i+1)/100
        R_f = 0.5
        q = q_one_one(10, alpha, R_f)
        print("Soft value of Q(1,1) 10-chain with temp of {alpha} and R_f={R_f} is: {q}".format(alpha=alpha, R_f=R_f, q=q))


    print("Value of taking action 2 at init state in 20-chain: 0.95")
    for i in range(100):
        alpha = float(i+1)/100
        R_f = 0.8
        q = q_one_one(20, alpha, R_f)
        print("Soft value of Q(1,1) 20-chain with temp of {alpha} and R_f={R_f} is: {q}".format(alpha=alpha, R_f=R_f, q=q))
              
    for i in range(100):
        alpha = float(i+1)/100
        R_f = 0.5
        q = q_one_one(20, alpha, R_f)
        print("Soft value of Q(1,1) 20-chain with temp of {alpha} and R_f={R_f} is: {q}".format(alpha=alpha, R_f=R_f, q=q))
