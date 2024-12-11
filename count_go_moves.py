"""
Counting num moves made
"""

import os
import sys
import glob
from collections import defaultdict



def count(match_csv_filename, avg_num_trials_dict, num_moves_dict, avg_num_trials_dict_pre_20, num_moves_dict_pre_20, avg_num_trials_dict_pre_50, num_moves_dict_pre_50):
    """Takes the filename of a match csv and reads the moves from it."""
    # Read in the file and remove the header from it, with a sanity check
    file_lines = []
    with open(match_csv_filename) as match_file:
        file_lines = match_file.readlines()
    if file_lines[1].split(",")[3] != "alg" or file_lines[1].split(",")[4] != "num_trials":
        raise Exception("Match file not in format expected")
    file_lines = file_lines[2:]

    # Parse each line and update averages
    for line in file_lines:
        csvs = line.split(",")
        move = int(csvs[0])
        alg_id = csvs[3]
        num_trials = float(csvs[4])

        num_moves_dict[alg_id] = num_moves_dict[alg_id] + 1
        avg_num_trials_dict[alg_id] += (num_trials - avg_num_trials_dict[alg_id]) / num_moves_dict[alg_id]

        if move < 50:
            num_moves_dict_pre_50[alg_id] = num_moves_dict_pre_50[alg_id] + 1
            avg_num_trials_dict_pre_50[alg_id] += (num_trials - avg_num_trials_dict_pre_50[alg_id]) / num_moves_dict_pre_50[alg_id]

        if move < 20:
            num_moves_dict_pre_20[alg_id] = num_moves_dict_pre_20[alg_id] + 1
            avg_num_trials_dict_pre_20[alg_id] += (num_trials - avg_num_trials_dict_pre_20[alg_id]) / num_moves_dict_pre_20[alg_id]



if __name__ == "__main__":
    # get the args
    if len(sys.argv) != 2:
        raise Exception("Expected one arg, a base directory")
    
    expr_dir = sys.argv[1]
    regex = os.path.join(expr_dir, "*", "match_*.csv")
    match_filenames = glob.glob(regex)

    avg_num_trials_dict = defaultdict(float)
    num_moves_dict = defaultdict(float)
    avg_num_trials_dict_pre_20 = defaultdict(float)
    num_moves_dict_pre_20 = defaultdict(float)
    avg_num_trials_dict_pre_50 = defaultdict(float)
    num_moves_dict_pre_50 = defaultdict(float)
    for filename in match_filenames:
        count(filename, avg_num_trials_dict, num_moves_dict, avg_num_trials_dict_pre_20, num_moves_dict_pre_20, avg_num_trials_dict_pre_50, num_moves_dict_pre_50)
    
    print("Avg num moves:")
    print(avg_num_trials_dict)
    print()
    print("Num moves made:")
    print(num_moves_dict)
    print()    

    print("Avg num moves (pre 20):")
    print(avg_num_trials_dict_pre_20)
    print()
    print("Num moves made (pre 20):")
    print(num_moves_dict_pre_20)
    print()

    print("Avg num moves (pre 50):")
    print(avg_num_trials_dict_pre_50)
    print()
    print("Num moves made (pre 50):")
    print(num_moves_dict_pre_50)
    print()
