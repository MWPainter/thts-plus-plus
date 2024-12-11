"""
Code to generate frozen lake envs, copied from: 
https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
"""

import numpy as np

def is_valid(board, max_size):
    frontier, discovered = [], set()
    frontier.append((0, 0))
    while frontier:
        r, c = frontier.pop()
        if not (r, c) in discovered:
            discovered.add((r, c))
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= max_size[0] or c_new < 0 or c_new >= max_size[1]:
                    continue
                if board[r_new][c_new] == "G":
                    return True
                if board[r_new][c_new] != "H":
                    frontier.append((r_new, c_new))
    return False


def generate_random_map(size, p):
    """Generates a random valid map (one that has a path from start to goal)
    Args:
        size: size of each side of the grid
        p: probability that a tile is frozen
    Returns:
        A random valid map
    """
    valid = False
    board = []  # initialize to make pyright happy

    while not valid:
        p = min(1, p)
        board = np.random.choice(["F", "H"], (size[0],size[1]), p=[p, 1 - p])
        board[0][0] = "S"
        board[-1][-1] = "G"
        valid = is_valid(board, size)
    return ["".join(x) for x in board]

if __name__ == "__main__":
    import sys
    env_size = (int(sys.argv[1]), int(sys.argv[2]))
    print(env_size)
    prob = float(sys.argv[3])
    map = generate_random_map(env_size, prob)
    for s in map:
        print(s)