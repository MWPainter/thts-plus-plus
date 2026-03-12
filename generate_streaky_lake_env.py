"""
Generate streaky frozen lake envs:
- Even columns (0, 2, 4, ...): all tiles are frozen.
- Odd columns (1, 3, 5, ...): each tile is a hole with probability (1-p), frozen with probability p;
  each odd column is forced to have at least one frozen tile so a path can exist.
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
    """Generates a random valid streaky map (path from start to goal).

    Even columns (0, 2, 4, ...): every tile is frozen.
    Odd columns (1, 3, 5, ...): each tile is frozen with probability p, hole otherwise;
    each odd column is guaranteed to have at least one frozen tile.

    Args:
        size: (rows, cols) of the grid
        p: probability that a tile in an odd column is frozen
    Returns:
        A random valid map as a list of strings
    """
    p = min(1.0, max(0.0, p))
    valid = False
    board = None

    while not valid:
        board = np.empty((size[0], size[1]), dtype="U1")
        for c in range(size[1]):
            if c % 2 == 0:
                board[:, c] = "F"
            else:
                for r in range(size[0]):
                    board[r, c] = np.random.choice(["F", "H"], p=[p, 1 - p])
                if np.all(board[:, c] == "H"):
                    r = np.random.randint(0, size[0])
                    board[r, c] = "F"
        board[0, 0] = "S"
        board[size[0] - 1, size[1] - 1] = "G"
        valid = is_valid(board, size)
    return ["".join(row) for row in board]

if __name__ == "__main__":
    import sys
    env_size = (int(sys.argv[1]), int(sys.argv[2]))
    print(env_size)
    prob = float(sys.argv[3])
    map = generate_random_map(env_size, prob)
    for s in map:
        print(s)