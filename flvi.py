"""
Value iteration for FrozenLake. Supports any map defined in main_aux/envs/frozen_lake.h.
Usage: python flvi.py [--map MAP_NAME] [--slippery]
"""
import argparse
import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym

# Maps from main_aux/envs/frozen_lake.h (same layout: S=Start, F=Frozen, H=Hole, G=Goal)
MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
    "6x6_no_hole": [
        "SFFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFFF",
        "FFFFFG",
    ],
    "gen_5x5": [
        "SFFFF",
        "HFFFH",
        "HHFFF",
        "FFFFF",
        "FFFHG",
    ],
    "gen_6x6": [
        "SFFFFF",
        "FHFHFF",
        "FFFFFF",
        "FFHFFF",
        "FFFFHH",
        "FFFFFG",
    ],
    "gen_4x8": [
        "SFFFHFFF",
        "FHFFFFFF",
        "FHFFHFFF",
        "FFFFFFHG",
    ],
    "gen_4x12": [
        "SFFHFFFFFFFH",
        "FFFFFHFFFFFF",
        "FFFFHFFFFFFF",
        "FFFFFFFFFFHG",
    ],
    "gen_8x16": [
        "SFFFFFFFFFFFFFHF",
        "FHHFFFHFHHFFFHFF",
        "FHFFHFFFHFFFFFFF",
        "FFFHFFFHFFFFFHHF",
        "HFFHFHFFFFFFHHFF",
        "FFFHFHFFHHFFFFFF",
        "FFFFFFFFFFFFFFHF",
        "FFFFFHFFHHFFHHFG",
    ],
    "gen_16x16": [
        "SFFHFHFFHFFFFFFF",
        "FFHHFHFFFHFFFHFF",
        "FFFFHFHFHFHHFFFF",
        "FFFHFFFFFHFFFFFF",
        "FFFFHFFFFFFHFHFF",
        "FFFFHFFFHFFFFFFF",
        "FFHHHFHHFHFFHFHH",
        "FFFFFFFFFFFFFFFF",
        "FFFFFFFFFFFFFFHF",
        "HFFFFHHFHFFFHFHF",
        "FFFHFFFFFFFFHFFF",
        "FFFFFFFHFHFFFFFF",
        "FFHHFFHHHHFFFFFF",
        "FFHFFFFHFFFFFFFH",
        "FFFFFFFFFFHFHFFF",
        "FFHFHFHHFFHFFFFG",
    ],
}


def _terminal_states_from_desc(desc):
    """State indices that are goal (G) or hole (H); no future actions."""
    width = len(desc[0])
    terminal = []
    for i, row in enumerate(desc):
        for j, c in enumerate(row):
            if str(c).upper() in ("G", "H"):
                terminal.append(i * width + j)
    return set(terminal)


def value_iteration(env, gamma=0.99, theta=1e-8, time_bound=None):
    """
    Value iteration. If time_bound is None, infinite horizon (original). Else finite horizon:
    states augmented with time t in [0, time_bound]; V(s, t) = 0 for t >= time_bound or s terminal.
    Returns V of shape (n_states,) for infinite horizon, or (n_states, time_bound+1) for finite.
    """
    n_s = env.observation_space.n
    P = env.unwrapped.P

    if time_bound is None:
        V = np.zeros(n_s)
        while True:
            delta = 0
            for s in range(n_s):
                v = V[s]
                action_values = []
                for a in range(env.action_space.n):
                    value = 0
                    for prob, next_state, reward, done in P[s][a]:
                        value += prob * (reward + gamma * V[next_state])
                    action_values.append(value)
                V[s] = max(action_values)
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break
        return V

    # Finite horizon: V[s, t] = value at state s with t steps remaining (t = 0..time_bound)
    T = time_bound
    V = np.zeros((n_s, T + 1))
    terminal = _terminal_states_from_desc(env.unwrapped.desc)

    for t in range(T - 1, -1, -1):
        for s in range(n_s):
            if s in terminal:
                V[s, t] = 0.0
                continue
            best = -np.inf
            for a in range(env.action_space.n):
                value = 0.0
                for prob, next_state, reward, done in P[s][a]:
                    value += prob * (reward + gamma * V[next_state, t + 1])
                best = max(best, value)
            V[s, t] = best

    return V


def main():
    parser = argparse.ArgumentParser(
        description="Value iteration for FrozenLake. Maps match main_aux/envs/frozen_lake.h."
    )
    parser.add_argument(
        "--map",
        choices=list(MAPS.keys()),
        default="4x4",
        metavar="MAP",
        help="Map name (default: 4x4). Choices: %(choices)s",
    )
    parser.add_argument(
        "--slippery",
        action="store_true",
        default=True,
        help="Use slippery (stochastic) dynamics (default: True)",
    )
    parser.add_argument(
        "--no-slippery",
        action="store_false",
        dest="slippery",
        help="Use deterministic dynamics",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=1e-8,
        help="Convergence threshold (default: 1e-8)",
    )
    parser.add_argument(
        "--time-bound",
        type=int,
        default=None,
        metavar="T",
        help="Finite horizon: max steps (states augmented with time). Default: None (infinite horizon).",
    )
    parser.add_argument(
        "--init-state-only",
        action="store_true",
        default=False,
        help="Only print the initial state",
    )
    args = parser.parse_args()

    desc = MAPS[args.map]
    env = gym.make(
        "FrozenLake-v1",
        desc=desc,
        is_slippery=args.slippery,
        render_mode=None,
    )

    time_bound = None if (args.time_bound is None or args.time_bound == 0) else args.time_bound
    V_opt = value_iteration(
        env, gamma=args.gamma, theta=args.theta, time_bound=time_bound
    )

    height, width = len(desc), len(desc[0])

    if V_opt.ndim == 1:
        V_print = V_opt
        time_info = "infinite horizon"
    else:
        V_print = V_opt[:, 0]  # value from each state at t=0 (full horizon)
        time_info = f"time_bound={time_bound}"

    print(f"Map: {args.map} ({height}x{width}), slippery={args.slippery}, {time_info}")
    print("Optimal state values (row, col):")
    for s, v in enumerate(V_print):
        row = s // width
        col = s % width
        if args.init_state_only and (row != 0 or col != 0):
            continue
        print(f"  ({row}, {col}): {v:.4f}")


if __name__ == "__main__":
    main()
