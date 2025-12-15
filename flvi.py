import gym
import numpy as np

# Create the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=None)

# Parameters
gamma = 0.99  # Discount factor
theta = 1e-8  # Convergence threshold

# Value Iteration
def value_iteration(env, gamma=0.99, theta=1e-8):
    V = np.zeros(env.observation_space.n)
    while True:
        delta = 0
        for s in range(env.observation_space.n):
            v = V[s]
            action_values = []
            for a in range(env.action_space.n):
                value = 0
                for prob, next_state, reward, done in env.P[s][a]:
                    value += prob * (reward + gamma * V[next_state])
                action_values.append(value)
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

# Run value iteration
V_opt = value_iteration(env, gamma, theta)

# Get grid size
grid_size = int(np.sqrt(env.observation_space.n))

# Print values in grid format
print("Optimal state values (row, col):")
for s, v in enumerate(V_opt):
    row = s // grid_size
    col = s % grid_size
    print(f"({row}, {col}): {v:.4f}")