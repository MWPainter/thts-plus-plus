"""
A script to test using the python thtsenv
"""
import numpy as np

from thts_env import ThtsEnv


class GridEnv(ThtsEnv):
    def __init__(self, grid_size=4, move_prob=1.0):
        """
        Constructor.
        """
        self.grid_size = grid_size
    
    def _setup_for_fork(
        self, 
        sink_state_fork:bool, 
        valid_actions_fork:bool, 
        get_transition_distribution_fork:bool, 
        sample_transition_distribution_fork:bool, 
        get_observation_distribution_fork:bool, 
        sample_observation_fork:bool, 
        get_reward_fork:bool, 
        heuristic_fn_fork:bool,
        prior_policy_fork:bool,
        sample_context_fork:bool):
        """
        Setup for forked versions of the environment.

        Behind the scenes, in the C++ implementation, a fork is created for calling each function in the env, so that 
        each of the functions can be called in parallel and bypass problems with pythons GIL.

        Args:
            *TODO*
        """
        pass

    def get_initial_state(self):
        """
        Returns the initial state of the environment.

        Returns:
            The initial state of the environment.
        """
        return (0,0)

    def is_sink_state(self, state):
        """
        Returns if 'state' is a sink state

        Args:
            state: A valid state for the ThtsEnv object.
        Returns:
            Boolean for if 'state' is a sink state
        """
        return state[0] == self.grid_size-1 and state[1] == self.grid_size-1
    
    def get_valid_actions(self, state):
        """
        Returns a list of valid actions that can be taken from 'state'.

        Args:
            state: A valid state for the ThtsEnv object.
        Returns:
            A list of valid actions that can be taken from 'state'.
        """
        if self.is_sink_state(state):
            return []
        return ["L", "R", "U", "D"]   

    def _next_state(self, state, action):
        next_state = (state[0], state[1])
        if action == "L" and state[0] > 0:
            next_state[0] -= 1
        elif action == "R" and state[0] < self.grid_size-1:
            next_state[0] += 1
        if action == "U" and state[1] > 0:
            next_state[1] -= 1
        elif action == "D" and state[1] < self.grid_size-1:
            next_state[1] += 1
        return next_state



    def get_transition_distribution(self, state, action):
        """
        Returns the transition distribution for taking 'action' from 'state'.

        Args:
            state: A valid state for the ThtsEnv object.
            action: A valid action for the ThtsEnv object.
        Returns:
            A dictionary mapping from state objects to their probability of being the next state.
        """
        possible_next_state = self._next_state(state)
        distr = {possible_next_state: self.move_prob}
        if self.move_prob < 1.0:
            distr[state] = 1.0 - self.move_prob
        return distr
        

    def sample_transition_distribution(self, state, action):
        """
        Returns a new state sampled from taking 'action' from 'state'.

        Args:
            state: A valid state for the ThtsEnv object.
            action: A valid action for the ThtsEnv object.
        Returns:
            The next state from taking 'action' from 'state'. This should be equivalent to sampling from the 
            distribution returned from 'get_transition_distribution'.
        """
        p = np.random.random()
        if p <= self.move_prob:
            return self._next_state(state,action)
        return state


    def get_observation_distribution(self, state, action):
        """
        Returns the distribution of possible observations for taking 'action' from 'state'.

        Args:
            state: A valid state for the ThtsEnv object.
            action: A valid action for the ThtsEnv object.
        Returns:
            A dictionary mapping from observation objects to their probability of being observed.
        """
        return self.get_transition_distribution(state,action)

    def sample_observation_distribution(self, state, action):
        """
        Returns an observation sampled when taking 'action' from 'state'.

        Args:
            state: A valid state for the ThtsEnv object.
            action: A valid action for the ThtsEnv object.
        Returns:
            An observation from taking 'action' from 'state'. This should be equivalent to sampling from the 
            distribution returned from 'get_observation_distribution'.
        """
        return self.sample_transition_distribution(state,action)

    def get_reward(self, state, action, observation=None):
        """
        Returns the reward for taking 'action' from 'state', while observing 'observation'.

        Args:
            state: A valid state for the ThtsEnv object.
            action: A valid action for the ThtsEnv object.
            observation: Optionally, an observation from taking 'action' from 'state'.
        Returns:
            Returns the value of the reward R(s,a).
        """
        return -1

    def heuristic_fn(self, state):
        """
        Returns a heuristic value of V(s).

        *TODO* More docstring
        """
        return 0

    def prior_policy(self, state):
        """
        Returns a dictionary, mapping from actions to probabilities (or weights) as a prior policy from 'state'.

        *TODO* More docstring
        """
        return None



if __name__ == "__main__":
    import thts
    grid_env = GridEnv()
    thts._test_thts_env(env=grid_env)