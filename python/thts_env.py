"""
The thts environment interface for python.

This can be inherited from or used as a template to implement the interface. 
"""


class ThtsEnv:
    def __init__(self):
        """
        Constructor.
        """
        pass
    
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
        pass
    
    def get_valid_actions(self, state):
        """
        Returns a list of valid actions that can be taken from 'state'.

        Args:
            state: A valid state for the ThtsEnv object.
        Returns:
            A list of valid actions that can be taken from 'state'.
        """
        pass

    def get_transition_distribution(self, state, action):
        """
        Returns the transition distribution for taking 'action' from 'state'.

        Args:
            state: A valid state for the ThtsEnv object.
            action: A valid action for the ThtsEnv object.
        Returns:
            A dictionary mapping from state objects to their probability of being the next state.
        """
        pass

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
        pass

    def get_observation_distribution(self, state, action):
        """
        Returns the distribution of possible observations for taking 'action' from 'state'.

        Args:
            state: A valid state for the ThtsEnv object.
            action: A valid action for the ThtsEnv object.
        Returns:
            A dictionary mapping from observation objects to their probability of being observed.
        """
        pass

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
        pass

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
        pass

    def heuristic_fn(self, state):
        """
        Returns a heuristic value of V(s).

        *TODO* More docstring
        """
        pass

    def prior_policy(self, state):
        """
        Returns a dictionary, mapping from actions to probabilities (or weights) as a prior policy from 'state'.

        *TODO* More docstring
        """
        pass

    def sample_context(self, state):
        """
        Sample a context

        *TODO* More docstring
        """
        return {}