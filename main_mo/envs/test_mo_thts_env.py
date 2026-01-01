from py.mo_py_thts_env import MoPyThtsEnv
import random

RIGHT = 0
DOWN = 1


class TestMoThtsEnv(MoPyThtsEnv):
    """
    A MoThtsEnv that will be used throughout testing. We use these tests to test this class and the interface.
    """

    def __init__(self, 
                 walk_len, 
                 wrong_dir_prob=0.0, 
                 add_extra_rewards=False, 
                 new_dir_bonus=0.5, 
                 same_dir_bonus=0.3, 
                 gamma=0.5):
        """
        Constructor for TestMoThtsEnv
        
        Args:
            walk_len: Length of the walk
            wrong_dir_prob: Probability of going in the wrong direction
            add_extra_rewards: Whether to add extra rewards (4D vs 2D reward)
            new_dir_bonus: Bonus for changing direction
            same_dir_bonus: Bonus for continuing in same direction
            gamma: Discount factor for extra rewards
        """
        reward_dim = 4 if add_extra_rewards else 2
        super().__init__(reward_dim=reward_dim, fully_observable=True)
        self.walk_len = int(walk_len)
        self.wrong_dir_prob = float(wrong_dir_prob)
        self.add_extra_rewards = bool(add_extra_rewards)
        self.new_dir_bonus = float(new_dir_bonus)
        self.same_dir_bonus = float(same_dir_bonus)
        self.gamma = float(gamma)

    def get_gamma(self):
        """Returns the discount factor gamma"""
        return self.gamma

    def get_x(self, state):
        """Returns the x coordinate from state"""
        return state[0]

    def get_y(self, state):
        """Returns the y coordinate from state"""
        return state[1]

    def get_last_direction(self, state):
        """Returns the last direction from state"""
        return state[2]

    def get_initial_state(self):
        """Returns the initial state of the environment"""
        return (0, 0, -1)

    def is_sink_state(self, state):
        """Returns if 'state' is a sink state"""
        return (self.get_x(state) + self.get_y(state)) == self.walk_len

    def get_valid_actions(self, state):
        """Returns a list of valid action objects that can be taken from 'state'"""
        if self.is_sink_state(state):
            return []
        return [RIGHT, DOWN]

    def make_candidate_next_state(self, state, action, wrong_dir):
        """
        Creates a candidate next state given current state, action, and whether to go wrong direction
        
        Args:
            state: Current state tuple (x, y, last_direction)
            action: Action to take (RIGHT or DOWN)
            wrong_dir: Whether to go in the wrong direction
        
        Returns:
            New state tuple
        """
        x, y, _ = state
        direction = action
        if wrong_dir:
            direction = 1 - direction

        if direction == RIGHT:
            x += 1
        elif direction == DOWN:
            y += 1
        
        return (x, y, direction)

    def get_transition_distribution(self, state, action):
        """
        Returns a dictionary mapping from next states to their transition probabilities
        
        Args:
            state: Current state tuple
            action: Action to take
        
        Returns:
            Dictionary mapping next states to probabilities
        """
        new_state = self.make_candidate_next_state(state, action, False)
        transition_distribution = {new_state: 1.0 - self.wrong_dir_prob}
        
        if self.wrong_dir_prob > 0.0:
            stay_state = self.make_candidate_next_state(state, action, True)
            transition_distribution[stay_state] = self.wrong_dir_prob
        
        return transition_distribution

    def sample_transition_distribution(self, state, action):
        """
        Samples a 'next_state' object from Pr('next_state'|'state','action') and returns it
        
        Args:
            state: Current state tuple
            action: Action to take
        
        Returns:
            Sampled next state tuple
        """
        if self.wrong_dir_prob > 0.0:
            sample = random.random()
            if sample < self.wrong_dir_prob:
                return self.make_candidate_next_state(state, action, True)
        
        return self.make_candidate_next_state(state, action, False)

    def get_mo_reward(self, state, action):
        """
        Returns the multi-objective reward for a given state and action
        
        Args:
            state: Current state tuple
            action: Action taken
        
        Returns:
            List of reward values (2D or 4D depending on add_extra_rewards)
        """
        if self.add_extra_rewards:
            r = [0.0] * 4
        else:
            r = [0.0] * 2
        
        r[RIGHT] = -1.0
        r[DOWN] = -1.0
        
        # Add bonus in r[dir], and add more if dir is different to last action
        if self.get_last_direction(state) == action:
            r[action] += self.same_dir_bonus
        else:
            r[action] += self.new_dir_bonus
        
        if not self.add_extra_rewards:
            return r
        
        if action == RIGHT:
            r[2] = self.gamma ** self.get_x(state)
        elif action == DOWN:
            r[3] = self.gamma ** self.get_y(state)
        
        return r

    def get_reward(self, state, action):
        """
        Alias for get_mo_reward to match the Python interface
        """
        return self.get_mo_reward(state, action)

