from py_thts_env import PyThtsEnv

import copy
from multiprocessing import Process, Pipe

def server_worker(thts_env, rpc_conn, unused_conn):
    """
    This function runs in a seperate python process
    It will just listen to pipe until EOF
    When it recieves an rpc, it runs it on the env
    """
    unused_conn.close()
    while True:
        try:
            cmd, args = rpc_conn.recv()

            # things called a lot
            if cmd == "is_sink_state":
                rpc_conn.send(thts_env.is_sink_state(*args))
            elif cmd == "get_valid_actions":
                rpc_conn.send(thts_env.get_valid_actions(*args))
            elif cmd == "get_reward":
                rpc_conn.send(thts_env.get_reward(*args))
            elif cmd == "sample_transition_distribution":
                rpc_conn.send(thts_env.sample_transition_distribution(*args))

            elif cmd == "get_transition_distribution":
                rpc_conn.send(thts_env.get_transition_distribution(*args))

            # things called less often
            elif cmd == "sample_context_and_reset":
                rpc_conn.send(thts_env.sample_context_and_reset(*args))
            elif cmd == "get_initial_state":
                rpc_conn.send(thts_env.get_initial_state(*args))
            elif cmd == "get_observation_distribution":
                rpc_conn.send(thts_env.get_observation_distribution(*args))
            elif cmd == "sample_observation_distribution":
                rpc_conn.send(thts_env.sample_observation_distribution(*args))
        
        except EOFError:
            break


class PyThtsEnvServerWrapper(PyThtsEnv):
    """
    A wrapper to get around issues with python 
    - numpy isn't supported by subinterpreters

    This wrapper can be used with subinterpreters
    This wrapper makes a seperate process to actually run the environment in
    """
    def __init__(self, env):
        self.env = env
        self.server_conn, rpc_conn = Pipe()
        self.server_process = Process(target=server_worker, args=(env,rpc_conn,self.server_conn))
        self.server_process.start()
        rpc_conn.close()
    
    def __del__(self):
        if (self.server_conn is not None):
            self.server_conn.close()
        if (self.server_process is not None):
            self.server_process.join()

    def is_fully_observable(self):
        return self.env.fully_observable

    def clone(self):
        """Creates a complete (deep) clone of this environment, for parallell use
        
        Cant deep copy multiprocessing things
        But need deep copy so that each PyThtsEnvServerWrapper is *completely* independent python object
        (including python functions etc)

        - shallow copy so dont destroy 'self'
        - remove refs to multiprocessing in copy
        - deep copy to work with c++
        - start new server process
        """
        new_env_wrapper = copy.copy(self)    
        new_env_wrapper.server_conn, new_env_wrapper.server_process = None, None
        new_env_wrapper = copy.deepcopy(new_env_wrapper)
        server_conn, rpc_conn = Pipe()
        new_env_wrapper.server_conn = server_conn
        new_env_wrapper.server_process = Process(target=server_worker, args=(new_env_wrapper.env,rpc_conn,server_conn))
        new_env_wrapper.server_process.start()
        rpc_conn.close()
        return new_env_wrapper

    def sample_context_and_reset(self, tid):
        """
        Returns a thts context to be used throughout the trial
        Most of the time this will be unused, or a dictionary is sufficient
        """
        cmd = "sample_context_and_reset"
        args = (tid,)
        self.server_conn.send((cmd,args))
        return self.server_conn.recv()

    def get_initial_state(self):
        """
        Returns the initial state of the environment
        """
        cmd = "get_initial_state"
        args = ()
        self.server_conn.send((cmd,args))
        return self.server_conn.recv()

    def is_sink_state(self, state, ctx=None):
        """
        Returns if 'state' is a sink state
        """
        cmd = "is_sink_state"
        args = (state, ctx)
        self.server_conn.send((cmd,args))
        return self.server_conn.recv()

    def get_valid_actions(self, state, ctx=None):
        """
        Returns a list of valid action objects that can be taken from 'state'
        """
        cmd = "get_valid_actions"
        args = (state, ctx)
        self.server_conn.send((cmd,args))
        return self.server_conn.recv()

    def get_transition_distribution(self, state, action, ctx=None):
        """
        Returns a dictionary mapping from next states to their transition probabilities
        This function is optional (but recommended)
            (-recommended as it lets us avoid having to repetatively call 'sample_transition_distribution' and use the 
                python interpreter, which is slow)
        """
        cmd = "get_transition_distribution"
        args = (state, action, ctx)
        self.server_conn.send((cmd,args))
        return self.server_conn.recv()

    def sample_transition_distribution(self, state, action, ctx=None):
        """
        Samples a 'next_state' object from Pr('next_state'|'state','action') and returns it
        """
        cmd = "sample_transition_distribution"
        args = (state, action, ctx)
        self.server_conn.send((cmd,args))
        return self.server_conn.recv()

    def get_observation_distribution(self, state, action, ctx=None):
        """
        Returns a dictionary mapping from observations to their observation probabilities
        In fully observable environments observations are just the next states
        """
        cmd = "get_observation_distribution"
        args = (state, action, ctx)
        self.server_conn.send((cmd,args))
        return self.server_conn.recv()

    def sample_observation_distribution(self, state, action, ctx=None):
        """
        Samples a 'obs' object from Pr('obs'|'state','action') and returns it
        """
        cmd = "sample_observation_distribution"
        args = (state, action, ctx)
        self.server_conn.send((cmd,args))
        return self.server_conn.recv()

    def get_reward(self, state, action, ctx=None):
        """
        Returns reward given a state, action, observation tuple
        """
        cmd = "get_reward"
        args = (state, action, ctx)
        self.server_conn.send((cmd,args))
        return self.server_conn.recv()
    