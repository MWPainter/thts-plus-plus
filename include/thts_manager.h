#pragma once

#include "default_dict.h"
#include "helper.h"
#include "thts_env.h"
#include "thts_types.h"

#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <thread>
#include <vector>


namespace thts {
    // Forward declare
    class ThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     * 
     * Member variables (that are different to RandManager/ThtsManager):
     *      seed:
     *          An integer seed to use for random number generation. Default of zero uses a 'random device' to generate 
     *          a seed 
     *      <for others see ThtsManager class definition>
     */
    struct ThtsManagerArgs {
        static const int max_depth_default = std::numeric_limits<int>::max();
        // static const HeuristicFnPtr heuristic_fn_default = helper::zero_heuristic_fn;
        // static const PriorFnPtr prior_fn_default = nullptr;
        static const bool mcts_mode_default = true;
        static const bool is_two_player_game_default = false;
        static const bool use_transposition_table_default = false;
        static const int seed_default = 0;
        static const int num_threads_default = 1;
        static const int num_envs_default = 1;
        
        std::shared_ptr<ThtsEnv> thts_env;
        int num_threads;
        int num_envs;
        int max_depth;
        HeuristicFnPtr heuristic_fn;
        PriorFnPtr prior_fn;

        bool mcts_mode;
        bool is_two_player_game;
        bool use_transposition_table;

        int seed;

        ThtsManagerArgs(std::shared_ptr<ThtsEnv> thts_env) :
            thts_env(thts_env),
            num_threads(num_threads_default),
            num_envs(num_envs_default),
            max_depth(max_depth_default),
            heuristic_fn(helper::zero_heuristic_fn),
            prior_fn(nullptr),
            mcts_mode(mcts_mode_default),
            is_two_player_game(is_two_player_game_default),
            use_transposition_table(use_transposition_table_default),
            seed(seed_default) {}

        virtual ~ThtsManagerArgs() = default;
    };

    /**
     * Rand Manager. A manager for random number generation.
     * 
     * This class manages any random number generation needed, and is used as a base class for algorithm managers.
     * 
     * Member variables:
     *      rng_lock:
     *          A mutex to protect random number generation function calls.
     *      rd:
     *          A 'random_device' which is the computers source of (psuedo) random numbers
     *      int_gen:   
     *          A 'mersenne_twister_engine' used to seed the uniform [0,1) random number generation
     *      real_gen:   
     *          A 'mersenne_twister_engine' used to seed the uniform [0,1) random number generation
     *      int_distr: 
     *          A random number generator for integer numbers in the range [0,RAND_MAX)
     *      real_distr: 
     *          A random number generator for real numbers in the range [0,1)
     */
    class RandManager { 
        protected:
            std::mutex rng_lock;
            std::random_device rd;
            std::mt19937 int_gen;
            std::mt19937 real_gen;
            std::uniform_int_distribution<int> int_distr;
            std::uniform_real_distribution<double> real_distr;

            void init_random_seed() {
                int_gen = std::mt19937(rd());
                real_gen = std::mt19937(rd());
            }
        
        public:
            /**
             * Seed is used to set the seed for cstdlib's rand(), and for the uniform random number generator. If the 
             * seed is set to 0, then we use a std::random_device object to generate a random seed.
            */
            RandManager(const int seed=ThtsManagerArgs::seed_default) :
                rng_lock(),
                int_gen(seed),
                real_gen(seed),
                int_distr(0,RAND_MAX),
                real_distr(0.0,1.0) 
            {
                if (seed == 0) init_random_seed();
            }

            /**
             * Returns a uniform random integer in the range [min_included, max_excluded).
             * N.B. Marked virtual so that these functions can be mocked easily.
             */
            virtual int get_rand_int(int min_included, int max_excluded) {
                std::lock_guard<std::mutex> lg(rng_lock);
                int len = int_distr(int_gen) % (max_excluded - min_included);
                return min_included + len;
            };
            
            /**
             * Returns a uniform random number in the range [0,1).
             * N.B. Marked virtual so that these functions can be mocked easily.
             */
            virtual double get_rand_uniform() {
                std::lock_guard<std::mutex> lg(rng_lock);
                return real_distr(real_gen);
            };
    };
    
    /**
     * ThtsManager is an object used to manage all the things that need to be 'global' space within Thts.
     * 
     * Primarily a thts manager stores all of the options that thts can be run with (see options section below).
     * 
     * As part of managing the 'global' space, the manager is responsible for storing the transposition table, 
     * although interaction with them is defined in thts_decision_node.cpp and thts_chance_node.cpp. Additionally it 
     * stores the pointers to the heuristic and prior function pointers that can be used by Thts algorithms. 
     * 
     * Additionally, the thts manager is used to wrap any random number generation required, to provide a simple 
     * interface to node classes, without having to construct a random number generator in every node.
     * 
     * TODO: want to pass context into ThtsEnv's 'is_sink_state' function.
     * But is_sink_state called from Decision Node constructor
     * So when making root node it is called
     * Before any ThtsEnvContexts can be registered
     * My dirty solution = use a default dict
     * HOWEVER, this requires any implementing thts env's for the moment to ignore the context
     * OPTIONS
     * 1. remove context from is_sink_state in ThtsEnv
     * 2. think about how could avoid this cleaner
     * 
     * Options:
     *      mcts_mode:
     *          If true, trials end at the first leaf node added to the search tree. If false, trials alway run until 
     *          the search horizon is reached or a sink state in the environment is reached, and all nodes are added 
     *          to the search tree.
     *      transposition_table:
     *          Specifies if a transposition table is to be used. Nodes are stored in a table upon creation, keyed by 
     *          (depth, Observation) tuples. When creating a new node, we first look if it exists in the table 
     *          already, and if it does we return that instead. This requires State and Action objects to have 
     *          std::hash and std::equal_to definitions. NOTE: should only use transposition_table if the 
     *          (depth,Observation) tuples have a one to one correspondance with decision nodes, otherwise this may 
     *          cause bugs.
     *      is_two_player_game:
     *          Specifies if we are planning for a two player game
     * 
     * Member variables (environment):
     *      thts_envs:
     *          A vector of ThtsEnv object that provides the dynamics of the environment to plan in
     *      num_threads:
     *          The number of threads to use in the ThtsPool
     *      num_envs:
     *          The number of ThtsEnv objectes in the thts_envs vector 
     *      max_depth:
     *          The maximum depth that we want to allow our thts to search to.
     *      heuristic_fn:
     *          A pointer to the heuristic function to use. Defaults to return a constant zero value.
     *      prior_fn:
     *          A pointer to the prior function, that returns a map representing a policy. Defaults to nullptr to 
     *          indicate no prior. Prior may be able to be unormalised depending on the algorithm being used.
     * Member variables (options):
     *      mcts_mode:
     *          If mcts_mode is true, then only one node is added per trial (and initialised using the heuristic 
     *          function). If mcts_mode is false, then trials are run to completion (until max depth or a sink state is 
     *          reached).
     *      use_transposition_table:
     *          Specifies if a transposition table is to be used. Nodes are stored in a table upon creation, keyed by 
     *          (depth, Observation) tuples. When creating a new node, we first look if it exists in the table 
     *          already, and if it does we return that instead. This requires State and Action objects to have 
     *          std::hash and std::equal_to definitions. NOTE: should only use transposition_table if the 
     *          (depth,Observation) tuples have a one to one correspondance with decision nodes, otherwise this may 
     *          cause bugs.
     *      is_two_player_game:
     *          If we are planning for a two player game, rather than a reward maximisation environment
     * Member variables (transposition table):
     *      dmap:
     *          A transposition table for decision nodes. Note that a transposition table for chance nodes is 
     *          unnecessary, as an chance nodes that are transpositions, will be children of decision nodes that are 
     *          transpositions.
     *      dmap_mutexes:
     *          A vector of mutexes to use for protection around the dmap. Accessing 'dmap[dnode_id]', should be 
     *          protected by the 'dmap_mutexes[hash(dnode_id) % dmap_mutexes.size()]'.
     * Member variables (thread_id_map):
     *      thread_id_map_lock:
     *          A shared lock to protect accesses to the thread_id_map, 
     *          used as reader/writer lock (shared_lock/unique_lock)
     *      thread_id_map:
     *          A mapping from C++ std::thread::id to the thts thread id given at spawn (from 0,...,num_threads-1)
     *          (Sometimes its a bit annoying to pass through a million function calls)
     * Member variables (thts_context_map):
     *      thts_context_map_lock:
     *          A shared lock to protect accesses to the thts_context_map, 
     *          used as reader/writer lock (shared_lock/unique_lock)
     *      thts_context_map:
     *          A mapping from thts thread id to the current ThtsEnvContext for the trial it is running
     */
    class ThtsManager : public RandManager {
        protected:
            std::vector<std::shared_ptr<ThtsEnv>> thts_envs;
        public:
            int num_threads;
            int num_envs;
            int max_depth;
            HeuristicFnPtr heuristic_fn;
            PriorFnPtr prior_fn;

            bool mcts_mode;
            bool use_transposition_table;
            bool is_two_player_game;

            std::shared_mutex dmap_lock;
            DNodeTable dmap;

            std::shared_mutex thread_id_map_lock;
            std::unordered_map<std::thread::id, int> thread_id_map;

            std::shared_mutex thts_context_map_lock;
            thts::helper::unordered_map_with_default<int,std::shared_ptr<ThtsEnvContext>> thts_context_map;

            /**
             * Constructor. Initialises values directly other than random number generation.
             */    
            ThtsManager(const ThtsManagerArgs& args);

            /**
             * Get a thts env for the calling thread from the vector
            */
            std::shared_ptr<ThtsEnv> thts_env();
            std::shared_ptr<ThtsEnv> thts_env(int tid);

            /**
             * Register the thread calling this function with thts thread id 'tid' (called in thts.cpp worker_fn)
            */
            void register_thread_id(int tid);

            /**
             * Get the thts thread id of the current thread
            */
            int get_thts_thread_id();

            /**
             * Register the thts context 'ctx' with thts thread id 'tid'
            */
            void register_thts_context(int tid, std::shared_ptr<ThtsEnvContext> ctx);

            /**
             * Gets the current thts context for the calling thread
            */
            std::shared_ptr<ThtsEnvContext> get_thts_context();
            std::shared_ptr<ThtsEnvContext> get_thts_context(int tid);

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~ThtsManager() = default;
    };
}