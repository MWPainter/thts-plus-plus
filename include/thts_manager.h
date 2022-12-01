#pragma once

#include "helper.h"
#include "thts_env.h"
#include "thts_types.h"

#include <cstdlib>
#include <limits>
#include <memory>
#include <mutex>
#include <random>
#include <vector>


namespace thts {
    // Forward declare
    class ThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct ThtsManagerArgs {
        static const int max_depth_default = std::numeric_limits<int>::max();
        // static const HeuristicFnPtr heuristic_fn_default = helper::zero_heuristic_fn;
        // static const PriorFnPtr prior_fn_default = nullptr;
        static const bool mcts_mode_default = true;
        static const bool is_two_player_game_default = false;
        static const bool use_transposition_table_default = false;
        static const int num_transposition_table_mutexes_default = 1;
        static const int seed_default = 60415;
        
        std::shared_ptr<ThtsEnv> thts_env;
        int max_depth;
        HeuristicFnPtr heuristic_fn;
        PriorFnPtr prior_fn;
        bool mcts_mode;
        bool is_two_player_game;
        bool use_transposition_table;
        int num_transposition_table_mutexes;
        int seed;

        ThtsManagerArgs(std::shared_ptr<ThtsEnv> thts_env) :
            thts_env(thts_env),
            max_depth(max_depth_default),
            heuristic_fn(helper::zero_heuristic_fn),
            prior_fn(nullptr),
            mcts_mode(mcts_mode_default),
            is_two_player_game(is_two_player_game_default),
            use_transposition_table(use_transposition_table_default),
            num_transposition_table_mutexes(num_transposition_table_mutexes_default),
            seed(seed_default) {}

        virtual ~ThtsManagerArgs() = default;
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
     * Options:
     *      mcts_mode:
     *          If mcts_mode is true, then only one node is added per trial (and initialised using the heuristic function). 
     *          If mcts_mode is false, then trials are run to completion (until max depth or a sink state is reached).
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
     * Member variables:
     *      thts_env:
     *          A ThtsEnv object that provides the dynamics of the environment being planned in
     *      dmap:
     *          A transposition table for decision nodes
     *      dmap_mutexes:
     *          A vector of mutexes to use for protection around the dmap. Accessing 'dmap[dnode_id]', should be 
     *          protected by the 'dmap_mutexes[hash(dnode_id) % dmap_mutexes.size()]'.
     *      max_depth:
     *          The maximum depth that we want to allow our thts to search to.
     *      mcts_mode:
     *          If running in mcts_mode (see options above)
     *      use_transposition_table:
     *          If using the transposition tables (i.e. dmap and cmap)
     *      is_two_player_game:
     *          If we are planning for a two player game, rather than a reward maximisation environment
     *      heuristic_psuedo_trials:    
     *          The number of 'psuedo trials' to weight the heuristic_fn by. Typically an thts implementation would set 
     *          the number of visits in the node to 
     *      heuristic_fn_ptr:
     *          A pointer to the heuristic function
     *      prior_fn_ptr:
     *          A pointer to the prior (Q-value) function, that returns a map from actions to prior value estimates
     * Private member variables:
     *      rng_lock:
     *          A mutex to protect random number generation function calls.
     *      rd:
     *          A 'random_device' which is the computers source of (psuedo) random numbers
     *      real_gen:   
     *          A 'mersenne_twister_engine' used to seed the uniform [0,1) random number generation
     *      real_distr: 
     *          A random number generator for real numbers in the range [0,1)
     *      int_gen:   
     *          A 'mersenne_twister_engine' used to seed the uniform [0,1) random number generation
     *      int_distr: 
     *          A random number generator for integer numbers in the range [0,RAND_MAX)
     */
    class ThtsManager {
        private:
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
            std::shared_ptr<ThtsEnv> thts_env;

            DNodeTable dmap;
            std::vector<std::mutex> dmap_mutexes;

            int max_depth;
            bool mcts_mode;
            bool use_transposition_table;
            bool is_two_player_game;

            HeuristicFnPtr heuristic_fn;
            PriorFnPtr prior_fn;

            /**
             * Constructor. Initialises values directly other than random number generation.
             * 
             * Seed is used to set the seed for cstdlib's rand(), and for the uniform random number generator. If the 
             * seed is set to 0, then we use a std::random_device object to generate a random seed.
             */    
            ThtsManager(ThtsManagerArgs& args) : 
                rng_lock(),
                int_gen(args.seed),
                real_gen(args.seed),
                int_distr(0,RAND_MAX),
                real_distr(0.0,1.0),
                thts_env(args.thts_env),
                dmap(),
                dmap_mutexes(args.num_transposition_table_mutexes),
                max_depth(args.max_depth),
                mcts_mode(args.mcts_mode), 
                use_transposition_table(args.use_transposition_table), 
                is_two_player_game(args.is_two_player_game),
                heuristic_fn(args.heuristic_fn),
                prior_fn(args.prior_fn)
            {
                if (args.seed == 0) init_random_seed();
            }

            ThtsManager(
                std::shared_ptr<ThtsEnv> thts_env,
                int max_depth=ThtsManagerArgs::max_depth_default,
                HeuristicFnPtr heuristic_fn=helper::zero_heuristic_fn,
                PriorFnPtr prior_fn=nullptr,
                bool mcts_mode=ThtsManagerArgs::mcts_mode_default, 
                bool is_two_player_game=ThtsManagerArgs::is_two_player_game_default,
                bool use_transposition_table=ThtsManagerArgs::use_transposition_table_default, 
                int num_transposition_table_mutexes=ThtsManagerArgs::num_transposition_table_mutexes_default,
                int seed=ThtsManagerArgs::seed_default) :
                    rng_lock(),
                    int_gen(seed),
                    real_gen(seed),
                    int_distr(0,RAND_MAX),
                    real_distr(0.0,1.0),
                    thts_env(thts_env),
                    dmap(),
                    dmap_mutexes(num_transposition_table_mutexes),
                    max_depth(max_depth),
                    mcts_mode(mcts_mode), 
                    use_transposition_table(use_transposition_table), 
                    is_two_player_game(is_two_player_game),
                    heuristic_fn(heuristic_fn),
                    prior_fn(prior_fn)
            {
                if (seed == 0) init_random_seed();
            };

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~ThtsManager() = default;

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
}