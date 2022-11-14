#pragma once

#include "helper.h"
#include "thts_types.h"

#include <cstdlib>
#include <memory>
#include <ostream>
#include <random>
#include <tuple>
#include <unordered_map>


namespace thts {
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
     *          (depth, State, optional<Action>) tuples. When creating a new node, we first look if it exists in the table 
     *          already, and if it does we return that instead. This requires State and Action objects to have std::hash and 
     *          std::equal_to definitions.
     *      is_two_player_game:
     *          Specifies if we are planning for a two player game
     * 
     * Member variables:
     *      dmap:
     *          A transposition table for decision nodes
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
     *      gen:   
     *          A 'mersenne_twister_engine' used to see the (uniform) random number generation
     *      uniform_distr: 
     *          A random number generator for real numbers in the range [0,1)
     */
    class ThtsManager {
        private:
            std::mt19937 gen;
            std::uniform_real_distribution<double> uniform_distr;

        public:
            DNodeTable dmap;

            bool mcts_mode = true;
            bool use_transposition_table = false;
            bool is_two_player_game = false;

            HeuristicFnPtr heuristic_fn;
            PriorFnPtr prior_fn;

            /**
             * Constructor. Initialises values directly other than random number generation.
             * 
             * Seed is used to set the seed for cstdlib's rand(), and for the uniform random number generator. If the 
             * seed is set to 0, then we use a std::random_device object to generate a random seed.
             */            
            ThtsManager(
                bool mcts_mode=true, 
                bool use_transposition_table=false, 
                bool is_two_player_game=false,
                HeuristicFnPtr heuristic_fn=helper::zero_heuristic_fn,
                PriorFnPtr prior_fn=nullptr,
                int seed=60415) :
                    gen(seed),
                    uniform_distr(0.0,1.0),
                    mcts_mode(mcts_mode), 
                    use_transposition_table(use_transposition_table), 
                    is_two_player_game(is_two_player_game),
                    heuristic_fn(heuristic_fn),
                    prior_fn(prior_fn)
            {
                if (seed == 0) {
                    std::random_device rd;
                    gen = std::mt19937(rd());
                }
                std::srand(seed);
            };

            /**
             * Returns a uniform random integer in the range [min_included, max_excluded).
             * N.B. Marked virtual so that these functions can be mocked easily.
             */
            virtual int get_rand_int(int min_included, int max_excluded) {
                int len = std::rand() % (max_excluded - min_included);
                return min_included + len;
            };
            
            /**
             * Returns a uniform random number in the range [0,1).
             * N.B. Marked virtual so that these functions can be mocked easily.
             */
            virtual double get_rand_uniform() {
                return uniform_distr(gen);
            };
    };
}