#pragma once

#include "thts_manager.h"

#include <cstdlib>
#include <random>

namespace thts {
    /**
     * A specific instance of ThtsManager for UCT algorithms.
     * 
     * Options:
     *      using_heuristic_function (heuristic_psuedo_trials > 0):
     *          Uses the heurisitic function to initialise the value of nodes, with an initial weight ('number of 
     *          visits' of 'heuristic_psuedo_trials'). 
     *      using_epsilon_exploration (epsilon_exploration > 0):
     *          Uses uniformly random exploration of actions 'epsilon_exploration' proportion of the time.
     *      recommend_most_visited:
     *          If true then on recommendations return the action corresponding to the child that has been visited the 
     *          most. By default (when false) recommend the child with the best empirical average.
     * 
     * Member variables:
     *      bias:
     *          The bias to use in the ucb values at decision nodes. If set to 'USE_AUTO_BIAS' then an adaptive bias is
     *          used as outlined by the PROST planner (https://www.aaai.org/ocs/index.php/ICAPS/ICAPS12/paper/viewFile/4715/4721).
     *      heuristic_psuedo_trials:
     *          The number of 'psuedo trials' to weight the value of the heuristic functino by. Should be used to 
     *          initialise the 'num_visits' of UCT nodes. A value of zero indicates that the heuristic function should 
     *          be ignored entirely (bool use_heuristic_fn == (heuristic_psuedo_trials == 0)).
     *      use_heuristic_at_chance_nodes:
     *          A boolean specifying if chance nodes values should be initialised using the heuristic function.
     *      epsilon_exploration:
     *          Defines the proportion of time to be spent exploring uniformly randomly. Default set to zero and to 
     *          purely use the primary action selection. Should be in the range [0,1].
     *      recommend_most_visited:
     *          A boolean storing if we are using the 'recommend_most_visited' option above.
     */
    class UctManager : public ThtsManager {
        public:
            static constexpr double USE_AUTO_BIAS = -1.0;
            static constexpr double AUTO_BIAS_MIN_BIAS = 0.001;

            double bias;
            int heuristic_psuedo_trials;
            bool use_heuristic_at_chance_nodes;
            double epsilon_exploration;
            bool recommend_most_visited;

            UctManager(
                double bias=USE_AUTO_BIAS,
                bool mcts_mode=true, 
                bool use_transposition_table=false, 
                bool is_two_player_game=false,
                int heuristic_psuedo_trials=0,
                bool use_heuristic_at_chance_nodes=false,
                double epsilon_exploration=0.0,
                bool recommend_most_visited=false,
                HeuristicFnPtr heuristic_fn=helper::zero_heuristic_fn,
                PriorFnPtr prior_fn=nullptr,
                int seed=60415) :
                    ThtsManager(
                        mcts_mode,
                        use_transposition_table,
                        is_two_player_game,
                        heuristic_fn,
                        prior_fn,
                        seed),
                    bias(bias),
                    heuristic_psuedo_trials(heuristic_psuedo_trials),
                    use_heuristic_at_chance_nodes(use_heuristic_at_chance_nodes),
                    epsilon_exploration(epsilon_exploration),
                    recommend_most_visited(recommend_most_visited) {};
    };
}