#pragma once

#include "thts_manager.h"

#include <cstdlib>
#include <limits>
#include <random>

namespace thts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct UctManagerArgs : public ThtsManagerArgs {
        static constexpr double USE_AUTO_BIAS = -1.0;
        static constexpr double AUTO_BIAS_MIN_BIAS = 0.001;

        static constexpr double bias_default=USE_AUTO_BIAS;
        static const int heuristic_psuedo_trials_default=0;
        static constexpr double epsilon_exploration_default=0.0;
        static const bool recommend_most_visited_default=false;

        double bias;
        int heuristic_psuedo_trials;
        double epsilon_exploration;
        bool recommend_most_visited;

        UctManagerArgs(std::shared_ptr<ThtsEnv> thts_env) :
            ThtsManagerArgs(thts_env),
            bias(bias_default),
            heuristic_psuedo_trials(heuristic_psuedo_trials_default),
            epsilon_exploration(epsilon_exploration_default),
            recommend_most_visited(recommend_most_visited_default) {}

        virtual ~UctManagerArgs() = default;
    };

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
            double epsilon_exploration;
            bool recommend_most_visited;

            UctManager(UctManagerArgs& args) :
                ThtsManager(args),
                bias(args.bias),
                heuristic_psuedo_trials(args.heuristic_psuedo_trials),
                epsilon_exploration(args.epsilon_exploration),
                recommend_most_visited(args.recommend_most_visited) {};

            UctManager(
                std::shared_ptr<ThtsEnv> thts_env,
                int max_depth=UctManagerArgs::max_depth_default,
                double bias=UctManagerArgs::bias_default,
                int heuristic_psuedo_trials=UctManagerArgs::heuristic_psuedo_trials_default,
                HeuristicFnPtr heuristic_fn=helper::zero_heuristic_fn,
                PriorFnPtr prior_fn=nullptr,
                bool mcts_mode=UctManagerArgs::mcts_mode_default, 
                bool use_transposition_table=UctManagerArgs::use_transposition_table_default, 
                int num_transposition_table_mutexes=UctManagerArgs::num_transposition_table_mutexes_default,
                bool is_two_player_game=UctManagerArgs::is_two_player_game_default,
                double epsilon_exploration=UctManagerArgs::epsilon_exploration_default,
                bool recommend_most_visited=UctManagerArgs::recommend_most_visited_default,
                int seed=UctManagerArgs::seed_default) :
                    ThtsManager(
                        thts_env,
                        max_depth,
                        heuristic_fn,
                        prior_fn,
                        mcts_mode,
                        is_two_player_game,
                        use_transposition_table,
                        num_transposition_table_mutexes,
                        seed),
                    bias(bias),
                    heuristic_psuedo_trials(heuristic_psuedo_trials),
                    epsilon_exploration(epsilon_exploration),
                    recommend_most_visited(recommend_most_visited) {};
    };
}