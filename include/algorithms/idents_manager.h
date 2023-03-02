#pragma once

#include "dents_manager.h"

namespace thts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct IDentsManagerArgs : public DentsManagerArgs {
        static constexpr double search_temp_default=1.0;

        double search_temp;

        IDentsManagerArgs(std::shared_ptr<ThtsEnv> thts_env) :
            DentsManagerArgs(thts_env),
            search_temp(search_temp_default) {}

        virtual ~IDentsManagerArgs() = default;
    };

    /**
     * A specific instance of ThtsManager for I(mproved)DENTS algorithms.
     * 
     * Member variables:
     *      search_temp: The temperature to use locally for searching
     */
    class IDentsManager : public DentsManager {
        public:
            double search_temp;

            IDentsManager(IDentsManagerArgs& args) :
                DentsManager(args),
                search_temp(args.search_temp) {};

            IDentsManager(
                std::shared_ptr<ThtsEnv> thts_env,
                int max_depth=IDentsManagerArgs::max_depth_default,
                double temp=IDentsManagerArgs::temp_default,
                double min_temp=IDentsManagerArgs::min_temp_default,
                double search_temp=IDentsManagerArgs::search_temp_default,
                double default_q_value=IDentsManagerArgs::default_q_value_default,
                HeuristicFnPtr heuristic_fn=nullptr,
                bool use_prior_shift=IDentsManagerArgs::use_prior_shift_default,
                double prior_policy_boost=IDentsManagerArgs::prior_policy_boost_default,
                double prior_policy_search_weight=IDentsManagerArgs::prior_policy_search_weight_default,
                PriorFnPtr prior_fn=nullptr,
                bool mcts_mode=IDentsManagerArgs::mcts_mode_default, 
                bool is_two_player_game=IDentsManagerArgs::is_two_player_game_default,
                bool use_transposition_table=IDentsManagerArgs::use_transposition_table_default, 
                int num_transposition_table_mutexes=IDentsManagerArgs::num_transposition_table_mutexes_default,
                double epsilon=IDentsManagerArgs::epsilon_default,
                double root_node_extra_epsilon=IDentsManagerArgs::root_node_extra_epsilon_default,
                double max_explore_prob=IDentsManagerArgs::max_explore_prob_default,
                bool recommend_visit_threshold=IDentsManagerArgs::recommend_visit_threshold_default,
                int seed=IDentsManagerArgs::seed_default) :
                    DentsManager(
                        thts_env,
                        max_depth,
                        temp,
                        min_temp,
                        default_q_value,
                        heuristic_fn,
                        use_prior_shift,
                        prior_policy_boost,
                        prior_policy_search_weight,
                        prior_fn,
                        mcts_mode,
                        is_two_player_game,
                        use_transposition_table,
                        num_transposition_table_mutexes,
                        epsilon,
                        root_node_extra_epsilon,
                        max_explore_prob,
                        recommend_visit_threshold,
                        seed), 
                    search_temp(search_temp) {};
    };
}