#pragma once

#include "thts_manager.h"

namespace thts {
    /**
     * Args object so that params can be set in a more named args way
     */
    struct MentsManagerArgs : public ThtsManagerArgs {
        static constexpr double temp_default=1.0;
        static constexpr double default_q_value_default=0.0;
        static constexpr double epsilon_default=0.5;
        static constexpr double max_explore_prob_default=0.5;
        static const int recommend_visit_threshold_default=0;
        static constexpr double prior_policy_boost_default=0.0;

        double temp;
        double default_q_value;
        double epsilon;
        double max_explore_prob;
        int recommend_visit_threshold;
        double prior_policy_boost;

        MentsManagerArgs(std::shared_ptr<ThtsEnv> thts_env) :
            ThtsManagerArgs(thts_env),
            temp(temp_default),
            default_q_value(default_q_value_default),
            epsilon(epsilon_default),
            max_explore_prob(max_explore_prob_default),
            recommend_visit_threshold(recommend_visit_threshold_default),
            prior_policy_boost(prior_policy_boost_default) {}

        virtual ~MentsManagerArgs() = default;
    };

    /**
     * A specific instance of ThtsManager for MENTS algorithms.
     * 
     * Options:
     *      using_heuristic_function (heuristic_fn != nullptr):
     *          Uses the heurisitic function to initialise the soft_value of nodes.
     *      thresholded_recommend (recommend_visit_threshold > 0):
     *          Used to provide a bit more stability to reommendations, similar to recommending the most visited child 
     *          node in uct. When the threshold is positive, we recommend the child with largest soft_value that has 
     *          been visited at least 'recommend_visit_threshold' many times. If no node has been visited at least 
     *          'recommend_visit_threshold' many times, then we recommend the best soft_value (as if this value 
     *          was zero).
     * 
     * Member variables:
     *      temp:
     *          The temperature to use in maximum entropy (soft) backups 
     *      default_q_value:
     *          A default value to use for q-values, typically 0 in a reward maximisation setting. An example of this 
     *          being useful is in cost minimisation settings, where values are non-positive, in which initialising 
     *          q-values to 0.0 leads to running a BFS, which is not the intended operation.
     *      epsilon:
     *          The epsilon exploration parameters from MENTS. I.e. MENTS will uniformly randomly sample an action with 
     *          probability 'epsilon_exploration / log(num_visits+1)' (assuming its a valid probability!).
     *      max_explore_prob:
     *          In MENTS action selection an action is uniformly randomly sampled with probability 
     *          'epsilon_exploration / log(num_visits+1)'. This value provides a maximum probability of uniformly 
     *          exploring. This value must be in the range [0,1].
     *      recommend_visit_threshold:
     *          A minimum number of visits required to recommend an action. We can use this to force recommendations 
     *          to have a minimum number of samples before its a candidate for recommendation.
     *      prior_policy_boost:
     *          For a given state action pair (s,a), when we have policy_prior pi(a|s), the 'psuedo q-value' is set to 
     *          'log(pi(a|s)) + prior_policy_boost'. This doesn't change the distribution when there are zero children 
     *          at a decision node, but when a decision node has children, it will change the relative weight of 
     *          actions that do and dont have a child node created.
     */
    class MentsManager : public ThtsManager {
        public:
            double temp;
            double default_q_value;
            double epsilon;
            double max_explore_prob;
            int recommend_visit_threshold;
            double prior_policy_boost;

            MentsManager(MentsManagerArgs& args) :
                ThtsManager(args),
                temp(args.temp),
                default_q_value(args.default_q_value),
                epsilon(args.epsilon),
                recommend_visit_threshold(args.recommend_visit_threshold) {};

            MentsManager(
                std::shared_ptr<ThtsEnv> thts_env,
                int max_depth=MentsManagerArgs::max_depth_default,
                double temp=MentsManagerArgs::temp_default,
                double default_q_value=MentsManagerArgs::default_q_value_default,
                HeuristicFnPtr heuristic_fn=nullptr,
                double prior_policy_boost=MentsManagerArgs::prior_policy_boost_default,
                PriorFnPtr prior_fn=nullptr,
                bool mcts_mode=MentsManagerArgs::mcts_mode_default, 
                bool is_two_player_game=MentsManagerArgs::is_two_player_game_default,
                bool use_transposition_table=MentsManagerArgs::use_transposition_table_default, 
                int num_transposition_table_mutexes=MentsManagerArgs::num_transposition_table_mutexes_default,
                double epsilon=MentsManagerArgs::epsilon_default,
                double max_explore_prob=MentsManagerArgs::max_explore_prob_default,
                bool recommend_visit_threshold=MentsManagerArgs::recommend_visit_threshold_default,
                int seed=MentsManagerArgs::seed_default) :
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
                    temp(temp),
                    default_q_value(default_q_value),
                    epsilon(epsilon),
                    max_explore_prob(max_explore_prob),
                    recommend_visit_threshold(recommend_visit_threshold),
                    prior_policy_boost(prior_policy_boost) {};
    };
}