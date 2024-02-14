#pragma once

#include "mo/smt_manager.h"
#include "algorithms/common/decaying_temp.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct SmtBtsManagerArgs : public SmtThtsManagerArgs {
        static constexpr double temp_default=1.0;
        static constexpr double epsilon_default=0.5;
        static constexpr double root_node_epsilon_default=0.5;
        static constexpr double max_explore_prob_default=1.0;

        static constexpr TempDecayFnPtr temp_decay_fn_default=nullptr;
        static constexpr double temp_decay_min_temp_default=1.0e-6;
        static constexpr double temp_decay_visits_scale_default=1.0;

        double temp; 
        double epsilon;
        double root_node_epsilon;
        double max_explore_prob;

        TempDecayFnPtr temp_decay_fn;
        double temp_decay_min_temp;
        double temp_decay_visits_scale;

        SmtBtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env, Eigen::ArrayXd default_q_value) :
            SmtThtsManagerArgs(thts_env, default_q_value),
            temp(temp_default),
            epsilon(epsilon_default),
            root_node_epsilon(root_node_epsilon_default),
            max_explore_prob(max_explore_prob_default),
            temp_decay_fn(temp_decay_fn_default),
            temp_decay_min_temp(temp_decay_min_temp_default),
            temp_decay_visits_scale(temp_decay_visits_scale_default)
        {
        }

        virtual ~SmtBtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      bias:
     *          The CZT bias
     */
    class SmtBtsManager : public SmtThtsManager {
        public:
            double temp; 
            double epsilon;
            double root_node_epsilon;
            double max_explore_prob;

            TempDecayFnPtr temp_decay_fn;
            double temp_decay_min_temp;
            double temp_decay_visits_scale;

            /**
             * Constructor.
             */    
            SmtBtsManager(const SmtBtsManagerArgs& args) : 
                SmtThtsManager(args),
                temp(args.temp),
                epsilon(args.epsilon),
                root_node_epsilon(args.root_node_epsilon),
                max_explore_prob(args.max_explore_prob),
                temp_decay_fn(args.temp_decay_fn),
                temp_decay_min_temp(args.temp_decay_min_temp),
                temp_decay_visits_scale(args.temp_decay_visits_scale)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmtBtsManager() = default;
            
    };
}