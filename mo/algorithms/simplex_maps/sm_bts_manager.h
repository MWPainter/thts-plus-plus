#pragma once

#include "mo/algorithms/simplex_maps/sm_manager.h"
#include "algorithms/common/decaying_temp.h"


namespace thts {
    
    // Forward declare
    class MoThtsEnv;

    /**
     * Args object so that params can be set in a more named args way
     */
    struct SmBtsManagerArgs : public SmThtsManagerArgs {
        static constexpr double temp_default=1.0;
        static constexpr double epsilon_default=0.5;
        static constexpr double root_node_epsilon_default=0.5;
        static constexpr double max_explore_prob_default=1.0;

        std::shared_ptr<Schedule> temp_schedule_ptr;
        double epsilon;
        double root_node_epsilon;
        double max_explore_prob;

        SmBtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env, Eigen::ArrayXd default_q_value) :
            SmThtsManagerArgs(thts_env, default_q_value),
            temp_schedule_ptr(std::make_shared<ConstSchedule>(temp_default)),
            epsilon(epsilon_default),
            root_node_epsilon(root_node_epsilon_default),
            max_explore_prob(max_explore_prob_default)
        {
        }

        virtual ~SmBtsManagerArgs() = default;
    };
    
    /**
     * ThtsManager + stuff for multi objective environments
     * 
     * Member variables (environment):
     *      bias:
     *          The CZT bias
     */
    class SmBtsManager : public SmThtsManager {
        public:
            std::shared_ptr<Schedule> temp_schedule_ptr;
            double epsilon;
            double root_node_epsilon;
            double max_explore_prob;

            /**
             * Constructor.
             */    
            SmBtsManager(const SmBtsManagerArgs& args) : 
                SmThtsManager(args),
                temp_schedule_ptr(args.temp_schedule_ptr),
                epsilon(args.epsilon),
                root_node_epsilon(args.root_node_epsilon),
                max_explore_prob(args.max_explore_prob)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmBtsManager() = default;
            
    };
}