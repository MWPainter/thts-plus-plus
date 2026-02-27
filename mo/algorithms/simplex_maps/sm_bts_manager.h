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
        static constexpr double max_explore_prob_default=1.0;

        static constexpr double default_q_utility_default=0.0;

        std::shared_ptr<Schedule> temp_schedule_ptr;
        double epsilon;
        double max_explore_prob;

        double default_q_utility;

        SmBtsManagerArgs(std::shared_ptr<MoThtsEnv> thts_env) :
            SmThtsManagerArgs(thts_env),
            temp_schedule_ptr(std::make_shared<ConstSchedule>(temp_default)),
            epsilon(epsilon_default),
            max_explore_prob(max_explore_prob_default),
            default_q_utility(default_q_utility_default)
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
            double max_explore_prob;
            double default_q_utility;

            /**
             * Constructor.
             */    
            SmBtsManager(const SmBtsManagerArgs& args) : 
                SmThtsManager(args),
                temp_schedule_ptr(args.temp_schedule_ptr),
                epsilon(args.epsilon),
                max_explore_prob(args.max_explore_prob),
                default_q_utility(args.default_q_utility)
            {
            }

            /**
             * Any classes intended to be inherited from should make destructor virtual
             */
            virtual ~SmBtsManager() = default;
            
    };
}